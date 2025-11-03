"""Tabular baseline models for SoCal housing price prediction."""

import argparse
import joblib
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor


def load_data(train_path: str, val_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load train and validation CSVs.
    
    Args:
        train_path: Path to training CSV
        val_path: Path to validation CSV
        
    Returns:
        Tuple of (train_df, val_df)
        
    Raises:
        FileNotFoundError: If CSVs don't exist
    """
    train_path = Path(train_path)
    val_path = Path(val_path)
    
    if not train_path.exists():
        raise FileNotFoundError(f"Train CSV not found: {train_path}")
    if not val_path.exists():
        raise FileNotFoundError(f"Val CSV not found: {val_path}")
    
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    
    print(f"✓ Loaded train: {len(train_df):,} samples")
    print(f"✓ Loaded val: {len(val_df):,} samples")
    
    return train_df, val_df


def preprocess_features(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: list,
    target_col: str = 'price'
) -> Tuple[ColumnTransformer, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Preprocess features and apply log transform to target.
    
    Args:
        train_df: Training DataFrame
        val_df: Validation DataFrame
        feature_cols: List of feature column names
        target_col: Target column name
        
    Returns:
        Tuple of (preprocessor, X_train, X_val, y_train_log, y_val_log)
    """
    print("\n" + "="*70)
    print("PREPROCESSING FEATURES")
    print("="*70)
    
    # Extract features and target
    X_train = train_df[feature_cols].copy()
    X_val = val_df[feature_cols].copy()
    y_train = train_df[target_col].values
    y_val = val_df[target_col].values
    
    # Log transform target
    y_train_log = np.log1p(y_train)
    y_val_log = np.log1p(y_val)
    
    print(f"Features: {feature_cols}")
    print(f"Target: log({target_col})")
    
    # Define preprocessing
    categorical_features = ['n_citi']
    numerical_features = ['bed', 'bath', 'sqft']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features),
            ('num', StandardScaler(), numerical_features)
        ],
        remainder='drop'
    )
    
    # Fit on train, transform both
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_val_transformed = preprocessor.transform(X_val)
    
    # Get feature info
    cat_encoder = preprocessor.named_transformers_['cat']
    n_cat_features = len(cat_encoder.get_feature_names_out(categorical_features))
    
    print(f"\nPreprocessing:")
    print(f"  Categorical (OneHot): {categorical_features} → {n_cat_features} features")
    print(f"  Numerical (Scaled): {numerical_features}")
    print(f"  Total features: {X_train_transformed.shape[1]}")
    
    print(f"\nTarget statistics (log scale):")
    print(f"  Train: mean={y_train_log.mean():.3f}, std={y_train_log.std():.3f}")
    print(f"  Val:   mean={y_val_log.mean():.3f}, std={y_val_log.std():.3f}")
    
    print(f"\nTarget statistics (original scale):")
    print(f"  Train: mean=${y_train.mean():,.0f}, median=${np.median(y_train):,.0f}")
    print(f"  Val:   mean=${y_val.mean():,.0f}, median=${np.median(y_val):,.0f}")
    print("="*70 + "\n")
    
    return preprocessor, X_train_transformed, X_val_transformed, y_train_log, y_val_log


def calculate_metrics(
    y_true: np.ndarray,
    y_pred_log: np.ndarray,
    name: str = "Model"
) -> Dict[str, float]:
    """
    Calculate regression metrics in original price scale.
    
    Args:
        y_true: True prices (original scale)
        y_pred_log: Predicted log prices
        name: Model name for display
        
    Returns:
        Dictionary of metrics
    """
    # Convert predictions back to original scale
    y_pred = np.expm1(y_pred_log)
    
    # Calculate metrics
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # MAPE (handle zeros)
    nonzero_mask = y_true != 0
    if np.any(nonzero_mask):
        mape = np.mean(np.abs((y_true[nonzero_mask] - y_pred[nonzero_mask]) / y_true[nonzero_mask])) * 100
    else:
        mape = np.nan
    
    return {
        'mae': mae,
        'r2': r2,
        'mape': mape
    }


def print_metrics(metrics: Dict[str, float], split: str = "Val"):
    """Print formatted metrics."""
    print(f"  {split:5s}: MAE=${metrics['mae']:>10,.0f} | R²={metrics['r2']:>6.4f} | MAPE={metrics['mape']:>6.2f}%")


def train_models(
    X_train: np.ndarray,
    X_val: np.ndarray,
    y_train_log: np.ndarray,
    y_val_log: np.ndarray,
    y_train_orig: np.ndarray,
    y_val_orig: np.ndarray
) -> Tuple[object, str, Dict[str, Dict[str, float]]]:
    """
    Train XGBoost and LightGBM, return best model by validation MAE.
    
    Args:
        X_train: Preprocessed training features
        X_val: Preprocessed validation features
        y_train_log: Log-transformed training target
        y_val_log: Log-transformed validation target
        y_train_orig: Original scale training target
        y_val_orig: Original scale validation target
        
    Returns:
        Tuple of (best_model, best_model_name, all_metrics)
    """
    print("\n" + "="*70)
    print("TRAINING MODELS")
    print("="*70)
    
    models = {}
    all_metrics = {}
    
    # Train XGBoost
    print("\n[1/2] Training XGBoost...")
    print("  Hyperparameters: n_estimators=600, learning_rate=0.05, max_depth=6")
    
    xgb = XGBRegressor(
        n_estimators=600,
        learning_rate=0.05,
        max_depth=6,
        random_state=42,
        n_jobs=-1,
        verbosity=0
    )
    xgb.fit(X_train, y_train_log)
    
    # XGBoost predictions
    y_train_pred_xgb = xgb.predict(X_train)
    y_val_pred_xgb = xgb.predict(X_val)
    
    # XGBoost metrics
    xgb_train_metrics = calculate_metrics(y_train_orig, y_train_pred_xgb, "XGBoost")
    xgb_val_metrics = calculate_metrics(y_val_orig, y_val_pred_xgb, "XGBoost")
    
    print(f"\n  XGBoost Results:")
    print_metrics(xgb_train_metrics, "Train")
    print_metrics(xgb_val_metrics, "Val")
    
    models['xgboost'] = xgb
    all_metrics['xgboost'] = {'train': xgb_train_metrics, 'val': xgb_val_metrics}
    
    # Train LightGBM
    print(f"\n[2/2] Training LightGBM...")
    print("  Hyperparameters: n_estimators=1000, learning_rate=0.05, num_leaves=63")
    
    lgb = LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=63,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    lgb.fit(X_train, y_train_log)
    
    # LightGBM predictions
    y_train_pred_lgb = lgb.predict(X_train)
    y_val_pred_lgb = lgb.predict(X_val)
    
    # LightGBM metrics
    lgb_train_metrics = calculate_metrics(y_train_orig, y_train_pred_lgb, "LightGBM")
    lgb_val_metrics = calculate_metrics(y_val_orig, y_val_pred_lgb, "LightGBM")
    
    print(f"\n  LightGBM Results:")
    print_metrics(lgb_train_metrics, "Train")
    print_metrics(lgb_val_metrics, "Val")
    
    models['lightgbm'] = lgb
    all_metrics['lightgbm'] = {'train': lgb_train_metrics, 'val': lgb_val_metrics}
    
    # Select best model by validation MAE
    print(f"\n{'─'*70}")
    print("MODEL SELECTION (by validation MAE)")
    print(f"{'─'*70}")
    
    xgb_val_mae = xgb_val_metrics['mae']
    lgb_val_mae = lgb_val_metrics['mae']
    
    if xgb_val_mae < lgb_val_mae:
        best_model = xgb
        best_name = 'xgboost'
        improvement = ((lgb_val_mae - xgb_val_mae) / lgb_val_mae) * 100
        print(f"✓ XGBoost selected (MAE=${xgb_val_mae:,.0f})")
        print(f"  {improvement:.2f}% better than LightGBM")
    else:
        best_model = lgb
        best_name = 'lightgbm'
        improvement = ((xgb_val_mae - lgb_val_mae) / xgb_val_mae) * 100
        print(f"✓ LightGBM selected (MAE=${lgb_val_mae:,.0f})")
        print(f"  {improvement:.2f}% better than XGBoost")
    
    print("="*70 + "\n")
    
    return best_model, best_name, all_metrics


def save_artifacts(
    model: object,
    model_name: str,
    preprocessor: ColumnTransformer,
    metrics: Dict[str, Dict[str, float]],
    models_dir: str
) -> None:
    """
    Save model, preprocessor, and metrics.
    
    Args:
        model: Trained model
        model_name: Name of the model ('xgboost' or 'lightgbm')
        preprocessor: Fitted preprocessor
        metrics: Dictionary of all metrics
        models_dir: Directory to save artifacts
    """
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    
    print("Saving artifacts...")
    
    # Save model
    model_path = models_dir / f'{model_name}_tabular_baseline.pkl'
    joblib.dump(model, model_path)
    print(f"✓ Saved model: {model_path}")
    
    # Save preprocessor
    preprocessor_path = models_dir / 'tabular_preprocessor.pkl'
    joblib.dump(preprocessor, preprocessor_path)
    print(f"✓ Saved preprocessor: {preprocessor_path}")
    
    # Save metrics
    metrics_path = models_dir / 'tabular_baseline_metrics.json'
    import json
    with open(metrics_path, 'w') as f:
        # Convert numpy types to python types
        metrics_serializable = {}
        for model_key, splits in metrics.items():
            metrics_serializable[model_key] = {}
            for split_key, split_metrics in splits.items():
                metrics_serializable[model_key][split_key] = {
                    k: float(v) if not np.isnan(v) else None
                    for k, v in split_metrics.items()
                }
        json.dump(metrics_serializable, f, indent=2)
    print(f"✓ Saved metrics: {metrics_path}")
    
    # Save summary
    summary_path = models_dir / 'tabular_baseline_summary.txt'
    with open(summary_path, 'w') as f:
        f.write("Tabular Baseline Model Summary\n")
        f.write("="*70 + "\n\n")
        f.write(f"Best Model: {model_name.upper()}\n\n")
        
        for model_key in ['xgboost', 'lightgbm']:
            f.write(f"{model_key.upper()}:\n")
            for split in ['train', 'val']:
                m = metrics[model_key][split]
                f.write(f"  {split.capitalize():5s}: MAE=${m['mae']:,.0f} | R²={m['r2']:.4f} | MAPE={m['mape']:.2f}%\n")
            f.write("\n")
    
    print(f"✓ Saved summary: {summary_path}")


def main():
    """Main training pipeline."""
    parser = argparse.ArgumentParser(
        description="Train tabular baseline models (XGBoost vs LightGBM)"
    )
    parser.add_argument(
        '--train',
        type=str,
        required=True,
        help='Path to training CSV'
    )
    parser.add_argument(
        '--val',
        type=str,
        required=True,
        help='Path to validation CSV'
    )
    parser.add_argument(
        '--models_dir',
        type=str,
        default='models',
        help='Directory to save models'
    )
    parser.add_argument(
        '--fig_dir',
        type=str,
        default='reports/figures',
        help='Directory to save figures (not used in this script)'
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("TABULAR BASELINE TRAINING")
    print("="*70)
    print(f"Train: {args.train}")
    print(f"Val: {args.val}")
    print(f"Models dir: {args.models_dir}")
    print("="*70 + "\n")
    
    try:
        # Load data
        train_df, val_df = load_data(args.train, args.val)
        
        # Define features
        feature_cols = ['n_citi', 'bed', 'bath', 'sqft']
        target_col = 'price'
        
        # Preprocess
        preprocessor, X_train, X_val, y_train_log, y_val_log = preprocess_features(
            train_df, val_df, feature_cols, target_col
        )
        
        # Get original targets for metric calculation
        y_train_orig = train_df[target_col].values
        y_val_orig = val_df[target_col].values
        
        # Train models
        best_model, best_name, all_metrics = train_models(
            X_train, X_val, y_train_log, y_val_log, y_train_orig, y_val_orig
        )
        
        # Save artifacts
        save_artifacts(best_model, best_name, preprocessor, all_metrics, args.models_dir)
        
        print("\n" + "="*70)
        print("✅ TRAINING COMPLETE!")
        print("="*70)
        print(f"\nBest model: {best_name.upper()}")
        print(f"Validation MAE: ${all_metrics[best_name]['val']['mae']:,.0f}")
        print(f"Validation R²: {all_metrics[best_name]['val']['r2']:.4f}")
        print(f"\nArtifacts saved to: {args.models_dir}/")
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
