"""Fusion model combining tabular features and image embeddings."""

import argparse
import joblib
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_csv', required=True)
    parser.add_argument('--val_csv', required=True)
    parser.add_argument('--train_embeddings', required=True)
    parser.add_argument('--val_embeddings', required=True)
    parser.add_argument('--preprocessor', required=True)
    parser.add_argument('--models_dir', default='models')
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("FUSION MODEL TRAINING")
    print("="*70)
    
    # Load data
    train_df = pd.read_csv(args.train_csv)
    val_df = pd.read_csv(args.val_csv)
    
    print(f"Train: {len(train_df):,} samples")
    print(f"Val: {len(val_df):,} samples")
    
    # Load embeddings
    train_emb = np.load(args.train_embeddings)
    val_emb = np.load(args.val_embeddings)
    
    print(f"Train embeddings: {train_emb.shape}")
    print(f"Val embeddings: {val_emb.shape}")
    
    # Load preprocessor and process tabular features
    preprocessor = joblib.load(args.preprocessor)
    
    feature_cols = ['n_citi', 'bed', 'bath', 'sqft']
    X_train_tab = preprocessor.transform(train_df[feature_cols])
    X_val_tab = preprocessor.transform(val_df[feature_cols])
    
    # Concatenate tabular + embeddings
    X_train_fusion = np.hstack([X_train_tab, train_emb])
    X_val_fusion = np.hstack([X_val_tab, val_emb])
    
    print(f"\nFusion features:")
    print(f"  Tabular: {X_train_tab.shape[1]}")
    print(f"  Embeddings: {train_emb.shape[1]}")
    print(f"  Total: {X_train_fusion.shape[1]}")
    
    # Targets
    y_train = train_df['price'].values
    y_val = val_df['price'].values
    y_train_log = np.log1p(y_train)
    y_val_log = np.log1p(y_val)
    
    # Train fusion model
    print("\nTraining XGBoost fusion model...")
    print("  n_estimators=600, learning_rate=0.05, max_depth=6")
    
    model = XGBRegressor(
        n_estimators=600,
        learning_rate=0.05,
        max_depth=6,
        random_state=42,
        n_jobs=-1,
        verbosity=0
    )
    model.fit(X_train_fusion, y_train_log)
    
    # Predictions
    y_train_pred_log = model.predict(X_train_fusion)
    y_val_pred_log = model.predict(X_val_fusion)
    
    y_train_pred = np.expm1(y_train_pred_log)
    y_val_pred = np.expm1(y_val_pred_log)
    
    # Metrics
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    val_mae = mean_absolute_error(y_val, y_val_pred)
    val_r2 = r2_score(y_val, y_val_pred)
    
    print(f"\nResults:")
    print(f"  Train: MAE=${train_mae:,.0f} | R²={train_r2:.4f}")
    print(f"  Val:   MAE=${val_mae:,.0f} | R²={val_r2:.4f}")
    
    # Save
    models_dir = Path(args.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = models_dir / 'xgboost_fusion.pkl'
    joblib.dump(model, model_path)
    print(f"\n✓ Saved model: {model_path}")
    
    metrics = {
        'train': {'mae': float(train_mae), 'r2': float(train_r2)},
        'val': {'mae': float(val_mae), 'r2': float(val_r2)}
    }
    
    with open(models_dir / 'fusion_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
