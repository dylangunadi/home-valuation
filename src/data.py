"""Data loading, validation, and preprocessing for SoCal housing dataset."""

import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# Required CSV columns
REQUIRED_COLS = ['image_id', 'n_citi', 'bed', 'bath', 'sqft', 'price']
FEATURE_COLS = ['n_citi', 'bed', 'bath', 'sqft']
TARGET_COL = 'price'


def load_csv(csv_path: str) -> pd.DataFrame:
    """
    Load and validate CSV file with required columns.
    
    Args:
        csv_path: Path to CSV file
        
    Returns:
        Validated DataFrame
        
    Raises:
        FileNotFoundError: If CSV doesn't exist
        ValueError: If required columns missing or data invalid
    """
    csv_path = Path(csv_path)
    
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    print(f"✓ Loaded {len(df):,} rows from {csv_path.name}")
    
    # Check required columns
    missing_cols = set(REQUIRED_COLS) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Check for NaN values in required columns
    nan_counts = df[REQUIRED_COLS].isna().sum()
    if nan_counts.any():
        raise ValueError(f"Found NaN values:\n{nan_counts[nan_counts > 0]}")
    
    # Check for duplicates in image_id
    duplicates = df['image_id'].duplicated()
    if duplicates.any():
        dup_ids = df['image_id'][duplicates].tolist()
        raise ValueError(f"Duplicate image_ids found: {dup_ids[:5]}...")
    
    # Validate data types and ranges
    for col in ['bed', 'bath', 'sqft', 'price']:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise ValueError(f"Column {col} must be numeric")
        if (df[col] < 0).any():
            raise ValueError(f"Column {col} contains negative values")
    
    if (df['price'] <= 0).any():
        raise ValueError("Price must be positive")
    
    print(f"  Columns: {list(df.columns)}")
    print(f"  Price range: ${df['price'].min():,.0f} - ${df['price'].max():,.0f}")
    
    return df


def map_images_by_stem(img_dir: str) -> Dict[str, str]:
    """
    Map image stems to full paths (case-insensitive extensions).
    
    Args:
        img_dir: Directory containing images
        
    Returns:
        Dictionary mapping stem -> full path
        
    Raises:
        ValueError: If directory doesn't exist or is empty
    """
    img_dir = Path(img_dir)
    
    if not img_dir.exists():
        raise ValueError(f"Image directory not found: {img_dir}")
    
    # Case-insensitive extensions
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
    
    image_map = {}
    for ext in extensions:
        for img_path in img_dir.glob(ext):
            stem = img_path.stem
            if stem in image_map:
                print(f"  Warning: Duplicate stem '{stem}' - keeping first occurrence")
            else:
                image_map[stem] = str(img_path)
    
    if not image_map:
        raise ValueError(f"No images found in {img_dir}")
    
    print(f"✓ Found {len(image_map):,} images in {img_dir.name}/")
    return image_map


def split_train_val_using_dirs(
    df: pd.DataFrame,
    train_dir: str,
    val_dir: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split DataFrame into train/val based on image directory location.
    
    Args:
        df: Full DataFrame with image_id column
        train_dir: Directory containing training images
        val_dir: Directory containing validation images
        
    Returns:
        Tuple of (train_df, val_df) with 'image_path' column added
        
    Raises:
        ValueError: If directories overlap or no valid samples found
    """
    print("\n" + "="*70)
    print("SPLITTING TRAIN/VAL BY IMAGE DIRECTORIES")
    print("="*70)
    
    # Map images from both directories
    train_image_map = map_images_by_stem(train_dir)
    val_image_map = map_images_by_stem(val_dir)
    
    # Check for overlap between directories
    train_stems = set(train_image_map.keys())
    val_stems = set(val_image_map.keys())
    overlap = train_stems & val_stems
    
    if overlap:
        raise ValueError(
            f"Train/val overlap detected: {len(overlap)} images in both directories. "
            f"Examples: {list(overlap)[:5]}"
        )
    
    print(f"✓ No overlap between train/val directories")
    
    # Convert image_id to string for matching
    df = df.copy()
    df['image_id'] = df['image_id'].astype(str)
    
    # Match to train directory
    train_mask = df['image_id'].isin(train_stems)
    train_df = df[train_mask].copy()
    train_df['image_path'] = train_df['image_id'].map(train_image_map)
    
    # Match to val directory
    val_mask = df['image_id'].isin(val_stems)
    val_df = df[val_mask].copy()
    val_df['image_path'] = val_df['image_id'].map(val_image_map)
    
    # Check for missing images
    df_ids = set(df['image_id'])
    missing_images = df_ids - train_stems - val_stems
    
    if missing_images:
        print(f"⚠ Warning: {len(missing_images):,} image_ids have no image in either directory")
        print(f"  Examples: {list(missing_images)[:5]}")
    
    # Check for images without CSV rows
    extra_train = train_stems - df_ids
    extra_val = val_stems - df_ids
    
    if extra_train:
        print(f"⚠ Warning: {len(extra_train):,} images in {Path(train_dir).name}/ not in CSV")
    if extra_val:
        print(f"⚠ Warning: {len(extra_val):,} images in {Path(val_dir).name}/ not in CSV")
    
    # Drop any rows with missing image paths
    train_df = train_df.dropna(subset=['image_path']).reset_index(drop=True)
    val_df = val_df.dropna(subset=['image_path']).reset_index(drop=True)
    
    print(f"\n📊 Final Split:")
    print(f"  Train: {len(train_df):,} samples")
    print(f"  Val:   {len(val_df):,} samples")
    print(f"  Total: {len(train_df) + len(val_df):,} samples")
    print(f"  Dropped: {len(df) - len(train_df) - len(val_df):,} samples (missing images)")
    
    if len(train_df) == 0:
        raise ValueError("No training samples! Check that image_ids match image filenames")
    if len(val_df) == 0:
        raise ValueError("No validation samples! Check that image_ids match image filenames")
    
    print("="*70 + "\n")
    
    return train_df, val_df


def get_tabular_and_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Extract tabular features and target from DataFrame.
    
    Args:
        df: DataFrame with feature and target columns
        
    Returns:
        Tuple of (X_df, y_series)
        
    Raises:
        ValueError: If required columns missing
    """
    missing_features = set(FEATURE_COLS) - set(df.columns)
    if missing_features:
        raise ValueError(f"Missing feature columns: {missing_features}")
    
    if TARGET_COL not in df.columns:
        raise ValueError(f"Missing target column: {TARGET_COL}")
    
    X = df[FEATURE_COLS].copy()
    y = df[TARGET_COL].copy()
    
    return X, y


def preprocess_tabular(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame
) -> Tuple[ColumnTransformer, np.ndarray, np.ndarray]:
    """
    Preprocess tabular features: OneHotEncode n_citi, StandardScale numeric features.
    
    Args:
        X_train: Training features DataFrame
        X_val: Validation features DataFrame
        
    Returns:
        Tuple of (fitted_preprocessor, X_train_transformed, X_val_transformed)
        
    Raises:
        ValueError: If required columns missing
    """
    print("\n" + "="*70)
    print("PREPROCESSING TABULAR FEATURES")
    print("="*70)
    
    # Validate columns
    for X_df, name in [(X_train, 'train'), (X_val, 'val')]:
        missing = set(FEATURE_COLS) - set(X_df.columns)
        if missing:
            raise ValueError(f"Missing columns in {name}: {missing}")
    
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
    
    # Get feature names
    cat_encoder = preprocessor.named_transformers_['cat']
    cat_features = cat_encoder.get_feature_names_out(categorical_features)
    all_feature_names = list(cat_features) + numerical_features
    
    print(f"✓ Preprocessing complete")
    print(f"  Categorical (OneHot): {categorical_features}")
    print(f"    → {len(cat_features)} encoded features")
    print(f"  Numerical (Scaled): {numerical_features}")
    print(f"  Total features: {X_train_transformed.shape[1]}")
    print(f"  Train shape: {X_train_transformed.shape}")
    print(f"  Val shape: {X_val_transformed.shape}")
    print("="*70 + "\n")
    
    return preprocessor, X_train_transformed, X_val_transformed


def safe_log_target(y: pd.Series) -> np.ndarray:
    """
    Apply log1p transform to target (handles zeros safely).
    
    Args:
        y: Target series
        
    Returns:
        Log-transformed array
    """
    return np.log1p(y.values)


def inverse_log_target(y_log: np.ndarray) -> np.ndarray:
    """
    Inverse log1p transform to get original scale.
    
    Args:
        y_log: Log-transformed values
        
    Returns:
        Original scale values
    """
    return np.expm1(y_log)


def main():
    """CLI for processing and splitting data."""
    parser = argparse.ArgumentParser(
        description="Load, validate, and split SoCal housing data"
    )
    parser.add_argument(
        '--csv',
        type=str,
        required=True,
        help='Path to input CSV file'
    )
    parser.add_argument(
        '--train_dir',
        type=str,
        required=True,
        help='Path to training images directory'
    )
    parser.add_argument(
        '--val_dir',
        type=str,
        required=True,
        help='Path to validation images directory'
    )
    parser.add_argument(
        '--out_dir',
        type=str,
        required=True,
        help='Output directory for processed CSVs'
    )
    
    args = parser.parse_args()
    
    # Create output directory
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*70)
    print("SOCAL HOUSING DATA PROCESSING")
    print("="*70)
    print(f"Input CSV: {args.csv}")
    print(f"Train dir: {args.train_dir}")
    print(f"Val dir: {args.val_dir}")
    print(f"Output dir: {args.out_dir}")
    print("="*70 + "\n")
    
    try:
        # Load CSV
        df = load_csv(args.csv)
        original_count = len(df)
        
        # Split by directories
        train_df, val_df = split_train_val_using_dirs(
            df, args.train_dir, args.val_dir
        )
        
        # Save processed CSVs
        train_csv_path = out_dir / 'train.csv'
        val_csv_path = out_dir / 'val.csv'
        
        train_df.to_csv(train_csv_path, index=False)
        val_df.to_csv(val_csv_path, index=False)
        
        print(f"✓ Saved {train_csv_path}")
        print(f"✓ Saved {val_csv_path}")
        
        # Create summary
        summary_path = out_dir / 'summary.txt'
        dropped = original_count - len(train_df) - len(val_df)
        
        summary = f"""SoCal Housing Data Processing Summary
{'='*70}

Input:
  CSV file: {args.csv}
  Total rows: {original_count:,}

Output:
  Train samples: {len(train_df):,}
  Val samples: {len(val_df):,}
  Total kept: {len(train_df) + len(val_df):,}
  Dropped (missing images): {dropped:,}

Train Statistics:
  Price range: ${train_df['price'].min():,.0f} - ${train_df['price'].max():,.0f}
  Mean price: ${train_df['price'].mean():,.0f}
  Median price: ${train_df['price'].median():,.0f}

Val Statistics:
  Price range: ${val_df['price'].min():,.0f} - ${val_df['price'].max():,.0f}
  Mean price: ${val_df['price'].mean():,.0f}
  Median price: ${val_df['price'].median():,.0f}

Feature Columns: {FEATURE_COLS}
Target Column: {TARGET_COL}

Files Generated:
  - {train_csv_path}
  - {val_csv_path}
  - {summary_path}

{'='*70}
Processing completed successfully!
"""
        
        with open(summary_path, 'w') as f:
            f.write(summary)
        
        print(f"✓ Saved {summary_path}")
        print("\n" + "="*70)
        print("✅ DATA PROCESSING COMPLETE!")
        print("="*70 + "\n")
        print(summary)
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
