"""Extract image embeddings using simple image features."""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from tqdm import tqdm


def extract_image_features(img_path: str, output_dim: int = 128) -> np.ndarray:
    """
    Extract simple image features.
    
    Args:
        img_path: Path to image
        output_dim: Output feature dimension
        
    Returns:
        Feature vector
    """
    try:
        img = Image.open(img_path).convert('RGB')
        img = img.resize((224, 224))
        img_array = np.array(img) / 255.0
        
        features = []
        
        # Channel statistics
        for c in range(3):
            features.extend([
                img_array[:,:,c].mean(),
                img_array[:,:,c].std(),
                img_array[:,:,c].min(),
                img_array[:,:,c].max(),
            ])
        
        # Histogram features
        for c in range(3):
            hist, _ = np.histogram(img_array[:,:,c], bins=16, range=(0, 1))
            features.extend(hist / (hist.sum() + 1e-10))
        
        # Spatial grid features
        grid_size = 4
        h, w = img_array.shape[:2]
        h_step, w_step = h // grid_size, w // grid_size
        for i in range(grid_size):
            for j in range(grid_size):
                patch = img_array[i*h_step:(i+1)*h_step, j*w_step:(j+1)*w_step]
                features.append(patch.mean())
        
        features = np.array(features)
        
        # Pad or truncate
        if len(features) < output_dim:
            features = np.pad(features, (0, output_dim - len(features)))
        else:
            features = features[:output_dim]
        
        return features
        
    except Exception as e:
        print(f"  Error: {img_path}: {e}")
        return np.zeros(output_dim)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_csv', required=True)
    parser.add_argument('--val_csv', required=True)
    parser.add_argument('--output_dir', default='data/embeddings')
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("IMAGE FEATURE EXTRACTION")
    print("="*70)
    
    # Load data
    train_df = pd.read_csv(args.train_csv)
    val_df = pd.read_csv(args.val_csv)
    
    print(f"Train: {len(train_df):,} samples")
    print(f"Val: {len(val_df):,} samples")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract train
    print("\nExtracting train features...")
    train_features = []
    for img_path in tqdm(train_df['image_path']):
        train_features.append(extract_image_features(img_path))
    train_features = np.array(train_features)
    
    # Extract val
    print("Extracting val features...")
    val_features = []
    for img_path in tqdm(val_df['image_path']):
        val_features.append(extract_image_features(img_path))
    val_features = np.array(val_features)
    
    # Save
    np.save(output_dir / 'train_embeddings.npy', train_features)
    np.save(output_dir / 'val_embeddings.npy', val_features)
    
    print(f"\n✓ Train shape: {train_features.shape}")
    print(f"✓ Val shape: {val_features.shape}")
    print(f"✓ Saved to: {output_dir}/")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
