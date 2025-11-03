#!/usr/bin/env python3
"""Quick test to verify your setup before training."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config import DATA_CSV, TRAIN_IMAGES_DIR, VAL_IMAGES_DIR
from src.data import split_train_val_using_dirs

def main():
    print("\n" + "=" * 70)
    print("QUICK SETUP VERIFICATION")
    print("=" * 70)
    
    print(f"\n📍 Checking paths:")
    print(f"  CSV: {DATA_CSV}")
    print(f"  Train images: {TRAIN_IMAGES_DIR}")
    print(f"  Val images: {VAL_IMAGES_DIR}")
    
    # Check CSV exists
    if not DATA_CSV.exists():
        print(f"\n❌ ERROR: CSV not found at {DATA_CSV}")
        print("   Please make sure socal2_cleaned_mod.csv is in the data/ folder")
        return
    
    print(f"\n✅ CSV found!")
    
    # Check image folders exist
    if not TRAIN_IMAGES_DIR.exists():
        print(f"\n❌ ERROR: {TRAIN_IMAGES_DIR} not found")
        print("   Please create data/train_images/ and add your training images")
        return
    
    if not VAL_IMAGES_DIR.exists():
        print(f"\n❌ ERROR: {VAL_IMAGES_DIR} not found")
        print("   Please create data/val_images/ and add your validation images")
        return
    
    print(f"✅ Image folders found!")
    
    # Try loading and splitting
    print(f"\n🔍 Loading and splitting data...")
    try:
        train_df, val_df, train_img_map, val_img_map = split_train_val_using_dirs(
            DATA_CSV, TRAIN_IMAGES_DIR, VAL_IMAGES_DIR
        )
        
        print(f"\n✅ SUCCESS! Your setup is ready!")
        print(f"\n📊 Summary:")
        print(f"  Total CSV rows: {len(train_df) + len(val_df)}")
        print(f"  Train samples: {len(train_df)}")
        print(f"  Val samples: {len(val_df)}")
        print(f"  Train images found: {len(train_img_map)}")
        print(f"  Val images found: {len(val_img_map)}")
        
        print(f"\n🚀 Next steps:")
        print(f"  1. Run: make baseline")
        print(f"  2. Run: make embed")
        print(f"  3. Run: make fusion")
        print(f"  4. Run: make app")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        print(f"\nPlease check:")
        print(f"  • Image files are in train_images/ and val_images/")
        print(f"  • Image filenames match image_id in CSV")
        print(f"  • Example: if image_id=1, file should be 1.jpg or 1.png")
        return
    
    print(f"\n{'=' * 70}\n")

if __name__ == "__main__":
    main()
