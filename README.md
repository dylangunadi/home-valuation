# home-valuation
# SoCal Housing Price Prediction

Multi-modal machine learning system for predicting Southern California residential property prices using tabular features and computer vision.

## Performance

| Model | Validation MAE | Validation R² | Validation MAPE |
|-------|----------------|---------------|-----------------|
| **XGBoost (Tabular)** | **$233,987** | **0.354** | **27.65%** |
| LightGBM (Tabular) | $253,910 | 0.274 | 30.06% |
| XGBoost (Fusion)* | ~$180,000 | ~0.55 | ~20% |

*Fusion model combines tabular features with image embeddings (expected performance)

### Dataset
- **Total Properties**: 12,518
- **Training Set**: 1,530 properties with images
- **Validation Set**: 1,000 properties with images
- **Price Range**: $195,000 - $2,000,000
- **Mean Price**: $587,879 (train), $935,659 (val)

---

## Features

### Tabular Features
- `n_citi`: Neighborhood/city identifier (one-hot encoded, 150 categories)
- `bed`: Number of bedrooms (standardized)
- `bath`: Number of bathrooms (standardized)
- `sqft`: Square footage (standardized)

### Image Features
- Extracted using simple statistical features (128-D vectors)
- Color histograms, spatial grid statistics, channel moments

---

## Technical Stack

**Machine Learning:**
- XGBoost 3.1.1
- LightGBM 4.6.0
- scikit-learn 1.7.2

**Data Processing:**
- pandas, numpy

**Visualization:**
- Streamlit 1.51.0

**Environment:**
- Python 3.12+

---

## Installation

### 1. Clone Repository
```bash
git clone https://github.com/yourusername/socal-housing.git
cd socal-housing
```

### 2. Install Dependencies
```bash
pip install pandas numpy scikit-learn xgboost lightgbm joblib pillow streamlit tqdm
```

### 3. Verify Data Structure
```
socal-housing/
├── data/
│   ├── csv/
│   │   ├── train.csv
│   │   └── val.csv
│   ├── train_images/
│   └── val_images/
├── models/
│   ├── xgboost_tabular_baseline.pkl
│   └── tabular_preprocessor.pkl
└── app.py
```

---

## Usage

### Interactive Demo
```bash
streamlit run app.py
```

Opens browser at `http://localhost:8501` with property image viewer, real-time predictions, and error analysis.

### Command Line Training

**Data Processing:**
```bash
python -m src.data \
  --csv data/socal2_cleaned_mod.csv \
  --train_dir data/train_images \
  --val_dir data/val_images \
  --out_dir data/csv
```

**Train Tabular Baseline:**
```bash
python -m src.tabular_baseline \
  --train data/csv/train.csv \
  --val data/csv/val.csv \
  --models_dir models
```

**Extract Image Embeddings:**
```bash
python -m src.embeddings \
  --train_csv data/csv/train.csv \
  --val_csv data/csv/val.csv \
  --output_dir data/embeddings
```

**Train Fusion Model:**
```bash
python -m src.fusion_train \
  --train_csv data/csv/train.csv \
  --val_csv data/csv/val.csv \
  --train_embeddings data/embeddings/train_embeddings.npy \
  --val_embeddings data/embeddings/val_embeddings.npy \
  --preprocessor models/tabular_preprocessor.pkl \
  --models_dir models
```

---

## Methodology

### 1. Data Preprocessing
- One-hot encoding for categorical features (n_citi)
- Standard scaling for numerical features (bed, bath, sqft)
- Log transformation of target variable (price)

### 2. Baseline Models
- **XGBoost**: 600 estimators, learning_rate=0.05, max_depth=6
- **LightGBM**: 1000 estimators, learning_rate=0.05, num_leaves=63
- Model selection based on validation MAE

### 3. Image Processing
- Resize to 224x224
- Extract statistical features: channel-wise stats, histograms, spatial grids
- 128-dimensional feature vectors

### 4. Fusion Architecture
- Concatenate preprocessed tabular features with image embeddings
- Train XGBoost on combined feature space (153 tabular + 128 image features)

---

## Results Analysis

### Model Comparison
XGBoost outperformed LightGBM by 7.85% on validation MAE.

### Error Distribution
- Median absolute error: ~$180,000
- Model underpredicts high-value properties (>$1M)
- Best performance on properties in $400K-$800K range

### Feature Importance
1. Square footage (sqft)
2. Number of bathrooms (bath)
3. Neighborhood identifier (n_citi)
4. Number of bedrooms (bed)

---

## Limitations

1. **Dataset Size**: 2,530 properties with images (20% of full dataset)
2. **Geographic Scope**: Limited to Southern California
3. **Image Quality**: Variable resolution and composition
4. **Feature Engineering**: Simple statistical features vs deep learning embeddings
5. **Temporal Factors**: No time-series modeling

---

## Future Improvements

- [ ] Implement ResNet50/EfficientNet for image embeddings
- [ ] Add attention mechanisms for image regions
- [ ] Hyperparameter optimization using Optuna
- [ ] Add temporal features (days on market, seasonality)
- [ ] Incorporate location data (distance to amenities)
- [ ] Containerize with Docker
- [ ] Implement REST API with FastAPI
- [ ] Add MLflow for experiment tracking

---

## Contact

**Author**: [Your Name]  
**Email**: your.email@example.com  
**LinkedIn**: [linkedin.com/in/yourprofile](https://linkedin.com/in/yourprofile)  
**GitHub**: [github.com/yourusername](https://github.com/yourusername)

---

## License

MIT License
