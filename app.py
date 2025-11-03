"""Simple Streamlit app for SoCal Housing Price Prediction."""

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path
from PIL import Image

# Page config
st.set_page_config(
    page_title="SoCal Housing Predictor",
    page_icon="��",
    layout="wide"
)

# Paths
DATA_DIR = Path("data")
MODELS_DIR = Path("models")

@st.cache_data
def load_data():
    """Load validation data and fix image paths."""
    val_df = pd.read_csv(DATA_DIR / "csv" / "val.csv")
    
    # Fix image paths - extract just the filename and rebuild path
    def fix_image_path(old_path):
        filename = Path(old_path).name
        # Try val_images first
        new_path = DATA_DIR / "val_images" / filename
        if new_path.exists():
            return str(new_path)
        # Try train_images as fallback
        new_path = DATA_DIR / "train_images" / filename
        if new_path.exists():
            return str(new_path)
        return old_path
    
    val_df['image_path'] = val_df['image_path'].apply(fix_image_path)
    return val_df

@st.cache_resource
def load_model():
    """Load trained model and preprocessor."""
    model = joblib.load(MODELS_DIR / "xgboost_tabular_baseline.pkl")
    preprocessor = joblib.load(MODELS_DIR / "tabular_preprocessor.pkl")
    return model, preprocessor

# Main app
st.title("🏠 SoCal Housing Price Predictor")
st.markdown("**XGBoost Model - Tabular Features**")
st.divider()

# Load data and model
try:
    val_df = load_data()
    model, preprocessor = load_model()
except Exception as e:
    st.error(f"Error loading data/models: {e}")
    st.stop()

# Sidebar
st.sidebar.header("🔍 Select Property")

idx = st.sidebar.selectbox(
    "Property Index",
    range(len(val_df)),
    format_func=lambda x: f"Property {x} - ${val_df.iloc[x]['price']:,.0f}"
)

property_data = val_df.iloc[idx]

# Property details
st.sidebar.divider()
st.sidebar.subheader("📋 Property Details")
st.sidebar.metric("Cities", f"{property_data['n_citi']:.0f}")
st.sidebar.metric("Bedrooms", f"{property_data['bed']:.0f}")
st.sidebar.metric("Bathrooms", f"{property_data['bath']:.0f}")
st.sidebar.metric("Square Feet", f"{property_data['sqft']:,.0f}")
st.sidebar.metric("💰 Actual Price", f"${property_data['price']:,.0f}")

# Main content
col1, col2 = st.columns([1, 1])

# Left - Image
with col1:
    st.subheader("🏡 Property Image")
    try:
        img_path = property_data['image_path']
        if Path(img_path).exists():
            img = Image.open(img_path)
            st.image(img, use_container_width=True)
        else:
            st.error(f"Image not found: {Path(img_path).name}")
            st.caption(f"Looking for: {img_path}")
    except Exception as e:
        st.error(f"Error loading image: {e}")

# Right - Prediction
with col2:
    st.subheader("🎯 Price Prediction")
    
    # Prepare features
    feature_cols = ['n_citi', 'bed', 'bath', 'sqft']
    X_sample = property_data[feature_cols].values.reshape(1, -1)
    X_sample_df = pd.DataFrame(X_sample, columns=feature_cols)
    
    # Preprocess and predict
    X_processed = preprocessor.transform(X_sample_df)
    pred_log = model.predict(X_processed)[0]
    pred_price = np.expm1(pred_log)
    
    actual_price = property_data['price']
    error_pct = ((pred_price - actual_price) / actual_price) * 100
    
    st.metric(
        "📊 XGBoost Prediction",
        f"${pred_price:,.0f}",
        f"{error_pct:+.1f}%",
        delta_color="inverse"
    )
    
    st.divider()
    
    # Feature table
    st.subheader("📈 Features")
    feature_df = pd.DataFrame({
        'Feature': ['Cities', 'Bedrooms', 'Bathrooms', 'Sq Ft'],
        'Value': [
            property_data['n_citi'],
            property_data['bed'],
            property_data['bath'],
            property_data['sqft']
        ]
    })
    st.dataframe(feature_df, hide_index=True, use_container_width=True)

# Footer
st.divider()
st.caption("SoCal Housing Price Prediction System")
