import streamlit as st
import tensorflow as tf
import matplotlib.pyplot as plt
from pathlib import Path
import os
import numpy as np
from model import Model, categorical_focal_loss

# 1. Page Configuration
st.set_page_config(page_title="Lung Cancer Classifier & Grad-CAM", layout="wide")
st.title("Lung Cancer Classification & Grad-CAM Visualization")

# Constants
IMAGE_DIR = Path('/Users/kcavalli/Library/CloudStorage/OneDrive-SouthOrangetownCentralSchoolDistrict/datasets')
H, W = 224, 224
BATCH_SIZE = 32
MODEL_WEIGHT_PATH = 'lung_cancer.keras'

# 2. Cache the Model (Prevents reloading on every button click)
@st.cache_resource
def load_trained_model():
    model = Model(IMAGE_DIR, H, W, BATCH_SIZE)
    model.comp()
    
    if os.path.exists(MODEL_WEIGHT_PATH):
        custom_loss = categorical_focal_loss(gamma=2.5, alpha=1.0)
        model.keras_model = tf.keras.models.load_model(
            MODEL_WEIGHT_PATH,
            custom_objects={'focal_loss_fixed': custom_loss},
            safe_mode=False
        )
    else:
        raise FileNotFoundError(f"Missing weights file at {MODEL_WEIGHT_PATH}. Run train.py first!")
    
    return model

# Load model with a loading spinner
try:
    with st.spinner("Loading pre-trained model weights..."):
        model = load_trained_model()
except Exception as e:
    st.error(str(e))
    st.stop()

# 3. Load Test Dataset
test_path = Path(model.data_dir) / 'test'
test_dataset = tf.keras.utils.image_dataset_from_directory(
    str(test_path),
    image_size=(H, W),
    batch_size=BATCH_SIZE,
    label_mode='categorical',
    shuffle=True,
)
categories = sorted(test_dataset.class_names)
model.class_names = categories

st.sidebar.header("Controls")
st.sidebar.write("**Verified Class Order:**", categories)

# Button to grab a new random batch of 10 images
if st.sidebar.button("Sample New Test Batch (10 Images)", type="primary") or "images" not in st.session_state:
    for img_batch, lbl_batch in test_dataset.take(1):
        st.session_state.images = img_batch[:10]
        st.session_state.true_labels = lbl_batch[:10]
        break

images = st.session_state.images
true_labels = st.session_state.true_labels

# 4. Run Predictions
predictions = model.predict(tf.convert_to_tensor(images))
predicted_classes = np.argmax(predictions, axis=-1)
actual_classes = np.argmax(true_labels.numpy(), axis=-1)

# 5. Display Text Predictions in a Clean Grid
st.subheader("AI Model Prediction Results")
cols = st.columns(5)  # 5 columns x 2 rows = 10 images

for i in range(len(predicted_classes)):
    p_idx = predicted_classes[i]
    t_idx = actual_classes[i]
    pred_name = categories[p_idx]
    actual_name = categories[t_idx]
    
    with cols[i % 5]:
        st.markdown(f"**Image {i+1}**")
        if pred_name == actual_name:
            st.success(f"Pred: {pred_name}\n\nActual: {actual_name}")
        else:
            st.error(f"Pred: {pred_name}\n\nActual: {actual_name}")

st.divider()

# 6. Display Grad-CAM Results
st.subheader("Grad-CAM Heatmap Visualizations")
with st.spinner("Generating Grad-CAM heatmaps..."):
    # Clear any old plots in memory
    plt.close('all')
    
    # Run your Grad-CAM function
    fig = model.grad_cam(images)
    
    # Fallback in case model.grad_cam() creates a figure without returning it
    if fig is None:
        fig = plt.gcf()
        
    st.pyplot(fig)
