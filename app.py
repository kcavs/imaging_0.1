import tensorflow as tf
import keras
from keras import layers
from keras import models
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
from pathlib import Path
from model import Model
import os
import numpy as np
from train import model
# 1. Load your directory as a tf.data.Dataset
test_dataset = tf.keras.utils.image_dataset_from_directory(
    model.data_dir,
    batch_size=4,
    shuffle=False
)
categories = test_dataset.class_names
# 2. Run predict on the dataset
predictions = model.predict(test_dataset, steps=1) 
 # Match your class labels
predicted_classes = (predictions > 0.5).astype("int32").flatten()
true_labels = np.concatenate([y for x, y in test_dataset.take(1)], axis=0)

print("\n--- AI Model Prediction Results ---")
for i, pred_idx in enumerate(predicted_classes):
    true_idx = true_labels[i]
    
    print(f"Image {i+1}:")
    print(f"  Predicted Class: {categories[pred_idx]}")
    print(f"  Actual Class:    {categories[true_idx]}")
    if predictions.shape[-1] > 1:
        print(f"  Confidence:      {predictions[i][pred_idx]:.2%}")
    else:
        conf = predictions[i][0] if pred_idx == 1 else 1 - predictions[i][0]
        print(f"  Confidence:      {conf:.2%}")
    print("-" * 30)