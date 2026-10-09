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
# 2. Run predict on the dataset
predictions = model.predict(test_dataset, steps=1)
categories = ["Cat", "Dog"]  # Match your class labels
predicted_classes = np.argmax(
    predictions, axis=1
)  # Or use int(prediction) for binary

for pred_idx in predicted_classes:
    print(f"Predicted Class: {categories[pred_idx]}")