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

test_dataset = tf.keras.utils.image_dataset_from_directory(
    model.data_dir,
    batch_size=4,
    shuffle=False 
)

num_images=5
categories = test_dataset.class_names
run_dataset=(
    test_dataset.unbatch()
    .shuffle(buffer_size=1000)
    .take(num_images)
)

eval_steps=1
images=[]
true_labels=[]
for img, lbl in run_dataset:
    images.append(img.numpy())
    true_labels.append(lbl.numpy())
images=np.array(images)
true_labels=np.array(true_labels)
images = tf.image.resize(images, (224,224))

predictions = model.predict(images, steps=eval_steps) 
predicted_classes=np.argmax(predictions, axis=-1).flatten().astype(int)
true_labels = true_labels.flatten().astype(int)

print("\n--- AI Model Prediction Results ---")
num_matches = min(len(predicted_classes), len(true_labels))
for i in range(num_matches):
    p_idx=predicted_classes[i]
    true_idx = true_labels[i]
    t_idx=int(true_idx)
    print(f"Image {i+1}:")
    print(f"  Predicted Class: {categories[p_idx]}")
    print(f"  Actual Class:    {categories[t_idx]}")
    print("-" * 30)

model.grad_cam(images)