import tensorflow as tf
import keras
import matplotlib.pyplot as plt
from pathlib import Path
from model import Model
import os
import numpy as np
from train import model

test_dataset = tf.keras.utils.image_dataset_from_directory(
    str(Path(model.data_dir)/'test'),
    image_size=(224, 224), 
    batch_size=32,         
    label_mode='categorical',
    shuffle=True,          
    seed=model.seed
)

categories = test_dataset.class_names

for img_batch, lbl_batch in test_dataset.take(1):
    images = img_batch[:5]       
    true_labels = lbl_batch[:5]   
    break

predictions = model.predict(images) 

predicted_classes = np.argmax(predictions, axis=-1)
actual_classes = np.argmax(true_labels.numpy(), axis=-1)

print("\n--- AI Model Prediction Results ---")
for i in range(len(predicted_classes)):
    p_idx = predicted_classes[i]
    t_idx = actual_classes[i]
    
    print(f"Image {i+1}:")
    print(f"  Predicted Class: {categories[p_idx]}")
    print(f"  Actual Class:    {categories[t_idx]}")
    print("-" * 30)

model.grad_cam(images)