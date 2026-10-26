import tensorflow as tf
import keras
import matplotlib.pyplot as plt
from pathlib import Path
from model import Model
import os
import numpy as np

IMAGE_DIR = Path('/Users/kcavalli/Library/CloudStorage/OneDrive-SouthOrangetownCentralSchoolDistrict/datasets')
H = 224
W = 224
batch_size = 32

global model
model = Model(IMAGE_DIR, H, W, batch_size)

model.comp()

model_weight_path = 'lung_cancer.keras'
if os.path.exists(model_weight_path):
    print("Loading pre-trained model weights from disk...")
    model.keras_model = tf.keras.models.load_model(model_weight_path, safe_mode=False)
else:
    raise FileNotFoundError(f"Missing weights file at {model_weight_path}. Run train.py first!")

test_path = Path(model.data_dir) / 'test'
test_dataset = tf.keras.utils.image_dataset_from_directory(
    str(test_path),
    image_size=(H, W), 
    batch_size=batch_size,         
    label_mode='categorical',
    shuffle=True,          
)

categories = sorted(test_dataset.class_names)
model.class_names = categories 
print(f"Verified App Class Order Evaluation: {model.class_names}")

for img_batch, lbl_batch in test_dataset.take(1):
    images = img_batch[:5]       
    true_labels = lbl_batch[:5]   
    break

predictions = model.predict(tf.convert_to_tensor(images))
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
