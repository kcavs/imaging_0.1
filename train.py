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

print("Tensorflow version: ", tf.__version__)
IMAGE_DIR= Path('/Users/kcavalli/Library/CloudStorage/OneDrive-SouthOrangetownCentralSchoolDistrict/datasets')
H = 224
W = 224
epochs = 5
batch_size = 32
global model
model=Model(IMAGE_DIR,H,W,batch_size)
model.train()
model.val()
model.comp()
model.fit(40)
print("Saving trained model weights...")
model.keras_model.save('lung_cancer.keras')
print("Model saved successfully!")