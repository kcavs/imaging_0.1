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
IMAGE_DIR = Path("C:/Users/kcavalli/OneDrive - South Orangetown Central School District/Image Classification/dogs_cats")
H = 224
W = 224
epochs = 5
batch_size = 100
global model
model=Model(IMAGE_DIR,H,W,batch_size)
def complile():
    model.train()
    model.val()
    model.comp()
