import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
from pathlib import Path
print("Tensorflow version: ", tf.__version__)
IMAGE_DIR = Path("C:/Users/kcavalli/OneDrive - South Orangetown Central School District/Image Classification/dogs_cats")
H = 224
W = 224
epochs = 5
batch_size = 100
seeds = 42
from model import dataset
dataset(IMAGE_DIR,H,W,batch_size)