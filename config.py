import tensorflow as tf
import keras
from keras import layers
from keras import models
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
from pathlib import Path
import os
class DataConfig:
    data_dir=None
    img_height, img_width = None, None
    batch_size = None
    seed=42
    # 1. Set path to your extracted dataset folder
    def __init__(self,DIR,H,W,size):
        self.data_dir = DIR
        self.img_height, self.img_width = H, W
        self.batch_size = size
