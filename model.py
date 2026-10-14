import tensorflow as tf
import keras
from keras import layers
from keras import models
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
from pathlib import Path
import os
class Model:
    seed=42
    def __init__(self, dirD, H, W, size):
        global data_dir
        global img_height
        global img_width
        global batch_size
        global keras_model
        self.data_dir= dirD
        self.img_height= H
        self.img_width = W
        self.batch_size = size
        self.keras_model= None
    # 1. Set path to your extracted dataset folder

    # 2. Load training data (80%) and validation data (20%)
    def train(self):
        global train_ds
        self.train_ds = tf.keras.utils.image_dataset_from_directory(
            directory=str(self.data_dir),
            validation_split=0.2,
            subset="training",
            seed=self.seed,
            image_size=(self.img_height, self.img_width),
            batch_size=self.batch_size
        )

    def val(self):
        global val_ds
        self.val_ds = tf.keras.utils.image_dataset_from_directory(
            directory=str(self.data_dir),
            validation_split=0.2,
            subset="validation",
            seed=self.seed,
            image_size=(self.img_height, self.img_width),
            batch_size=self.batch_size
    )
    
    def comp(self):
        self.keras_model = Sequential([
            layers.Rescaling(1./255, input_shape=(self.img_height, self.img_width, 3)),
            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Dense(2) 
        ])
        self.keras_model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy'] )

    # ADD THIS MISSING METHOD TO FIX YOUR ERROR
    def predict(self, data, steps=1):
        if self.keras_model is None:
            raise ValueError("The neural network hasn't been compiled yet. Run model.comp() first!")
        return self.keras_model.predict(data, steps=steps)
    
    def fit(self, epochs):
        if self.keras_model is None:
            raise ValueError("The neural network hasn't been compiled yet. Run model.comp() first!")
            
        # This trains the model and outputs the live accuracy metrics
        history = self.keras_model.fit(
            self.train_ds,
            validation_data=self.val_ds,
            epochs=epochs
        )
        return history

