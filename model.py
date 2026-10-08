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
        self.data_dir= dirD
        self.img_height= H
        self.img_width = W
        self.batch_size = size

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
        class_names = self.train_ds.class_names
        print("Found classes:", class_names)

        # 3. Normalize pixel values from [0, 255] to [0, 1]
        normalization_layer = layers.Rescaling(1./255)
        train_ds = self.train_ds.map(lambda x, y: (normalization_layer(x), y))
        val_ds = self.val_ds.map(lambda x, y: (normalization_layer(x), y))

        # 4. Build the Convolutional Neural Network (CNN) model
        model = models.Sequential([
            layers.Conv2D(16, (3, 3), activation='relu', input_shape=(self.img_height, self.img_width, 3)),
            layers.MaxPooling2D(),
            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.MaxPooling2D(),
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dense(len(class_names), activation='softmax')  # Output layer for classes
        ])

        # 5. Compile the model
        model.compile(
            optimizer='adam',
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=['accuracy']
        )

        # 6. Train the model
        epochs = 5
        history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)

        # 7. Save the trained model
        model.save("demo_model.keras")
        print("Model training complete and saved!")
