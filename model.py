import tensorflow as tf
import keras
from keras import layers
from keras import models
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
from pathlib import Path
import os
import numpy as np
import cv2
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
            batch_size=self.batch_size,
            label_mode='binary',  
            shuffle=True     
        )

    def val(self):
        global val_ds
        self.val_ds = tf.keras.utils.image_dataset_from_directory(
            directory=str(self.data_dir),
            validation_split=0.2,
            subset="validation",
            seed=self.seed,
            image_size=(self.img_height, self.img_width),
            batch_size=self.batch_size,
            label_mode='binary',  
            shuffle=True     
    )
    
    def comp(self):
        self.keras_model = Sequential([
            layers.Rescaling(1./255),

            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(pool_size=(2, 2)),

            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(pool_size=(2, 2)),

            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(pool_size=(2, 2)),

            layers.Flatten(),
            layers.Dropout(0.5),
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
    
    def grad_cam(self, imgs, intensity=0.5, res=250):
        if self.keras_model is None:
            raise ValueError("The neural network hasn't been compiled yet!")
        last_layer_name = None
        for layer in reversed(self.keras_model.layers):
            if "Conv2D" in layer.__class__.__name__:  # Simple, reliable string check
                last_layer_name = layer.name
                break
                
        if not last_layer_name:
            raise ValueError("Could not find a Conv2D layer in the model.")
        
        last_layer = self.keras_model.get_layer(last_layer_name)

        grad_model = tf.keras.models.Model(
            inputs=[self.keras_model.layers[0].input],  # Access input layer node directly
            outputs=[last_layer.output, self.keras_model.layers[-1].output]  # Explicit final Dense layer output
        )

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(imgs)
            pred_indices = tf.argmax(predictions, axis=-1)
            # Gather the predicted class channel scores for the batch
            batch_indices = tf.range(tf.shape(predictions)[0])
            indices = tf.stack([batch_indices, tf.cast(pred_indices, tf.int32)], axis=1)
            class_channels = tf.gather_nd(predictions, indices)

        # 4. Extract gradients of the top predicted classes w.r.t the last conv layer outputs
        grads = tape.gradient(class_channels, conv_outputs)

        # 5. Global average pooling across height and width for each feature map channel
        pooled_grads = tf.reduce_mean(grads, axis=(1, 2))

        # 6. Set up the matplotlib grid dynamically based on the batch size
        num_imgs = len(imgs)
        fig, axes = plt.subplots(1, num_imgs, figsize=(4 * num_imgs, 4))
        if num_imgs == 1:
            axes = [axes] # Handle single image subplot object edge-case

        # 7. Loop through each image in the batch to overlay heatmaps
        for idx in range(num_imgs):
            # Compute heatmap for this individual image
            img_conv_outputs = conv_outputs[idx]
            img_pooled_grads = pooled_grads[idx]
            
            # Weighted sum of feature channels
            heatmap = img_conv_outputs @ img_pooled_grads[..., tf.newaxis]
            heatmap = tf.squeeze(heatmap)

            # Apply ReLU and normalize
            heatmap = tf.maximum(heatmap, 0)
            max_val = tf.math.reduce_max(heatmap)
            if max_val == 0: 
                max_val = 1e-10 # Avoid division by zero if heatmap is entirely black
            heatmap = (heatmap / max_val).numpy()

            # Process original image for OpenCV overlay
            img_raw = imgs[idx].numpy() if hasattr(imgs[idx], 'numpy') else imgs[idx]
            if img_raw.max() <= 1.0:
                img_raw = (img_raw * 255).astype(np.uint8)
            else:
                img_raw = img_raw.astype(np.uint8)

            # Resize and map color
            heatmap_resized = cv2.resize(heatmap, (img_raw.shape[1], img_raw.shape[0]))
            heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)

            # Blend heatmap overlay with original image
            superimposed_img = heatmap_color * intensity + img_raw
            superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)

            # Render to subplot
            axes[idx].imshow(cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB))
            axes[idx].set_title(f"Img {idx+1} (Pred: {pred_indices[idx].numpy()})", fontsize=10)
            axes[idx].axis('off')

        plt.tight_layout()
        plt.show()