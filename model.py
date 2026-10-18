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
from keras import mixed_precision
from sklearn.utils.class_weight import compute_class_weight
mixed_precision.set_global_policy('mixed_float16')

class Model:
    seed=42
    def __init__(self, dirD, H, W, size):
        self.data_dir= dirD
        self.img_height= H
        self.img_width = W
        self.batch_size = size
        self.keras_model= None

    def train(self):
        self.train_ds = tf.keras.utils.image_dataset_from_directory(
            directory=str(self.data_dir),
            validation_split=0.2,
            subset="training",
            seed=self.seed,
            image_size=(self.img_height, self.img_width),
            batch_size=self.batch_size,
            label_mode='categorical',  
            shuffle=True     
        )
        
        self.train_ds = self.train_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
        
    def val(self):
        self.val_ds = tf.keras.utils.image_dataset_from_directory(
            directory=str(self.data_dir),
            validation_split=0.2,
            subset="validation",
            seed=self.seed,
            image_size=(self.img_height, self.img_width),
            batch_size=self.batch_size,
            label_mode='categorical',  
            shuffle=False     
        )
        self.val_ds = self.val_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    
    def comp(self):
        # Explicitly define a stable optimizer with a lower learning rate
        stable_adam = tf.keras.optimizers.Adam(learning_rate=1e-4)

        self.keras_model = Sequential([
            layers.Input(shape=(self.img_height, self.img_width, 3)),
            layers.Rescaling(1./255),
            
            # 1. Moderated Data Augmentation
            layers.RandomFlip("horizontal_and_vertical"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),

            # Conv Block 1
            layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(pool_size=(2, 2)),

            # Conv Block 2
            layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(pool_size=(2, 2)),

            # Conv Block 3
            layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(pool_size=(2, 2)),

            # Classification Head
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            
            # 2. FIX: Structural Separation for Mixed Precision
            layers.Dense(3, name='logits'), 
            layers.Activation('softmax', dtype='float32', name='softmax_output') 
        ])

        # 3. FIX: Switch CategoricalCrossentropy to from_logits=False since we use an explicit Softmax layer
        self.keras_model.compile(
            optimizer=stable_adam, 
            loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
            metrics=['accuracy']
        )


    def predict(self, data, steps=1):
        if self.keras_model is None:
            raise ValueError("The neural network hasn't been compiled yet. Run model.comp() first!")
        return self.keras_model.predict(data, steps=steps)
    
    def fit(self, epochs):
        if self.keras_model is None:
            raise ValueError("The neural network hasn't been compiled yet. Run model.comp() first!")

        print("Calculating class weights instantly from folder names...")
        
        class_names = sorted([f.name for f in Path(self.data_dir).iterdir() if f.is_dir()])
        
        y_train = []
        
        for class_idx, class_name in enumerate(class_names):
            class_path = Path(self.data_dir) / class_name
           
            num_images = len(list(class_path.glob('*')))
            y_train.extend([class_idx] * num_images)
            
        y_train = np.array(y_train)

        classes = np.unique(y_train)
        weights = compute_class_weight(
            class_weight='balanced', 
            classes=classes, 
            y=y_train
        )
        class_weight_dict = dict(zip(classes, weights))
        print(f"Applied Class Weights: {class_weight_dict}")

        history = self.keras_model.fit(
            self.train_ds,
            validation_data=self.val_ds,
            epochs=epochs,
            class_weight=class_weight_dict  
        )
        return history
    
    def grad_cam(self, imgs, intensity=0.5, res=250):
        if self.keras_model is None:
            raise ValueError("The neural network hasn't been compiled yet!")
        last_layer_name = None
        for layer in reversed(self.keras_model.layers):
            if "Conv2D" in layer.__class__.__name__: 
                last_layer_name = layer.name
                break
                
        if not last_layer_name:
            raise ValueError("Could not find a Conv2D layer in the model.")
        
        last_layer = self.keras_model.get_layer(last_layer_name)

        grad_model = tf.keras.models.Model(
            inputs=[self.keras_model.layers[0].input], 
            outputs=[last_layer.output, self.keras_model.get_layer('logits').output]  
        )

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(imgs)
            pred_indices = tf.argmax(predictions, axis=-1)
            batch_indices = tf.range(tf.shape(predictions)[0])
            indices = tf.stack([batch_indices, tf.cast(pred_indices, tf.int32)], axis=1)
            class_channels = tf.gather_nd(predictions, indices)

        grads = tape.gradient(class_channels, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(1, 2))

        num_imgs = len(imgs)
        fig, axes = plt.subplots(1, num_imgs, figsize=(4 * num_imgs, 4))
        if num_imgs == 1:
            axes = [axes] 

        for idx in range(num_imgs):
            img_conv_outputs = conv_outputs[idx]
            img_pooled_grads = pooled_grads[idx]
            
            heatmap = img_conv_outputs @ img_pooled_grads[..., tf.newaxis]
            heatmap = tf.squeeze(heatmap)

            heatmap = tf.maximum(heatmap, 0)
            max_val = tf.math.reduce_max(heatmap)
            if max_val == 0: 
                max_val = 1e-10 
           
            heatmap = (heatmap / max_val).numpy()
            heatmap = heatmap.astype(np.float32)  

            heatmap = (heatmap / max_val).numpy()
            heatmap = heatmap.astype(np.float32)  
            
            img_raw = imgs[idx].numpy() if hasattr(imgs[idx], 'numpy') else imgs[idx]
            if img_raw.max() <= 1.0:
                img_raw = (img_raw * 255).astype(np.uint8)
            else:
                img_raw = img_raw.astype(np.uint8)

            height, width = img_raw.shape[0], img_raw.shape[1]
            heatmap_resized = cv2.resize(heatmap, (width, height))
            
            heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)

            superimposed_img = heatmap_color * intensity + img_raw
            superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)

            axes[idx].imshow(cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB))
            axes[idx].set_title(f"Img {idx+1} (Pred: {pred_indices[idx]})", fontsize=10)
            axes[idx].axis('off')

        plt.tight_layout()
        plt.show()