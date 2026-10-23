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

mixed_precision.set_global_policy('float32')

class Model:
    seed=42
    def __init__(self, dirD, H, W, size):
        self.data_dir= dirD
        self.img_height= H
        self.img_width = W
        self.batch_size = size
        self.keras_model= None

    def train(self):
        train_path=Path(self.data_dir)/'train'
        raw_train_ds = tf.keras.utils.image_dataset_from_directory(
            directory=str(train_path),
            seed=self.seed,
            image_size=(self.img_height, self.img_width),
            batch_size=self.batch_size,
            label_mode='categorical',  
            shuffle=True     
        )
        self.class_names = raw_train_ds.class_names
        print(f"Found {len(self.class_names)} target classes: {self.class_names}")
        if len(self.class_names) != 3:
            raise ValueError(f"CRITICAL: Found {len(self.class_names)} folders instead of 3! Check for hidden folders.")
        self.train_ds = raw_train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
        
    def val(self):
        val_path = Path(self.data_dir) / 'val' 
        raw_val_ds = tf.keras.utils.image_dataset_from_directory( 
            directory=str(val_path), 
            seed=self.seed, 
            image_size=(self.img_height, self.img_width), 
            batch_size=self.batch_size, 
            label_mode='categorical', 
            shuffle=True 
        ) 
        val_classes = raw_val_ds.class_names
        if len(val_classes) != 3:
            raise ValueError(f"CRITICAL: Validation found {len(val_classes)} folders instead of 3!")

        self.val_ds = raw_val_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    def comp(self):
        from keras import mixed_precision
        mixed_precision.set_global_policy('float32')
        ultra_stable_adam = tf.keras.optimizers.Adam(learning_rate=5e-5)
        crop_h=int(self.img_height*0.85)
        crop_w=int(self.img_width*0.85)

        inputs = layers.Input(shape=(self.img_height, self.img_width, 3), name='input_layer')

        x = layers.CenterCrop(height=crop_h, width=crop_w)(inputs)
        x = layers.RandomFlip("horizontal")(x)
        x = layers.RandomRotation(0.02)(x)
        x = layers.Rescaling(1./255)(x)

        x1 = layers.Conv2D(32, (3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(x)
        x1 = layers.BatchNormalization()(x1)
        x1 = layers.Conv2D(32, (3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(x1)
        x1 = layers.BatchNormalization()(x1)
        shortcut1 = layers.Conv2D(32, (1, 1), padding='same',kernel_initializer='he_normal')(x)
        x = layers.add([x1, shortcut1])
        x = layers.MaxPooling2D(pool_size=(2, 2))(x)

        x2 = layers.Conv2D(64, (3, 3), padding='same', activation='relu',kernel_initializer='he_normal')(x)
        x2 = layers.BatchNormalization()(x2)
        x2 = layers.Conv2D(64, (3, 3), padding='same', activation='relu',kernel_initializer='he_normal')(x2)
        x2 = layers.BatchNormalization()(x2)
        shortcut2 = layers.Conv2D(64, (1, 1), padding='same',kernel_initializer='he_normal')(x)
        x = layers.add([x2, shortcut2])
        x = layers.MaxPooling2D(pool_size=(2, 2))(x)

        x3 = layers.Conv2D(128, (3, 3), padding='same', activation='relu',kernel_initializer='he_normal')(x)
        x3 = layers.BatchNormalization()(x3)
        x3 = layers.Conv2D(128, (3, 3), padding='same', activation='relu',kernel_initializer='he_normal')(x3)
        x3 = layers.BatchNormalization()(x3)
        shortcut3 = layers.Conv2D(128, (1, 1), padding='same',kernel_initializer='he_normal')(x)
        x = layers.add([x3, shortcut3])
        x = layers.MaxPooling2D(pool_size=(2, 2))(x)

        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        
        logits = layers.Dense(3, name='logits', kernel_initializer='glorot_uniform')(x)
        scaled_logits = layers.Lambda(lambda tensor: tensor / 1.5)(logits)
        outputs = layers.Activation('softmax', dtype='float32', name='softmax_output')(scaled_logits)

        self.keras_model = tf.keras.models.Model(inputs=inputs, outputs=outputs)

        stable_loss = tf.keras.losses.CategoricalCrossentropy(
            from_logits=False,
            label_smoothing=0.02
        )
        
        self.keras_model.compile( 
            optimizer=ultra_stable_adam, 
            loss=stable_loss, 
            metrics=['accuracy'] 
        )


    def predict(self, data, steps=1):
        if self.keras_model is None:
            raise ValueError("The neural network hasn't been compiled yet. Run model.comp() first!")
        return self.keras_model.predict(data, steps=steps)
    
    def fit(self, epochs):
        if self.keras_model is None:
            raise ValueError("The neural network hasn't been compiled yet. Run model.comp() first!")

        print("Extracting exact alphanumeric class mapping...")
        class_names = self.class_names
        print(f"Keras Class Mapping Order: {class_names}")

        class_counts = []
        for class_name in class_names:
            class_path = Path(self.data_dir) / "train" / class_name
            num_images = len(list(class_path.glob('*')))
            class_counts.append(num_images)
            
            
        total_samples = sum(class_counts)
        num_classes = len(class_names)

        class_weight_dict = {}
        for class_idx, count in enumerate(class_counts):
            if count > 0:
                weight = total_samples / (num_classes * count)
            else:
                weight = 1.0  
            class_weight_dict[int(class_idx)] = float(weight)

        print(f"Perfected Class Weights (Native Call): {class_weight_dict}")

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
            inputs=[self.keras_model.input], 
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