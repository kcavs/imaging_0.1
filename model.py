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
from keras.callbacks import EarlyStopping, ModelCheckpoint


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

        callbacks = (
            EarlyStopping(
                monitor='val_loss',
                patience=20,            
                restore_best_weights=True, 
                verbose=1
            ),
            ModelCheckpoint(
                filepath='lung_cancer.keras',
                monitor='val_loss',
                save_best_only=True,    
                verbose=1
            )
        )
        history = self.keras_model.fit(
            self.train_ds,
            validation_data=self.val_ds,
            epochs=epochs,          
            class_weight=class_weight_dict,
            callbacks=callbacks        
        )
        return history
    def grad_cam(self, imgs, intensity=0.5, res=224):
        if self.keras_model is None:
            raise ValueError("The neural network hasn't been compiled yet!")
        
        # 1. Target the actual combined output of the last residual block
        target_layer_name = None
        for layer in reversed(self.keras_model.layers):
            if "add" in layer.name.lower():
                target_layer_name = layer.name
                break
                
        if not target_layer_name:
            raise ValueError("Could not find the final addition layer (shortcut) in the model.")
            
        conv_layer = self.keras_model.get_layer(target_layer_name)
        logits_layer = self.keras_model.get_layer('logits')

          # 2. Build the gradient model using the exact internal symbolic input tensor
        grad_model = tf.keras.models.Model(
            inputs=self.keras_model.input,  # Removed the brackets [] to pass the single structural tensor
            outputs=[conv_layer.output, logits_layer.output]
        )

    # 3. Explicitly convert and ensure the batch matches the structural layer layout
        if isinstance(imgs, tf.Tensor):
            imgs_tensor = imgs
        else:
            imgs_tensor = tf.convert_to_tensor(imgs, dtype=tf.float32)


        # 3. CRITICAL: Use training=False to disable RandomFlip/RandomRotation during Grad-CAM!
        with tf.GradientTape() as tape:
            conv_outputs, logits_predictions = grad_model(imgs_tensor, training=False)
            pred_indices = tf.argmax(logits_predictions, axis=-1)
            
            batch_indices = tf.range(tf.shape(logits_predictions)[0])
            indices = tf.stack([batch_indices, tf.cast(pred_indices, tf.int32)], axis=1)
            class_channels = tf.gather_nd(logits_predictions, indices)

        # 4. Compute gradients
        grads = tape.gradient(class_channels, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(1, 2))

        num_imgs = len(imgs)
        fig, axes = plt.subplots(1, num_imgs, figsize=(4 * num_imgs, 4))
        if num_imgs == 1:
            axes = [axes]

          # Calculate the exact crop coordinates to match model.comp()
        crop_h = int(self.img_height * 0.85)
        crop_w = int(self.img_width * 0.85)
        start_y = (self.img_height - crop_h) // 2
        end_y = start_y + crop_h
        start_x = (self.img_width - crop_w) // 2
        end_x = start_x + crop_w

        for idx in range(num_imgs):
            img_conv_outputs = conv_outputs[idx]
            img_pooled_grads = pooled_grads[idx]

            # Element-wise multiply features by weights, then sum along channel axis
            heatmap = tf.reduce_sum(img_conv_outputs * img_pooled_grads, axis=-1)
            heatmap = tf.maximum(heatmap, 0)
            
            max_val = tf.math.reduce_max(heatmap)
            if max_val > 0:
                heatmap /= max_val
                
            heatmap = heatmap.numpy()

            # Handle and prepare the original image format
            original_img = imgs[idx].numpy() if isinstance(imgs[idx], tf.Tensor) else imgs[idx]
            
            # CRITICAL FIX: Crop the original image to match the model's receptive field
            original_img = original_img[start_y:end_y, start_x:end_x, :]

            # Standardize image range to 0-255
            if original_img.max() <= 1.0:
                original_img = np.uint8(255 * original_img)
            else:
                original_img = np.uint8(original_img)
                
            # Now both components share the exact same spatial bounding box
            original_img = cv2.resize(original_img, (res, res))
            heatmap = cv2.resize(heatmap, (res, res))
            heatmap = np.uint8(255 * heatmap)
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

            # Superimpose the maps together
            superimposed_img = cv2.addWeighted(original_img, 1 - intensity, heatmap, intensity, 0)
            
            # Display the result
            axes[idx].imshow(cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB))
            axes[idx].axis('off')
            
            predicted_index_value = int(pred_indices[idx].numpy())
            axes[idx].set_title(f"Pred: {self.class_names[predicted_index_value]}")
        plt.tight_layout()
        plt.show()