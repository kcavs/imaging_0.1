import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
from pathlib import Path
class model:
    data_dir=None
    img_height, img_width = None, None
    batch_size = None
    seed=None
    # 1. Set path to your extracted dataset folder
    def dataset(DIR,H,W,size,seeds):
        data_dir = DIR
        img_height, img_width = H, W
        batch_size = size
        seed=seeds

    # 2. Load training data (80%) and validation data (20%)
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed,
        image_size=(img_height, img_width),
        batch_size=batch_size
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed,
        image_size=(img_height, img_width),
        batch_size=batch_size
    )

    class_names = train_ds.class_names
    print("Found classes:", class_names)

    # 3. Normalize pixel values from [0, 255] to [0, 1]
    normalization_layer = layers.Rescaling(1./255)
    train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

    # 4. Build the Convolutional Neural Network (CNN) model
    model = models.Sequential([
        layers.Conv2D(16, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
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
