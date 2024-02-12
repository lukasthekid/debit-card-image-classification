import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
import pathlib
import os
import numpy as np
from PIL import Image


class CNN_v1:
    def __init__(self, badge_size: int, img_size: int, data_dir_path: pathlib.Path):
        self.badge_size = badge_size
        self.img_size = img_size
        self.data_dir_path = data_dir_path
        self.load_data()

    def load_data(self):
        train_ds = tf.keras.utils.image_dataset_from_directory(
            self.data_dir_path,
            validation_split=0.2,
            subset="training",
            seed=123,
            image_size=(self.img_size, self.img_size),
            batch_size=self.badge_size)

        val_ds = tf.keras.utils.image_dataset_from_directory(
            self.data_dir_path,
            validation_split=0.2,
            subset="validation",
            seed=123,
            image_size=(self.img_size, self.img_size),
            batch_size=self.badge_size)

        self.train_ds = prepare(train_ds, True, True)
        self.val_ds = prepare(val_ds, False, True)

    def build_model(self, epochs: int):
        data_augmentation = keras.Sequential(
            [
                layers.RandomFlip("horizontal_and_vertical",
                                  input_shape=(self.img_size,
                                               self.img_size,
                                               3)),
                layers.RandomRotation(0.3)
            ]
        )

        model = Sequential([
            data_augmentation,
            layers.Rescaling(1. / 255),
            layers.Conv2D(16, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(32, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Dropout(0.2),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dense(1, name="outputs")
        ])

        model.compile(optimizer='adam',
                      loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                      metrics=['binary_accuracy'])

        history = model.fit(
            self.train_ds,
            epochs=epochs,
            validation_data=self.val_ds
        )
        self.model = model
        return history

    def save_model(self, path: str):
        self.model.save(path)


class MobileNetV2_debit_card:
    def __init__(self, badge_size: int, img_size: int, data_dir_path: pathlib.Path):
        self.badge_size = badge_size
        self.img_size = img_size
        self.data_dir_path = data_dir_path
        self.load_data()

    def load_data(self):
        train_ds = tf.keras.utils.image_dataset_from_directory(
            self.data_dir_path,
            validation_split=0.2,
            subset="training",
            seed=123,
            image_size=(self.img_size, self.img_size),
            batch_size=self.badge_size)

        val_ds = tf.keras.utils.image_dataset_from_directory(
            self.data_dir_path,
            validation_split=0.2,
            subset="validation",
            seed=123,
            image_size=(self.img_size, self.img_size),
            batch_size=self.badge_size)

        self.train_ds = prepare(train_ds, True, True)
        self.val_ds = prepare(val_ds, False, True)
        self.filenames = val_ds.file_paths

    def build_model(self, epochs: int):
        preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
        base_model = MobileNetV2(input_shape=(self.img_size, self.img_size, 3), include_top=False, weights='imagenet')

        base_model.trainable = False
        # Create new model1 on top
        inputs = tf.keras.Input(shape=(self.img_size, self.img_size, 3))
        x = preprocess_input(inputs)
        x = base_model(x, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        x = layers.Dense(1)(x)  # Binary classification
        model = tf.keras.Model(inputs, x)

        model.compile(optimizer='adam',
                      loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                      metrics=['binary_accuracy'])

        history = model.fit(
            self.train_ds,
            epochs=epochs,
            validation_data=self.val_ds
        )
        self.model = model
        return history

    def save_model(self, path: str):
        self.model.save(path)

    def store_misclassified(self, directory):
        # Directory to save misclassified images
        os.makedirs(directory, exist_ok=True)

        # Open a text file to store the filenames of misclassified images
        with open(os.path.join(directory, 'misclassified_images.txt'), 'w') as f:
            # Iterate over the validation dataset
            total_images = 0
            for i, (images, labels) in enumerate(self.val_ds):
                predictions = self.model.predict(images)
                score = tf.nn.sigmoid(predictions)
                predictions = tf.where(score < 0.5, 0, 1)[0]
                # Find misclassified indices
                misclassified_indices = np.where(predictions != labels.numpy())[0]
                # Save misclassified images and their filenames
                for idx in misclassified_indices:
                    # Write the filename of the misclassified image to the text file
                    f.write(f'{self.filenames[total_images + idx]} was misclassified\n')
                total_images += len(images)


def prepare(ds, shuffle=False, augment=False):
    data_augmentation = keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.3)])
    AUTOTUNE = tf.data.AUTOTUNE
    if shuffle:
        ds = ds.shuffle(1000)
        # Use data augmentation only on the training set.
    if augment:
        ds = ds.map(lambda x, y: (data_augmentation(x, training=True), y),
                    num_parallel_calls=AUTOTUNE)
        # Use buffered prefetching on all datasets.
    return ds.prefetch(buffer_size=AUTOTUNE)
