import tensorflow as tf

from tensorflow import keras


def get_augmentation_layers() -> keras.Sequential:
    return tf.keras.Sequential([
        keras.layers.RandomFlip('horizontal_and_vertical'),
        keras.layers.RandomRotation(1),
        keras.layers.RandomBrightness((-.3, .3)),
        keras.layers.RandomContrast(.3),
        keras.layers.RandomZoom((.3, -.3), (.3, -.3))
    ])
