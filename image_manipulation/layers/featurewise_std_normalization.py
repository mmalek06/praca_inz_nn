import tensorflow as tf

from tensorflow import keras


class FeaturewiseStdNormalization(keras.layers.Layer):
    def __init__(self, dataset, batch_size: int = 32, **kwargs):
        super(FeaturewiseStdNormalization, self).__init__(**kwargs)

        self.std = FeaturewiseStdNormalization.compute_dataset_std(dataset, batch_size)

    def call(self, inputs):
        # Divide the inputs by the computed standard deviation
        return inputs / self.std

    @staticmethod
    def compute_dataset_std(dataset, batch_size):
        pixel_sum = tf.zeros([1, 1, 3])
        pixel_squared_sum = tf.zeros([1, 1, 3])
        num_samples = 0

        for img_batch, _ in dataset.batch(batch_size):
            current_batch_size = img_batch.shape[0]
            pixel_sum += tf.reduce_sum(img_batch, axis=[0, 1, 2])
            pixel_squared_sum += tf.reduce_sum(tf.square(img_batch), axis=[0, 1, 2])
            num_samples += current_batch_size * tf.reduce_prod(img_batch.shape[1:3])

        mean = pixel_sum / tf.cast(num_samples, tf.float32)
        variance = (pixel_squared_sum / tf.cast(num_samples, tf.float32)) - tf.square(mean)
        std = tf.sqrt(variance)

        return std
