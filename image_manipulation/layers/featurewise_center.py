import tensorflow as tf

from tensorflow import keras


class FeaturewiseCenter(keras.layers.Layer):
    """
    This class centers the whole dataset around 0. If there's a dominating color scheme
    or background then normalizing images feature-wise can help model converge better.
    """
    def __init__(self, dataset: tf.data.Dataset, batch_size: int = 32, **kwargs):
        super(FeaturewiseCenter, self).__init__(**kwargs)

        self._mean = FeaturewiseCenter.compute_dataset_mean(dataset, batch_size)

    def call(self, inputs):
        return inputs - self._mean

    @staticmethod
    def compute_dataset_mean(dataset, batch_size):
        # Accumulators for sum and number of samples
        pixel_sum = tf.zeros([1, 1, 3])
        num_samples = 0

        for img_batch, _ in dataset.batch(batch_size):
            current_batch_size = img_batch.shape[0]
            pixel_sum += tf.reduce_sum(img_batch, axis=[0, 1, 2])
            num_samples += current_batch_size * tf.reduce_prod(img_batch.shape[1:3])

        mean = pixel_sum / tf.cast(num_samples, tf.float32)

        return mean
