import tensorflow as tf
from .blocks import ConvBlock


class Sampling(tf.keras.layers.Layer):
    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


def build_encoder(latent_dim: int = 256) -> tf.keras.Model:
    encoder_inputs = tf.keras.Input(shape=(256, 256, 3))

    x = ConvBlock(filters=32, kernel_size=3, strides=2, dropout_rate=0.25)(encoder_inputs)
    x = ConvBlock(filters=32, kernel_size=3, strides=2, dropout_rate=0.25)(x)
    x = ConvBlock(filters=64, kernel_size=3, strides=2, dropout_rate=0.25)(x)
    x = ConvBlock(filters=64, kernel_size=3, strides=2, dropout_rate=0.25)(x)
    x = ConvBlock(filters=64, kernel_size=3, strides=2, dropout_rate=0.25)(x)
    x = tf.keras.layers.Flatten()(x)

    z_mean = tf.keras.layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = tf.keras.layers.Dense(latent_dim, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])

    return tf.keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
