import tensorflow as tf
from .blocks import ConvTBlock


def build_decoder(latent_dim: int = 256) -> tf.keras.Model:
    latent_inputs = tf.keras.layers.Input(shape=(latent_dim,))

    x = tf.keras.layers.Dense(4096)(latent_inputs)
    x = tf.keras.layers.Reshape((8, 8, 64))(x)
    x = ConvTBlock(filters=64, kernel_size=3, strides=2, dropout_rate=0.25)(x)
    x = ConvTBlock(filters=64, kernel_size=3, strides=2, dropout_rate=0.25)(x)
    x = ConvTBlock(filters=32, kernel_size=3, strides=2, dropout_rate=0.25)(x)
    x = ConvTBlock(filters=32, kernel_size=3, strides=2, dropout_rate=0.25)(x)
    x = tf.keras.layers.Conv2DTranspose(3, 3, strides=2, padding="same")(x)

    decoder_outputs = tf.keras.layers.Activation("sigmoid")(x)

    return tf.keras.Model(latent_inputs, decoder_outputs, name="decoder")
