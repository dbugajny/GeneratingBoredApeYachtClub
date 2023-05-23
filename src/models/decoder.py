import tensorflow as tf


class ConvTBlock(tf.keras.layers.Layer):
    def __init__(self, filters: int, kernel_size: int = 3, dropout_rate: float = 0.25) -> None:
        super().__init__()
        self.conv = tf.keras.layers.Conv2DTranspose(filters=filters, kernel_size=kernel_size, strides=2, padding="same")
        self.normalization = tf.keras.layers.BatchNormalization()
        self.activation = tf.keras.layers.ReLU()
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        x = self.conv(inputs)
        x = self.normalization(x)
        x = self.activation(x)
        x = self.dropout(x)

        return x


def build_decoder(latent_dim: int = 256) -> tf.keras.Model:
    latent_inputs = tf.keras.layers.Input(shape=(latent_dim,))

    x = tf.keras.layers.Dense(4096)(latent_inputs)
    x = tf.keras.layers.Reshape((8, 8, 64))(x)
    x = ConvTBlock(64, 3)(x)
    x = ConvTBlock(64, 3)(x)
    x = ConvTBlock(32, 3)(x)
    x = ConvTBlock(32, 3)(x)
    x = tf.keras.layers.Conv2DTranspose(3, 3, strides=2, padding="same")(x)

    decoder_outputs = tf.keras.layers.Activation("sigmoid")(x)

    return tf.keras.Model(latent_inputs, decoder_outputs, name="decoder")
