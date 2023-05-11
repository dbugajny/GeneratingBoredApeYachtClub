import tensorflow as tf


class ConvTBlock(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size=3, dropout_rate=0.25):
        super().__init__()
        self.conv = tf.keras.layers.Conv2DTranspose(filters=filters, kernel_size=kernel_size, strides=2, padding="same")
        self.normalization = tf.keras.layers.BatchNormalization()
        self.activation = tf.keras.layers.ReLU()
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, inputs):
        x = self.conv(inputs)
        x = self.normalization(x)
        x = self.activation(x)
        x = self.dropout(x)

        return x


def build_decoder(latent_dim=256):
    latent_inputs = tf.keras.layers.keras.Input(shape=(latent_dim,))

    x = tf.keras.layers.layers.Dense(4096)(latent_inputs)
    x = tf.keras.layers.layers.Reshape((8, 8, 64))(x)
    x = ConvTBlock(64, 3)(x)
    x = ConvTBlock(64, 3)(x)
    x = ConvTBlock(32, 3)(x)
    x = ConvTBlock(32, 3)(x)
    x = tf.keras.layers.Conv2DTranspose(3, 3, strides=2, padding="same")(x)

    decoder_outputs = tf.keras.layers.layers.Activation("sigmoid")(x)

    return tf.keras.layers.keras.Model(latent_inputs, decoder_outputs, name="decoder")
