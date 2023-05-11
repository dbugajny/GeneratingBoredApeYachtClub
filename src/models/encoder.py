import tensorflow as tf


class Sampling(tf.keras.layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class ConvBlock(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size=3, dropout_rate=0.25):
        super().__init__()
        self.conv = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=2, padding="same")
        self.normalization = tf.keras.layers.BatchNormalization()
        self.activation = tf.keras.layers.ReLU()
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, inputs):
        x = self.conv(inputs)
        x = self.normalization(x)
        x = self.activation(x)
        x = self.dropout(x)

        return x


def build_encoder(latent_dim=256):
    encoder_inputs = tf.keras.Input(shape=(256, 256, 3))

    x = ConvBlock(32, 3)(encoder_inputs)
    x = ConvBlock(32, 3)(x)
    x = ConvBlock(64, 3)(x)
    x = ConvBlock(64, 3)(x)
    x = ConvBlock(64, 3)(x)
    x = tf.keras.layers.Flatten()(x)

    z_mean = tf.keras.layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = tf.keras.layers.Dense(latent_dim, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])

    return tf.keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
