import tensorflow as tf


class DenseBlock(tf.keras.layers.Layer):
    def __init__(self, units: int, dropout_rate: float = 0.25) -> None:
        super().__init__()
        self.dense = tf.keras.layers.Dense(units)
        self.normalization = tf.keras.layers.BatchNormalization()
        self.activation = tf.keras.layers.ReLU()
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        x = self.dense(inputs)
        x = self.normalization(x)
        x = self.activation(x)
        x = self.dropout(x)

        return x


def build_classifier(
    encoder_model: tf.keras.Model,
    dense_blocks_units: list[int],
    dense_blocks_dropout_rates: list[float],
    n_unique_features: list[int],
    feature_names: list[str]
) -> tf.keras.Model:
    for i in range(len(encoder_model.layers)):
        encoder_model.layers[i].trainable = False

    inp = tf.keras.layers.Input((256, 256, 3))
    x = encoder_model(inp)
    x = tf.keras.layers.Concatenate()([x[0], x[1]])

    for dense_block_units, dense_block_dropout_rate in zip(dense_blocks_units, dense_blocks_dropout_rates):
        x = DenseBlock(dense_block_units, dense_block_dropout_rate)(x)

    outputs = []
    for n, feature_name in zip(n_unique_features, feature_names):
        outputs.append(tf.keras.layers.Dense(n, activation="sigmoid", name=feature_name)(x))

    return tf.keras.Model(inp, outputs, name="classifier")
