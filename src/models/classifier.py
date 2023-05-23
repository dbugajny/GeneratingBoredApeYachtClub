import tensorflow as tf


def build_classifier(
    encoder_model: tf.keras.Model, n_unique_features: list[int], feature_names: list[str]
) -> tf.keras.Model:
    for i in range(len(encoder_model.layers)):
        encoder_model.layers[i].trainable = False

    inp = tf.keras.layers.Input((256, 256, 3))
    x = encoder_model(inp)
    x = tf.keras.layers.Concatenate()([x[0], x[1]])

    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(512, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    x = tf.keras.layers.Dense(512, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.25)(x)

    outputs = []
    for n, feature_name in zip(n_unique_features, feature_names):
        outputs.append(tf.keras.layers.Dense(n, activation="sigmoid", name=feature_name)(x))

    return tf.keras.Model(inp, outputs, name="classifier")
