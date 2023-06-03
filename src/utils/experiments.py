import pandas as pd
import numpy as np
import tensorflow as tf
from utils import data_loading


def perform_image_shifting(
    encoder_model: tf.keras.Model,
    decoder_model: tf.keras.Model,
    dataset: tf.data.Dataset,
    apes_info: pd.DataFrame,
    feature: str,
    feature_name_1: str,
    feature_name_2: str,
    n_shifts: int,
    batch_size: int,
) -> tuple[np.array, np.array, list, list]:
    feature_1_images_ids = apes_info.loc[apes_info[feature] == feature_name_1, "image"].to_list()
    feature_2_images_ids = apes_info.loc[apes_info[feature] == feature_name_2, "image"].to_list()

    images_feature_1 = data_loading.load_specific_dataset(dataset, feature_1_images_ids, batch_size)
    images_feature_2 = data_loading.load_specific_dataset(dataset, feature_2_images_ids, batch_size)

    encoded_images_1 = encoder_model.predict(images_feature_1)
    encoded_images_2 = encoder_model.predict(images_feature_2)

    mean_images_1 = tf.reduce_mean(encoded_images_1[2], axis=0)
    mean_images_2 = tf.reduce_mean(encoded_images_2[2], axis=0)

    mean_difference_1 = mean_images_2 - mean_images_1
    mean_difference_2 = mean_images_1 - mean_images_2

    sample_1 = list(images_feature_1.take(1))[0][0].numpy()
    encoded_sample_1 = encoder_model(sample_1.reshape(1, 256, 256, 3))

    sample_2 = list(images_feature_2.take(1))[0][0].numpy()
    encoded_sample_2 = encoder_model(sample_2.reshape(1, 256, 256, 3))

    shifted_images_1 = []
    shifted_images_2 = []
    for i in range(0, n_shifts + 1):
        encoded_sample_1_shifted = encoded_sample_1[2] + mean_difference_1 * i / n_shifts
        decoded_sample_1_shifted = decoder_model(encoded_sample_1_shifted)

        encoded_sample_2_shifted = encoded_sample_2[2] + mean_difference_2 * i / n_shifts
        decoded_sample_2_shifted = decoder_model(encoded_sample_2_shifted)

        shifted_images_1.append(decoded_sample_1_shifted[0])
        shifted_images_2.append(decoded_sample_2_shifted[0])

    return sample_1, sample_2, shifted_images_1, shifted_images_2
