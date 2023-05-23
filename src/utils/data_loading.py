import pandas as pd
import tensorflow as tf
from pathlib import Path


def load_full_dataset(data_filepath: str | Path, image_size: tuple[int], image_ids: list[str]):
    return tf.keras.utils.image_dataset_from_directory(
        directory=data_filepath,
        batch_size=None,
        image_size=image_size,
        shuffle=False,
        labels=image_ids,
    ).map(lambda x, y: (x / 255, y))


def load_specific_dataset(dataset: tf.data.Dataset, image_ids: list[str], batch_size: None | int) -> tf.data.Dataset:
    specific_dataset = dataset.filter(lambda _, y: tf.math.reduce_any(y == image_ids)).map(select_x)

    return specific_dataset.batch(batch_size) if batch_size else specific_dataset



@tf.autograph.experimental.do_not_convert
def select_x(x, _):
    return x


def get_image_ids(apes_info: pd.DataFrame, filepath: Path) -> tuple[list[str], list[str], list[str], list[str]]:
    all_images_ids = sorted([item.stem for item in filepath.iterdir() if item.suffix == ".png"])

    train_ids = apes_info.loc[apes_info["dataset"] == "train", "image"].to_list()
    validation_ids = apes_info.loc[apes_info["dataset"] == "validation", "image"].to_list()
    test_ids = apes_info.loc[apes_info["dataset"] == "test", "image"].to_list()

    return all_images_ids, train_ids, validation_ids, test_ids


def get_feature_dataset(apes_info: pd.DataFrame, feature_names: list[str], dataset_type: str) -> tf.data.Dataset:
    apes_info = apes_info[apes_info["dataset"] == dataset_type].drop(columns="dataset")

    feature_datasets = []
    for feature in feature_names:
        feature_dummies = pd.get_dummies(apes_info[feature])
        feature_datasets.append(tf.data.Dataset.from_tensor_slices(feature_dummies))

    return tf.data.Dataset.zip(tuple(feature_datasets))
