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


def get_feature_dataset(
    apes_info: pd.DataFrame,
    feature_names: list[str],
) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, dict[str, list[str]]]:
    feature_datasets = {"train": [], "validation": [], "test": []}
    feature_value_names = {}

    for feature in feature_names:
        feature_dummies = pd.get_dummies(
            apes_info.loc[:, [feature, "dataset"]], columns=[feature], prefix="", prefix_sep=""
        )
        feature_value_names[feature] = feature_dummies.drop(columns="dataset").columns.tolist()

        for feature_dataset in feature_datasets:
            feature_dummies_dataset = feature_dummies[feature_dummies["dataset"] == feature_dataset].drop(
                columns="dataset"
            )
            feature_datasets[feature_dataset].append(tf.data.Dataset.from_tensor_slices(feature_dummies_dataset))

    if len(feature_names) == 1:
        return (
            feature_datasets["train"][0],
            feature_datasets["validation"][0],
            feature_datasets["test"][0],
            feature_value_names,
        )

    return (
        tf.data.Dataset.zip(tuple(feature_datasets["train"])),
        tf.data.Dataset.zip(tuple(feature_datasets["validation"])),
        tf.data.Dataset.zip(tuple(feature_datasets["test"])),
        feature_value_names,
    )
