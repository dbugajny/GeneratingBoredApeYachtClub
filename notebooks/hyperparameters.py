BATCH_SIZE = 128
STEPS_PER_EPOCH = 7000 // BATCH_SIZE
IMAGE_SIZE = (256, 256)
LATENT_DIM = 256

RECONSTRUCTION_LOSS_WEIGHT = 100
KL_LOSS_WEIGHT = 1

ENCODER_MODEL_HYPERPARAMETERS = {
    "image_shape": (256, 256, 3),
    "latent_dim": 256,
    "conv_block_filters": [32, 32, 64, 64, 64],
    "conv_block_kernel_sizes": [3, 3, 3, 3, 3],
    "conv_block_strides": [2, 2, 2, 2, 2],
    "conv_block_dropout_rates": [0.25, 0.25, 0.25, 0.25, 0.25],
}

DECODER_MODEL_HYPERPARAMETERS = {
    "latent_dim": 256,
    "dense_layer_units": 4096,
    "reshape_layer_target_shape": (8, 8, 64),
    "convt_block_filters": [64, 64, 32, 32],
    "convt_block_kernel_sizes": [3, 3, 3, 3],
    "convt_block_strides": [2, 2, 2, 2],
    "convt_block_dropout_rates": [0.25, 0.25, 0.25, 0.25],
}

MULTI_LABEL_CLASSIFICATION_MODEL_HYPERPARAMETERS = {
    "dense_block_units": [512, 1024, 2048],
    "dense_block_dropout_rates": [0.10, 0.10, 0.10],
}


MOUTH_CLASSIFICATION_MODEL_HYPERPARAMETERS = {
    "image_cropping": ((90, 54), (100, 60)),
    "conv_block_filters": [8, 8],
    "conv_block_kernel_sizes": [3, 3],
    "conv_block_strides": [4, 4],
    "conv_block_dropout_rates": [0.1, 0.1],
    "dense_block_units": [16, 16],
    "dense_block_dropout_rates": [0.1, 0.1],
    "n_unique_features": 33,
}

BACKGROUND_CLASSIFICATION_MODEL_HYPERPARAMETERS = {
    "image_cropping": ((218, 6), (218, 6)),
    "conv_block_filters": [8],
    "conv_block_kernel_sizes": [3],
    "conv_block_strides": [8],
    "conv_block_dropout_rates": [0.1],
    "dense_block_units": [4, 4],
    "dense_block_dropout_rates": [0.0, 0.0],
    "n_unique_features": 8,
}

HAT_CLASSIFICATION_MODEL_HYPERPARAMETERS = {
    "image_cropping": ((0, 120), (66, 54)),
    "conv_block_filters": [8, 8],
    "conv_block_kernel_sizes": [3, 3],
    "conv_block_strides": [4, 4],
    "conv_block_dropout_rates": [0.1, 0.1],
    "dense_block_units": [8, 8],
    "dense_block_dropout_rates": [0.1, 0.1],
    "n_unique_features": 37,
}

EYES_CLASSIFICATION_MODEL_HYPERPARAMETERS = {
    "image_cropping": ((70, 138), (106, 70)),
    "conv_block_filters": [16, 16],
    "conv_block_kernel_sizes": [3, 3],
    "conv_block_strides": [4, 2],
    "conv_block_dropout_rates": [0.1, 0.1],
    "dense_block_units": [16, 16],
    "dense_block_dropout_rates": [0.1, 0.1],
    "n_unique_features": 23,
}

CLOTHES_CLASSIFICATION_MODEL_HYPERPARAMETERS = {
    "image_cropping": ((192, 0), (37, 75)),
    "conv_block_filters": [16, 16, 16, 16],
    "conv_block_kernel_sizes": [3, 3, 3, 3],
    "conv_block_strides": [2, 2, 2, 2],
    "conv_block_dropout_rates": [0.2, 0.2, 0.2, 0.2],
    "dense_block_units": [32, 32],
    "dense_block_dropout_rates": [0.1, 0.1],
    "n_unique_features": 44,
}

FUR_CLASSIFICATION_MODEL_HYPERPARAMETERS = {
    "image_cropping": ((166, 10), (87, 105)),
    "conv_block_filters": [16, 16, 16, 16],
    "conv_block_kernel_sizes": [3, 3, 3, 3],
    "conv_block_strides": [2, 2, 2, 2],
    "conv_block_dropout_rates": [0.2, 0.2, 0.2, 0.2],
    "dense_block_units": [16, 16],
    "dense_block_dropout_rates": [0.1, 0.1],
    "n_unique_features": 19,
}

EARRING_CLASSIFICATION_MODEL_HYPERPARAMETERS = {
    "image_cropping": ((116, 120), (60, 172)),
    "conv_block_filters": [8, 8],
    "conv_block_kernel_sizes": [3, 3],
    "conv_block_strides": [2, 2],
    "conv_block_dropout_rates": [0.2, 0.2],
    "dense_block_units": [16, 8],
    "dense_block_dropout_rates": [0.1, 0.1],
    "n_unique_features": 7,
}
