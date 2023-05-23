BATCH_SIZE = 128
STEPS_PER_EPOCH = 7000 // BATCH_SIZE
IMAGE_SIZE = (256, 256)
LATENT_DIM = 256

DATA_FILEPATH = "../data/images"
APES_INFO_FILEPATH = "../data/others/apes_info.csv"

FEATURE_NAMES = ["Mouth", "Background", "Hat", "Eyes", "Clothes", "Fur", "Earring"]
N_UNIQUE_FEATURES = [33, 8, 37, 23, 44, 19, 7]

EPOCHS_VAE = 200
EPOCHS_CLASSIFIER = 5

MODEL_VAE_FILEPATH = "../data/models/vae/"
HISTORY_VAE_FILEPATH = "../data/others/history_vae_training.csv"

MODEL_CLASSIFIER_FILEPATH = "../data/models/classifier/"
HISTORY_CLASSIFIER_FILEPATH = "../data/others/history_classifier_training.csv"

RECONSTRUCTION_LOSS_WEIGHT = 100
KL_LOSS_WEIGHT = 1
