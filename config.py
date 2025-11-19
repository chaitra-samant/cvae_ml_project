import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_ROOT = None 
IMG_DIR = None
CSV_PATH = None

IMG_SIZE = 64
LATENT_DIM = 256
ATTRS = ["Eyeglasses", "Smiling", "Mustache"]
NUM_ATTRS = len(ATTRS)
BASE_CHANNELS = 128

NUM_EPOCHS = 75
WARMUP_EPOCHS = 5
BATCH_SIZE = 256
LEARNING_RATE = 6e-4
WEIGHT_DECAY = 1e-5
BETA = 4.0
GRAD_CLIP_NORM = 0.5

SAMPLE_DIR = "samples_v4"
MODEL_SAVE_PATH = "cvae_eyeglasses_smiling_mustache.pth"