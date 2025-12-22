
import os
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.dirname(BASE_DIR) 
DATASET_DIR = os.path.join(DATA_ROOT, "dataset", "without_mask") # Train on unmasked faces

CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
RESULT_DIR = os.path.join(BASE_DIR, "results")

# Hyperparameters
IMG_SIZE = 128
BATCH_SIZE = 16 
NUM_EPOCHS = 100 # Increased default limit
LEARNING_RATE_G = 0.0002
LEARNING_RATE_D = 0.0002 # PatchGAN usually benefits from similar LR
BETAS = (0.5, 0.999)
NUM_WORKERS = 2

# Loss Weights
LAMBDA_L1 = 100
LAMBDA_VGG = 10 # standard starting point, adjust if texture is too strong/weak
LAMBDA_ADV = 1

# Dataset Split
TRAIN_RATIO = 0.9 # 90% Train, 10% Test/Validation

LOAD_MODEL = True 
SAVE_EVERY = 1 # Save checkpoint every epoch

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)
