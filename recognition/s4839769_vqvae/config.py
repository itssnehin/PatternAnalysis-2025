# config.py

"""
Configuration file for the VQ-VAE model on the HipMRI dataset.
Centralizes all hyperparameters and settings for easy access and modification.
"""
import torch
import os # Import os for path joining

class Config:
    # --- General Project Settings ---
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # SEED = 42
    PROJECT_NAME = "snehin_HipMRI_VQVAE"
    
    # --- Dataset Settings ---
    # Default path for the Rangpur cluster
    DATASET_ROOT = '/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/'
    
    # Path for local execution. Assumes 'HipMRI_Study_open' is in the current directory as this script.
    LOCAL_DATASET_ROOT = os.path.join('HipMRI_Study_open', 'keras_slices_data')
    
    # The HipMRI slices are of different sizes. I resized them to a consistent dimension.
    IMAGE_SIZE = 128
    IN_CHANNELS = 1  # MRI scans are grayscale

    # --- Training Settings ---
    EPOCHS = 70
    WARMUP_EPOCHS = 3

    BATCH_SIZE = 64
    LEARNING_RATE = 1e-4
    NUM_WORKERS = 8 # For the DataLoader on Rangpur. May need to be 1 for local Windows.
    EARLY_STOP_SSIM = None # can set to None to disable
    #ALPHA = 0.1 # Weight Factor
    # --- VQ-VAE Model Hyperparameters ---
    HIDDEN_CHANNELS = 128
    NUM_RES_BLOCKS = 2
    RES_CHANNELS = 64

    # --- Vector Quantizer (Codebook) Settings ---
    NUM_EMBEDDINGS = 512 
    EMBEDDING_DIM = 128 
    COMMITMENT_COST = 0.25

    # --- Logging and Checkpointing ---
    CHECKPOINT_DIR = f'./checkpoints/'
    LOG_FREQ = 100 
    SAVE_IMAGE_EPOCH = 5

# Create a global instance of the configuration
cfg = Config()