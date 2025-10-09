"""
Configuration file for the VQ-VAE model on the HipMRI dataset.
Centralizes all hyperparameters and settings for easy access and modification.
"""
import torch

class Config:
    # --- General Project Settings ---
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    SEED = 42
    PROJECT_NAME = "snehin_HipMRI_VQVAE"
    
    # --- Dataset Settings ---
    # Path to the directory containing 'keras_slices_train', 'keras_slices_test', etc.
    DATASET_ROOT = '/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/'
    # The HipMRI slices are of different sizes. We will resize them to a consistent dimension.
    # 256x256 is a good starting point for detail. Use 128x128 for faster training.
    IMAGE_SIZE = 128 
    IN_CHANNELS = 1  # MRI scans are grayscale

    # --- Training Settings ---
    EPOCHS = 100
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-4 # Adam optimizer is often used for VAEs, so a smaller LR is typical
    NUM_WORKERS = 4      # For the DataLoader on Rangpur

    # --- VQ-VAE Model Hyperparameters ---
    # This MUST match the embedding_dim
    HIDDEN_CHANNELS = 128
    
    # Placeholder for residual blocks if we add them later
    NUM_RES_BLOCKS = 2
    RES_CHANNELS = 64

    # --- Vector Quantizer (Codebook) Settings ---
    # The size of the discrete latent space (the number of "codes" or "vectors")
    NUM_EMBEDDINGS = 512 
    # The dimensionality of each embedding vector. This must match the output of the encoder.
    EMBEDDING_DIM = 128 
    # The 'beta' factor in the loss function, balances reconstruction and codebook learning
    COMMITMENT_COST = 0.25

    # --- Logging and Checkpointing ---
    CHECKPOINT_DIR = f'./checkpoints/{PROJECT_NAME}'
    # Frequency to print training stats (in number of batches)
    LOG_FREQ = 100 
    # Frequency to save a grid of reconstructed images (in number of epochs)
    SAVE_IMAGE_EPOCH = 5

# Create a global instance of the configuration
cfg = Config()