# main.py

"""
Main entry point for the VQ-VAE project on the HipMRI dataset.
This script orchestrates the training and prediction workflows.

Usage:
    - To train the model on Rangpur:
      python main.py train

    - To train the model on a local machine:
      python main.py train --local

    - To generate images on Rangpur:
      python main.py predict

    - To generate images locally:
      python main.py predict --local
"""
import argparse
import torch
import os

from train import train_model
from predict import predict
from train_pixelcnn import train_pixelcnn # NEW IMPORT
from config import cfg
from dataset import get_dataloaders


def main():
    parser = argparse.ArgumentParser(
        description="VQ-VAE on HipMRI Dataset",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser.add_argument(
        'mode',
        choices=['train', 'train_pixelcnn', 'predict'], # NEW CHOICE
        help="Specify the operation to perform:\n"
             "  train          - Train the VQ-VAE model.\n"
             "  train_pixelcnn - Train the PixelCNN prior over the VQ-VAE's latents.\n"
             "  predict        - Generate images using both trained models."
    )
    
    parser.add_argument('--local', action='store_true', help="Run in local mode.")
    args = parser.parse_args()
    
    print("=" * 50)
    print(f"Starting Project: {cfg.PROJECT_NAME}")
    print(f"Selected Mode: {args.mode.upper()}")

    # --- UPDATE CONFIG BASED ON --local FLAG (NEW LOGIC) ---
    if args.local:
        print("--- Running in LOCAL mode ---")
        cfg.DATASET_ROOT = cfg.LOCAL_DATASET_ROOT
        # On local machines, especially Windows, num_workers > 0 can cause issues.
        # Set it to 0 for better compatibility.
        cfg.NUM_WORKERS = 0
        print(f"Using local dataset path: {cfg.DATASET_ROOT}")
        
        # Add a check to ensure the local directory exists
        if not os.path.isdir(cfg.DATASET_ROOT):
            print("\nERROR: Local dataset directory not found!")
            print(f"Please ensure the '{os.path.dirname(cfg.DATASET_ROOT)}' folder exists in your project directory.")
            return
    else:
        print("--- Running in CLUSTER mode ---")
        print(f"Using cluster dataset path: {cfg.DATASET_ROOT}")

    print("=" * 50)
    # --- Mode Execution ---
    if args.mode == 'train':
        try:
            # --- UPDATE THIS LINE ---
            # Unpack all three loaders, even though train_model only needs two.
            train_loader, val_loader, _ = get_dataloaders()
            
            # Pass only the required loaders to the training function
            train_model(train_loader, val_loader)
            
            print("\nTraining finished. Generating final predictions from the best model...")
            predict()
        except Exception as e: print(f"\nAn error occurred during VQ-VAE training: {e}")
            
    # --- NEW MODE ---
    elif args.mode == 'train_pixelcnn':
        try:
            train_pixelcnn()
            print("\nPixelCNN training finished. Now you can generate images by running:")
            print("python main.py predict --local")
        except Exception as e: print(f"\nAn error occurred during PixelCNN training: {e}")
    elif args.mode == 'predict':
        try:
            # (Predict logic is unchanged for now, we will update it next)
            predict()
        except Exception as e: print(f"\nAn error occurred during prediction: {e}")

if __name__ == '__main__':
    main()
