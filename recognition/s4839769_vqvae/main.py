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
import os # Import os for checking directories

from train import train_model
from predict import predict
from config import cfg

def main():
    """
    Parses command-line arguments and runs the specified mode (train or predict).
    """
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(
        description="VQ-VAE on HipMRI Dataset",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser.add_argument(
        'mode',
        choices=['train', 'predict'],
        help="Specify the operation to perform:\n"
             "  train   - Start or resume training the model.\n"
             "  predict - Generate reconstructions using a trained model."
    )
    
    # --- ADD THE NEW --local FLAG ---
    parser.add_argument(
        '--local',
        action='store_true', # This makes it a flag, e.g., --local
        help="Run in local mode. Uses the dataset path defined in 'LOCAL_DATASET_ROOT'."
    )
    
    args = parser.parse_args()
    
    # --- Welcome Message ---
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
            train_model()
        except KeyboardInterrupt:
            print("\nTraining interrupted by user. Exiting.")
        except Exception as e:
            print(f"\nAn error occurred during training: {e}")
            
    elif args.mode == 'predict':
        try:
            predict()
        except Exception as e:
            print(f"\nAn error occurred during prediction: {e}")

if __name__ == '__main__':
    main()