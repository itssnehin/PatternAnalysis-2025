"""
Main entry point for the VQ-VAE project on the HipMRI dataset.
This script orchestrates the training and prediction workflows.

Usage:
    - To train the model:
      python main.py train

    - To generate predictions with a trained model:
      python main.py predict
"""
import argparse
import torch

# Import the core functions from our other scripts
# Note: We need to handle the case where train.py or predict.py might do things
# upon import. The `if __name__ == '__main__':` block in those files prevents this.
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
        formatter_class=argparse.RawTextHelpFormatter  # For better help text formatting
    )
    
    parser.add_argument(
        'mode',
        choices=['train', 'predict'],
        help="Specify the operation to perform:\n"
             "  train   - Start or resume training the model.\n"
             "  predict - Generate reconstructions using a trained model."
    )
    
    args = parser.parse_args()
    
    # --- Welcome Message ---
    print("=" * 50)
    print(f"Starting Project: {cfg.PROJECT_NAME}")
    print(f"Selected Mode: {args.mode.upper()}")
    print("=" * 50)

    # --- Mode Execution ---
    if args.mode == 'train':
        try:
            train_model()
        except KeyboardInterrupt:
            print("\nTraining interrupted by user. Exiting.")
        except Exception as e:
            print(f"\nAn error occurred during training: {e}")
            # Optionally add more detailed error logging here
            
    elif args.mode == 'predict':
        try:
            predict()
        except Exception as e:
            print(f"\nAn error occurred during prediction: {e}")

if __name__ == '__main__':
    main()