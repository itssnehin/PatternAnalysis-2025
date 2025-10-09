"""
Demonstrates the usage of a trained VQ-VAE model for image reconstruction.
- Loads the final trained model weights.
- Fetches a batch of data from the validation set.
- Generates reconstructions.
- Saves a comparison image of original vs. reconstructed images.
"""
import os
import torch
from torchvision.utils import save_image

# Import our custom modules
from config import cfg
from modules import VQVAE
from dataset import get_dataloaders  # We'll use this to get some test data

def predict():
    """
    Loads a trained model and generates image reconstructions.
    """
    # Ensure the device is set correctly
    device = cfg.DEVICE
    print(f"Using device: {device}")

    # --- 1. Load the Trained Model ---
    model_path = os.path.join(cfg.CHECKPOINT_DIR, "vqvae_final_model.pth")
    if not os.path.exists(model_path):
        print(f"ERROR: Model file not found at {model_path}")
        print("Please run train.py first to train and save the model.")
        return

    # Initialize the model with the same architecture as during training
    model = VQVAE(
        in_channels=cfg.IN_CHANNELS,
        hidden_channels=cfg.HIDDEN_CHANNELS,
        out_channels=cfg.IN_CHANNELS,
        num_res_blocks=cfg.NUM_RES_BLOCKS,
        res_channels=cfg.RES_CHANNELS,
        num_embeddings=cfg.NUM_EMBEDDINGS,
        embedding_dim=cfg.EMBEDDING_DIM,
        commitment_cost=cfg.COMMITMENT_COST
    ).to(device)

    # Load the saved state dictionary
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()  # Set the model to evaluation mode
    print(f"Successfully loaded trained model from {model_path}")

    # --- 2. Get a Batch of Data for Prediction ---
    # We use the validation loader to get unseen data
    try:
        _, val_loader = get_dataloaders()
        original_images = next(iter(val_loader)).to(device)
    except (ValueError, FileNotFoundError) as e:
        print(f"ERROR: Could not load validation data. {e}")
        return
        
    # --- 3. Generate Reconstructions ---
    print("Generating reconstructions...")
    with torch.no_grad():
        # The model returns vq_loss and the reconstructed images
        _, reconstructed_images = model(original_images)
        
    # --- 4. Save the Comparison Image ---
    # We'll compare the first 8 images from the batch
    num_images_to_show = 8
    
    # Arrange images in a grid: originals on top, reconstructions on the bottom
    comparison_grid = torch.cat([
        original_images[:num_images_to_show], 
        reconstructed_images[:num_images_to_show]
    ])
    
    # Create an output directory for predictions if it doesn't exist
    output_dir = 'predictions'
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, f"{cfg.PROJECT_NAME}_prediction_result.png")
    
    # save_image normalizes the image from [-1, 1] to [0, 1] for saving
    save_image(comparison_grid.cpu(), output_path, nrow=num_images_to_show, normalize=True)
    
    print("-" * 50)
    print("Prediction complete!")
    print(f"Saved comparison image to: {output_path}")
    print("Top row: Original Images | Bottom row: Reconstructed Images")
    print("-" * 50)

if __name__ == '__main__':
    predict()