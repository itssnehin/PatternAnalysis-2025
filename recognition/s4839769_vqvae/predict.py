"""
Demonstrates the usage of a trained VQ-VAE model and reports SSIM score.
"""
import os
import torch
from torchvision.utils import save_image
from torchmetrics import StructuralSimilarityIndexMeasure

from config import cfg
from modules import VQVAE
from dataset import get_dataloaders

def predict():
    """Loads a trained model, generates reconstructions, and reports SSIM."""
    device = cfg.DEVICE
    print(f"Using device: {device}")

    # --- 1. Load the Best Trained Model ---
    # We load the 'best' model now, not just the final one
    model_path = os.path.join(cfg.CHECKPOINT_DIR, "vqvae_best_model.pth")
    if not os.path.exists(model_path):
        print(f"ERROR: Model file not found at {model_path}. Please run train.py first.")
        return

    model = VQVAE(
        in_channels=cfg.IN_CHANNELS, hidden_channels=cfg.HIDDEN_CHANNELS,
        out_channels=cfg.IN_CHANNELS, num_res_blocks=cfg.NUM_RES_BLOCKS,
        res_channels=cfg.RES_CHANNELS, num_embeddings=cfg.NUM_EMBEDDINGS,
        embedding_dim=cfg.EMBEDDING_DIM, commitment_cost=cfg.COMMITMENT_COST
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"Successfully loaded best trained model from {model_path}")

    # --- 2. Get Data and Initialize Metric ---
    try:
        _, val_loader = get_dataloaders()
        original_images = next(iter(val_loader)).to(device)
    except (ValueError, FileNotFoundError) as e:
        print(f"ERROR: Could not load validation data. {e}")
        return
        
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=2.0).to(device)
        
    # --- 3. Generate Reconstructions and Calculate SSIM ---
    print("Generating reconstructions and calculating SSIM...")
    with torch.no_grad():
        _, reconstructed_images = model(original_images)
        # Calculate SSIM just for this batch
        batch_ssim = ssim_metric(reconstructed_images, original_images)

    # --- 4. Save the Comparison Image ---
    comparison_grid = torch.cat([original_images[:8], reconstructed_images[:8]])
    output_dir = 'predictions'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{cfg.PROJECT_NAME}_prediction_result.png")
    save_image(comparison_grid.cpu(), output_path, nrow=8, normalize=True)
    
    print("-" * 50)
    print("Prediction complete!")
    print(f"SSIM for this batch: {batch_ssim.item():.4f}")
    print(f"Saved comparison image to: {output_path}")
    print("-" * 50)

if __name__ == '__main__':
    predict()