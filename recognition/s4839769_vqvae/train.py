# train.py

"""
The core training and validation loop for the VQ-VAE model.
- Handles the training over multiple epochs.
- Calculates reconstruction and VQ losses.
- Evaluates the model using SSIM on a validation set.
- Saves the best model based on SSIM and periodic image samples.
- Includes an early stopping mechanism based on a target SSIM score.
"""
import os
import torch
import torch.nn.functional as F
from torchvision.utils import save_image
from torchmetrics import StructuralSimilarityIndexMeasure
from tqdm import tqdm

from config import cfg
from modules import VQVAE
from dataset import get_dataloaders

def train_model():
    """
    Main function to orchestrate the VQ-VAE training process.
    """
    device = cfg.DEVICE
    print(f"Using device: {device}")
    os.makedirs(cfg.CHECKPOINT_DIR, exist_ok=True)

    # --- 1. Initialize Model, Optimizer, and DataLoaders ---
    model = VQVAE(
        in_channels=cfg.IN_CHANNELS, hidden_channels=cfg.HIDDEN_CHANNELS,
        out_channels=cfg.IN_CHANNELS, num_res_blocks=cfg.NUM_RES_BLOCKS,
        res_channels=cfg.RES_CHANNELS, num_embeddings=cfg.NUM_EMBEDDINGS,
        embedding_dim=cfg.EMBEDDING_DIM, commitment_cost=cfg.COMMITMENT_COST
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE)

    try:
        train_loader, val_loader = get_dataloaders()
    except (ValueError, FileNotFoundError) as e:
        print(f"ERROR: Could not load data. {e}")
        return

    # --- 2. Initialize Metrics and Tracking Variables ---
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=2.0).to(device)
    best_ssim = 0.0
    fixed_val_images = next(iter(val_loader)).to(device)

    # --- 3. The Main Training Loop ---
    print("Starting training...")
    for epoch in range(1, cfg.EPOCHS + 1):
        # --- Training Phase ---
        model.train()
        train_recon_loss = 0.0
        
        for batch_idx, data in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.EPOCHS} [Training]")):
            data = data.to(device)
            optimizer.zero_grad()
            vq_loss, data_recon = model(data)
            recon_loss = F.mse_loss(data_recon, data)
            loss = recon_loss + vq_loss
            loss.backward()
            optimizer.step()
            train_recon_loss += recon_loss.item()
        
        avg_train_loss = train_recon_loss / len(train_loader)

        # --- Validation Phase ---
        model.eval()
        val_recon_loss = 0.0
        
        with torch.no_grad():
            for data in tqdm(val_loader, desc=f"Epoch {epoch}/{cfg.EPOCHS} [Validation]"):
                data = data.to(device)
                _, data_recon = model(data)
                recon_loss = F.mse_loss(data_recon, data)
                val_recon_loss += recon_loss.item()
                ssim_metric.update(data_recon, data)

        avg_val_loss = val_recon_loss / len(val_loader)
        epoch_ssim = ssim_metric.compute()
        ssim_metric.reset()

        print(
            f"Epoch: {epoch}/{cfg.EPOCHS} | "
            f"Train Recon Loss: {avg_train_loss:.4f} | "
            f"Val Recon Loss: {avg_val_loss:.4f} | "
            f"Val SSIM: {epoch_ssim:.4f}"
        )

        # --- 4. Checkpointing and Saving ---
        if epoch_ssim > best_ssim:
            best_ssim = epoch_ssim
            model_path = os.path.join(cfg.CHECKPOINT_DIR, "vqvae_best_model.pth")
            torch.save(model.state_dict(), model_path)
            print(f"✨ New best model saved with SSIM: {best_ssim:.4f} ✨")

        if epoch % cfg.SAVE_IMAGE_EPOCH == 0 or epoch == cfg.EPOCHS:
            with torch.no_grad():
                _, reconstructed_samples = model(fixed_val_images)
            comparison_grid = torch.cat([fixed_val_images[:8], reconstructed_samples[:8]])
            img_path = os.path.join(cfg.CHECKPOINT_DIR, f"reconstruction_epoch_{epoch}.png")
            save_image(comparison_grid.cpu(), img_path, nrow=8, normalize=True)
            print(f"Saved sample reconstruction grid to {img_path}")

        # --- 5. EARLY STOPPING CHECK (NEW CODE) ---
        if best_ssim >= cfg.EARLY_STOP_SSIM:
            print(f"\n--- Early stopping triggered! ---")
            print(f"Validation SSIM ({best_ssim:.4f}) has reached the target ({cfg.EARLY_STOP_SSIM}).")
            break # Exit the training loop

    print("\nTraining complete!")
    print(f"Best validation SSIM achieved: {best_ssim:.4f}")