"""
Main script for training the VQ-VAE model on the HipMRI dataset.
Includes SSIM benchmarking on the validation set.
"""
import os
import torch
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from torchvision.utils import save_image
from torchmetrics import StructuralSimilarityIndexMeasure

# Import our custom modules
from config import cfg
from modules import VQVAE
from dataset import get_dataloaders

def setup_environment():
    """Sets up the environment for training."""
    torch.manual_seed(cfg.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(cfg.SEED)
    os.makedirs(cfg.CHECKPOINT_DIR, exist_ok=True)
    print(f"Using device: {cfg.DEVICE}")
    print(f"Checkpoint directory created at: {cfg.CHECKPOINT_DIR}")

def evaluate(model, val_loader, device, ssim_metric):
    """Evaluates the model on the validation set and computes SSIM."""
    model.eval()
    total_val_recon_loss = 0
    ssim_metric.reset()  # Reset metric state

    with torch.no_grad():
        for data in val_loader:
            data = data.to(device)
            vq_loss, data_recon = model(data)
            recon_loss = F.mse_loss(data_recon, data)
            total_val_recon_loss += recon_loss.item()
            
            # Update the SSIM metric with the batch of original and reconstructed images
            # Our images are in [-1, 1], so the data_range is 2.0
            ssim_metric.update(data_recon, data)

    avg_val_recon_loss = total_val_recon_loss / len(val_loader)
    final_ssim = ssim_metric.compute()  # Compute the final SSIM score
    return avg_val_recon_loss, final_ssim.item()


def train_model():
    """The core training function."""
    setup_environment()

    # 1. Initialize Model, Optimizer, DataLoaders, and Metrics
    model = VQVAE(
        in_channels=cfg.IN_CHANNELS, hidden_channels=cfg.HIDDEN_CHANNELS,
        out_channels=cfg.IN_CHANNELS, num_res_blocks=cfg.NUM_RES_BLOCKS,
        res_channels=cfg.RES_CHANNELS, num_embeddings=cfg.NUM_EMBEDDINGS,
        embedding_dim=cfg.EMBEDDING_DIM, commitment_cost=cfg.COMMITMENT_COST
    ).to(cfg.DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE)
    train_loader, val_loader = get_dataloaders()
    
    # Initialize the SSIM metric. data_range is crucial. Our data is [-1, 1].
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=2.0).to(cfg.DEVICE)
    
    print(f"Model initialized with {sum(p.numel() for p in model.parameters()):,} parameters.")
    fixed_val_images = next(iter(val_loader)).to(cfg.DEVICE)

    # 2. The Training Loop
    best_ssim = 0.0
    for epoch in range(cfg.EPOCHS):
        model.train()
        loop = tqdm(train_loader, leave=True, desc=f"Epoch [{epoch+1}/{cfg.EPOCHS}]")
        total_recon_loss, total_vq_loss = 0, 0

        for data in loop:
            data = data.to(cfg.DEVICE)
            optimizer.zero_grad()
            vq_loss, data_recon = model(data)
            recon_loss = F.mse_loss(data_recon, data)
            total_loss = recon_loss + vq_loss
            total_loss.backward()
            optimizer.step()
            
            total_recon_loss += recon_loss.item()
            total_vq_loss += vq_loss.item()
            loop.set_postfix(recon_loss=f"{recon_loss.item():.4f}", vq_loss=f"{vq_loss.item():.4f}")

        # 3. Validation and Benchmarking at the end of each epoch
        val_recon_loss, val_ssim = evaluate(model, val_loader, cfg.DEVICE, ssim_metric)
        print(f"\nEpoch {epoch+1} Summary: Train Recon Loss: {total_recon_loss / len(train_loader):.4f} | "
              f"Val Recon Loss: {val_recon_loss:.4f} | Val SSIM: {val_ssim:.4f}")

        # Save the model if it has the best SSIM score so far
        if val_ssim > best_ssim:
            best_ssim = val_ssim
            best_model_path = os.path.join(cfg.CHECKPOINT_DIR, "vqvae_best_model.pth")
            torch.save(model.state_dict(), best_model_path)
            print(f"✅ New best model saved with SSIM: {best_ssim:.4f}")

        # Save reconstructed images periodically
        if (epoch + 1) % cfg.SAVE_IMAGE_EPOCH == 0:
            model.eval()
            with torch.no_grad():
                _, recon_images = model(fixed_val_images)
                comparison = torch.cat([fixed_val_images[:8], recon_images[:8]])
                save_path = os.path.join(cfg.CHECKPOINT_DIR, f"reconstruction_epoch_{epoch+1}.png")
                save_image(comparison.cpu(), save_path, nrow=8, normalize=True)
                print(f"Saved sample reconstructions to {save_path}")

    print(f"\nTraining complete. Best SSIM achieved: {best_ssim:.4f}")