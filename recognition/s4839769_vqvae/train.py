# train.py

"""
The core training and validation loop for the VQ-VAE model.
- Uses a combined MSE + SSIM loss for a balance of pixel and structural accuracy.
- Evaluates the model using SSIM on a validation set.
- Saves the best model based on SSIM and periodic, labeled image samples.
- Includes an early stopping mechanism based on a target SSIM score.
- Plots the training history (loss and SSIM) at the end of training.
"""
import os
import torch
import torch.nn.functional as F
from torchvision.utils import make_grid
from torchmetrics.image import StructuralSimilarityIndexMeasure 
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
# --- LPIPS is no longer needed ---
# import lpips 

from config import cfg
from modules import VQVAE
from dataset import get_dataloaders

def plot_training_progress(history, save_path):
    """
    Plots and saves the training and validation loss, and validation SSIM.
    """
    print(f"Plotting training progress to {save_path}...")
    epochs = range(1, len(history['train_loss']) + 1)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    ax1.plot(epochs, history['train_loss'], 'bo-', label='Training Total Loss')
    ax1.plot(epochs, history['val_loss'], 'ro-', label='Validation Loss (MSE)')
    ax1.set_title('Training and Validation Loss')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    ax2.plot(epochs, history['val_ssim'], 'go-', label='Validation SSIM')
    ax2.set_title('Validation SSIM')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('SSIM Score')
    ax2.legend()
    ax2.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print("Plotting complete.")

def train_model():
    """
    Main function to orchestrate the VQ-VAE training process.
    """
    device = cfg.DEVICE
    print(f"Using device: {device}")
    os.makedirs(cfg.CHECKPOINT_DIR, exist_ok=True)

    model = VQVAE(
        in_channels=cfg.IN_CHANNELS, hidden_channels=cfg.HIDDEN_CHANNELS,
        out_channels=cfg.IN_CHANNELS, num_res_blocks=cfg.NUM_RES_BLOCKS,
        res_channels=cfg.RES_CHANNELS, num_embeddings=cfg.NUM_EMBEDDINGS,
        embedding_dim=cfg.EMBEDDING_DIM, commitment_cost=cfg.COMMITMENT_COST
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE)

    try: train_loader, val_loader = get_dataloaders()
    except (ValueError, FileNotFoundError) as e:
        print(f"ERROR: Could not load data. {e}")
        return

    # --- REMOVED LPIPS INITIALIZATION ---
    
    # --- NEW: Initialize an SSIM function specifically for the training loss ---
    ssim_loss_fn = StructuralSimilarityIndexMeasure(data_range=2.0).to(device)
    ssim_val_metric = StructuralSimilarityIndexMeasure(data_range=2.0).to(device)
    
    best_ssim = 0.0
    fixed_val_images = next(iter(val_loader)).to(device)
    
    history = {'train_loss': [], 'val_loss': [], 'val_ssim': []}

    print("Starting training with combined MSE + SSIM loss...")
    for epoch in range(1, cfg.EPOCHS + 1):
        # --- Training Phase ---
        model.train()
        train_total_loss = 0.0
        
        for batch_idx, data in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.EPOCHS} [Training]")):
            data = data.to(device)
            optimizer.zero_grad()

            vq_loss, data_recon = model(data)
            
            # --- LOSS CALCULATION ---
            # 1. Pixel-wise reconstruction loss
            mse_loss = F.mse_loss(data_recon, data)
            # 2. Structural similarity loss. Score is [0, 1], so loss is 1 - score.
            ssim_loss = 1.0 - ssim_loss_fn(data_recon, data)
            
            # Combine the two reconstruction losses. A weight (alpha) can be added.
            # e.g., recon_loss = alpha * mse_loss + (1-alpha) * ssim_loss
            # For simplicity, we'll weight them equally for now.
            alpha = cfg.ALPHA
            recon_loss = mse_loss + (alpha* ssim_loss)
            
            # Final total loss to backpropagate
            loss = recon_loss + vq_loss
            
            loss.backward()
            optimizer.step()
            
            train_total_loss += loss.item()
        
        avg_train_loss = train_total_loss / len(train_loader)

        # --- Validation Phase ---
        model.eval()
        val_recon_loss = 0.0
        
        with torch.no_grad():
            for data in tqdm(val_loader, desc=f"Epoch {epoch}/{cfg.EPOCHS} [Validation]"):
                data = data.to(device)
                _, data_recon = model(data)
                
                recon_loss = F.mse_loss(data_recon, data)
                val_recon_loss += recon_loss.item()
                
                # Use the separate validation metric
                ssim_val_metric.update(data_recon, data)

        avg_val_loss = val_recon_loss / len(val_loader)
        epoch_ssim = ssim_val_metric.compute()
        ssim_val_metric.reset() # Reset metric for the next epoch

        print(
            f"Epoch: {epoch}/{cfg.EPOCHS} | "
            f"Train Total Loss: {avg_train_loss:.4f} | "
            f"Val Recon Loss (MSE): {avg_val_loss:.4f} | "
            f"Val SSIM: {epoch_ssim:.4f}"
        )
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_ssim'].append(epoch_ssim.item())

        if epoch_ssim > best_ssim:
            best_ssim = epoch_ssim
            model_path = os.path.join(cfg.CHECKPOINT_DIR, "vqvae_best_model.pth")
            torch.save(model.state_dict(), model_path)
            print(f"✨ New best model saved with SSIM: {best_ssim:.4f} ✨")

        if epoch % cfg.SAVE_IMAGE_EPOCH == 0 or epoch == cfg.EPOCHS:
            # (Image saving logic is unchanged)
            with torch.no_grad():
                _, reconstructed_samples = model(fixed_val_images)
            all_images_tensor = torch.cat([fixed_val_images[:8], reconstructed_samples[:8]])
            grid_tensor = make_grid(all_images_tensor.cpu(), nrow=8, normalize=True)
            grid_pil = transforms.ToPILImage()(grid_tensor)
            label_width = 150
            canvas = Image.new('RGB', (grid_pil.width + label_width, grid_pil.height), 'white')
            canvas.paste(grid_pil, (label_width, 0))
            draw = ImageDraw.Draw(canvas)
            try: font = ImageFont.truetype("arial.ttf", size=24)
            except IOError: font = ImageFont.load_default()
            row_height = grid_pil.height // 2
            labels = ["Originals:", "Reconstructed:"]
            y_positions = [(row_height * i) + (row_height // 2) - 12 for i in range(len(labels))]
            for i, label in enumerate(labels):
                draw.text((10, y_positions[i]), label, fill="black", font=font)
            img_path = os.path.join(cfg.CHECKPOINT_DIR, f"reconstruction_epoch_{epoch}_labeled.png")
            canvas.save(img_path)
            print(f"Saved labeled sample reconstruction grid to {img_path}")
        
        if cfg.EARLY_STOP_SSIM:
            if best_ssim >= cfg.EARLY_STOP_SSIM:
                print(f"\n--- Early stopping triggered! ---")
                print(f"Validation SSIM ({best_ssim:.4f}) has reached the target ({cfg.EARLY_STOP_SSIM}).")
                break

    plot_path = os.path.join(cfg.CHECKPOINT_DIR, f"{cfg.PROJECT_NAME}_training_progress.png")
    plot_training_progress(history, save_path=plot_path)

    print("\nTraining complete!")
    print(f"Best validation SSIM achieved: {best_ssim:.4f}")