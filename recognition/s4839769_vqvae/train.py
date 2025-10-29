# train.py

"""
The core training and validation loop for the VQ-VAE model.
- Uses a WEIGHTED combined MSE + SSIM loss for stable training.
- Saves the best model based on SSIM and periodic, labeled image samples.
- Includes a robust early stopping mechanism that saves a final image.
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

from config import cfg
from modules import VQVAE
from dataset import get_dataloaders

def plot_training_progress(history, save_path):
    """Plots and saves the training and validation loss, and validation SSIM."""
    # (This function is correct and does not need changes)
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

def save_reconstruction_image(model, images, epoch_label, save_dir):
    """A helper function to generate and save a labeled reconstruction image."""
    with torch.no_grad():
        _, reconstructed_samples = model(images)
    all_images_tensor = torch.cat([images[:8], reconstructed_samples[:8]])
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
    
    # Use the epoch_label in the filename
    img_path = os.path.join(save_dir, f"reconstruction_{epoch_label}_labeled.png")
    canvas.save(img_path)
    print(f"Saved labeled sample reconstruction grid to {img_path}")

def train_model():
    """Main function to orchestrate the VQ-VAE training process."""
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

    ssim_loss_fn = StructuralSimilarityIndexMeasure(data_range=2.0).to(device)
    ssim_val_metric = StructuralSimilarityIndexMeasure(data_range=2.0).to(device)
    best_ssim = 0.0
    fixed_val_images = next(iter(val_loader)).to(device)
    history = {'train_loss': [], 'val_loss': [], 'val_ssim': []}
    
    # --- NEW: Flag to track if we stopped early ---
    early_stop_triggered = False

    print("Starting training with WEIGHTED combined MSE + SSIM loss...")
    for epoch in range(1, cfg.EPOCHS + 1):
        # (Training and Validation Phases are unchanged)
        model.train()
        train_total_loss = 0.0
        for data in tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.EPOCHS} [Training]"):
            data = data.to(device)
            optimizer.zero_grad()
            vq_loss, data_recon = model(data)
            mse_loss = F.mse_loss(data_recon, data)
            ssim_loss = 1.0 - ssim_loss_fn(data_recon, data)
            alpha = cfg.ALPHA
            recon_loss = mse_loss + (alpha * ssim_loss)
            loss = recon_loss + vq_loss
            loss.backward()
            optimizer.step()
            train_total_loss += loss.item()
        avg_train_loss = train_total_loss / len(train_loader)

        model.eval()
        val_recon_loss = 0.0
        with torch.no_grad():
            for data in tqdm(val_loader, desc=f"Epoch {epoch}/{cfg.EPOCHS} [Validation]"):
                data = data.to(device)
                _, data_recon = model(data)
                val_recon_loss += F.mse_loss(data_recon, data).item()
                ssim_val_metric.update(data_recon, data)
        avg_val_loss = val_recon_loss / len(val_loader)
        epoch_ssim = ssim_val_metric.compute()
        ssim_val_metric.reset()

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

        # Save periodic images
        if epoch % cfg.SAVE_IMAGE_EPOCH == 0:
            save_reconstruction_image(model, fixed_val_images, f"epoch_{epoch:03d}", cfg.CHECKPOINT_DIR)

        # Check for early stopping
        if cfg.EARLY_STOP_SSIM:
            if best_ssim >= cfg.EARLY_STOP_SSIM:
                print(f"\n--- Early stopping triggered! ---")
                print(f"Validation SSIM ({best_ssim:.4f}) has reached the target ({cfg.EARLY_STOP_SSIM}).")
                early_stop_triggered = True
                break

    # --- NEW LOGIC AFTER THE LOOP ---
    # Always save a final image, labeled appropriately
    final_epoch_label = f"epoch_{epoch:03d}"
    if early_stop_triggered:
        final_epoch_label += "_final_early_stop"
    else: # This handles the case where all epochs complete
        final_epoch_label += "_final"
    
    print(f"\nSaving final reconstruction image for epoch {epoch}...")
    # Load the BEST model to ensure the final image reflects the best performance
    model.load_state_dict(torch.load(os.path.join(cfg.CHECKPOINT_DIR, "vqvae_best_model.pth")))
    save_reconstruction_image(model, fixed_val_images, final_epoch_label, cfg.CHECKPOINT_DIR)

    plot_path = os.path.join(cfg.CHECKPOINT_DIR, f"{cfg.PROJECT_NAME}_training_progress.png")
    plot_training_progress(history, save_path=plot_path)

    print("\nTraining complete!")
    print(f"Best validation SSIM achieved: {best_ssim:.4f}")