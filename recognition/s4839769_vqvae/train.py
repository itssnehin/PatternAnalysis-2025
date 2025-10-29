# train.py

"""
The core training and validation loop for the VQ-VAE model.
- Uses a combined MSE + Perceptual (LPIPS) loss for high-quality reconstructions.
- Evaluates the model using SSIM on a validation set.
- Saves the best model based on SSIM and periodic, labeled image samples.
- Includes an early stopping mechanism based on a target SSIM score.
"""
import os
import torch
import torch.nn.functional as F
from torchvision.utils import make_grid
from torchmetrics import StructuralSimilarityIndexMeasure
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as transforms
# --- NEW IMPORT FOR PERCEPTUAL LOSS ---
import lpips

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

    # --- 1. Initialize Model and Optimizer ---
    model = VQVAE(
        in_channels=cfg.IN_CHANNELS, hidden_channels=cfg.HIDDEN_CHANNELS,
        out_channels=cfg.IN_CHANNELS, num_res_blocks=cfg.NUM_RES_BLOCKS,
        res_channels=cfg.RES_CHANNELS, num_embeddings=cfg.NUM_EMBEDDINGS,
        embedding_dim=cfg.EMBEDDING_DIM, commitment_cost=cfg.COMMITMENT_COST
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE)

    # --- 2. Initialize DataLoaders and Loss Functions ---
    try:
        train_loader, val_loader = get_dataloaders()
    except (ValueError, FileNotFoundError) as e:
        print(f"ERROR: Could not load data. {e}")
        return

    # --- INITIALIZE THE PERCEPTUAL LOSS (LPIPS) ---
    # It will download the VGG model weights automatically on first run.
    # Set verbose=False to prevent it from printing download progress every time.
    print("Initializing LPIPS perceptual loss function...")
    lpips_loss_fn = lpips.LPIPS(net='vgg', verbose=False).to(device)
    
    # --- Initialize Metrics and Tracking Variables ---
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=2.0).to(device)
    best_ssim = 0.0
    fixed_val_images = next(iter(val_loader)).to(device)

    # --- 3. The Main Training Loop ---
    print("Starting training...")
    for epoch in range(1, cfg.EPOCHS + 1):
        # --- Training Phase ---
        model.train()
        train_total_loss = 0.0
        
        # --- CORRECTED TRAINING LOOP LOGIC ---
        for batch_idx, data in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.EPOCHS} [Training]")):
            data = data.to(device)
            optimizer.zero_grad()

            vq_loss, data_recon = model(data)
            
            # Calculate the two parts of the reconstruction loss
            mse_recon_loss = F.mse_loss(data_recon, data)
            # The LPIPS function expects inputs in range [-1, 1], which our data already is.
            # .mean() is critical as LPIPS returns a loss per image in the batch.
            perceptual_loss = lpips_loss_fn(data_recon, data).mean()
            
            # Combine the reconstruction losses. You can add a weight here if desired.
            # For now, we'll give them equal importance.
            recon_loss = mse_recon_loss + perceptual_loss
            
            # Final total loss to backpropagate
            loss = recon_loss + vq_loss
            
            # --- CRITICAL FIXES ARE HERE ---
            loss.backward()
            optimizer.step()
            
            train_total_loss += loss.item() # Track the total loss
        
        avg_train_loss = train_total_loss / len(train_loader)

        # --- Validation Phase ---
        model.eval()
        val_recon_loss = 0.0
        
        with torch.no_grad():
            for data in tqdm(val_loader, desc=f"Epoch {epoch}/{cfg.EPOCHS} [Validation]"):
                data = data.to(device)
                _, data_recon = model(data)
                
                # We only need MSE for the validation loss printout, but SSIM is the key metric
                recon_loss = F.mse_loss(data_recon, data)
                val_recon_loss += recon_loss.item()
                
                ssim_metric.update(data_recon, data)

        avg_val_loss = val_recon_loss / len(val_loader)
        epoch_ssim = ssim_metric.compute()
        ssim_metric.reset()

        print(
            f"Epoch: {epoch}/{cfg.EPOCHS} | "
            f"Train Total Loss: {avg_train_loss:.4f} | "
            f"Val Recon Loss (MSE): {avg_val_loss:.4f} | "
            f"Val SSIM: {epoch_ssim:.4f}"
        )

        # --- 4. Checkpointing and Saving ---
        if epoch_ssim > best_ssim:
            best_ssim = epoch_ssim
            model_path = os.path.join(cfg.CHECKPOINT_DIR, "vqvae_best_model.pth")
            torch.save(model.state_dict(), model_path)
            print(f"✨ New best model saved with SSIM: {best_ssim:.4f} ✨")

        # (Image saving logic remains the same)
        if epoch % cfg.SAVE_IMAGE_EPOCH == 0 or epoch == cfg.EPOCHS:
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

        # --- 5. EARLY STOPPING CHECK ---
        if best_ssim >= cfg.EARLY_STOP_SSIM:
            print(f"\n--- Early stopping triggered! ---")
            print(f"Validation SSIM ({best_ssim:.4f}) has reached the target ({cfg.EARLY_STOP_SSIM}).")
            break

    print("\nTraining complete!")
    print(f"Best validation SSIM achieved: {best_ssim:.4f}")