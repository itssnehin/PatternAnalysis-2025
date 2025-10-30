# train.py
"""
The core training and validation loop for the VQ-VAE model.
- Uses a simple, stable MSE loss for reconstruction.
- NO learning rate scheduler is used (fixed LR).
- Evaluates the model using SSIM on a validation set.
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

# UPDATED PLOTTING FUNCTION
def plot_training_progress(history, save_path):
    """
    Plots and saves the training history, including linear and log loss, and SSIM.
    """
    print(f"Plotting training progress to {save_path}...")
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Create a figure with THREE subplots, and make it taller
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
    
    # --- Subplot 1: Loss (Linear Scale) ---
    ax1.plot(epochs, history['train_loss'], 'bo-', label='Training Total Loss')
    ax1.plot(epochs, history['val_loss'], 'ro-', label='Validation Loss (MSE)')
    ax1.set_title('Training and Validation Loss (Linear Scale)')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # --- Subplot 2: Loss (Logarithmic Scale) ---
    ax2.plot(epochs, history['train_loss'], 'bo-', label='Training Total Loss')
    ax2.plot(epochs, history['val_loss'], 'ro-', label='Validation Loss (MSE)')
    ax2.set_yscale('log') # This is the key change for this subplot
    ax2.set_title('Training and Validation Loss (Logarithmic Scale)')
    ax2.set_ylabel('Loss (log)')
    ax2.legend()
    ax2.grid(True, which='both') # Use 'both' for major and minor grid lines on log scale
    
    # --- Subplot 3: SSIM ---
    ax3.plot(epochs, history['val_ssim'], 'go-', label='Validation SSIM')
    ax3.set_title('Validation SSIM')
    ax3.set_xlabel('Epochs')
    ax3.set_ylabel('SSIM Score')
    ax3.legend()
    ax3.grid(True)
    
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
    
    img_path = os.path.join(save_dir, f"reconstruction_{epoch_label}_labeled.png")
    canvas.save(img_path)
    print(f"Saved labeled sample reconstruction grid to {img_path}")

def train_model(train_loader, val_loader):
    """
    Main function to orchestrate the VQ-VAE training process.
    This version uses a simple MSE loss and a fixed learning rate to establish a stable baseline.
    """
    # SETUP
    # Get the device (CUDA or CPU) from the config file and print it.
    device = cfg.DEVICE
    print(f"Using device: {device}")
    # Create the directory for saving model checkpoints if it doesn't already exist.
    os.makedirs(cfg.CHECKPOINT_DIR, exist_ok=True)

    # INITIALIZE MODEL AND OPTIMIZER
    # Create an instance of the VQVAE model from modules.py, using hyperparameters from the config.
    model = VQVAE(
        in_channels=cfg.IN_CHANNELS, hidden_channels=cfg.HIDDEN_CHANNELS,
        out_channels=cfg.IN_CHANNELS, num_res_blocks=cfg.NUM_RES_BLOCKS,
        res_channels=cfg.RES_CHANNELS, num_embeddings=cfg.NUM_EMBEDDINGS,
        embedding_dim=cfg.EMBEDDING_DIM, commitment_cost=cfg.COMMITMENT_COST
    ).to(device) # Move the model to the selected device (GPU or CPU).

    # Initialize the Adam optimizer. It will update the model's parameters.
    # The learning rate is fixed and taken from the config file.
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE)
    
    # The learning rate scheduler has been removed in this version for simplicity.


    # INITIALIZE METRICS AND TRACKING VARIABLES 
    # Initialize the SSIM metric calculator for the validation phase.
    ssim_val_metric = StructuralSimilarityIndexMeasure(data_range=2.0).to(device)
    # Variable to keep track of the best SSIM score seen so far.
    best_ssim = 0.0
    # Get a fixed batch of validation images to generate consistent visual samples throughout training.
    fixed_val_images = next(iter(val_loader)).to(device)
    # A dictionary to store the loss and SSIM from each epoch for final plotting.
    history = {'train_loss': [], 'val_loss': [], 'val_ssim': []}
    # A flag to check if the training loop was exited due to early stopping.
    early_stop_triggered = False

    print("Starting VANILLA training with pure MSE loss...")
    # MAIN TRAINING LOOP
    # Loop through the total number of epochs specified in the config.
    for epoch in range(1, cfg.EPOCHS + 1):
        
        #  LEARNING RATE WARM-UP ---
        if epoch <= cfg.WARMUP_EPOCHS:
            # Linearly increase the LR from a small value to the target LR
            new_lr = cfg.LEARNING_RATE * (epoch / cfg.WARMUP_EPOCHS)
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lr
        
        current_lr = optimizer.param_groups[0]['lr']
        # TRAINING PHASE
        # Set the model to training mode. This enables layers like BatchNorm (if they existed).
        model.train()
        # Variable to accumulate the total loss over the training epoch.
        train_total_loss = 0.0
        
        # Loop through each batch of data in the training loader.
        for data in tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.EPOCHS} [Training, LR={current_lr:.6f}]"):
            # Move the batch of images to the selected device.
            data = data.to(device)
            # Reset the gradients of all model parameters before calculating new ones.
            optimizer.zero_grad()

            # Perform a forward pass: get the VQ loss and the reconstructed images from the model.
            vq_loss, data_recon = model(data)
            
            # Calculate the reconstruction loss using Mean Squared Error (MSE).
            recon_loss = F.mse_loss(data_recon, data)
            
            # The total loss is the sum of the reconstruction loss and the VQ-related losses.
            loss = recon_loss + vq_loss
            
            # Perform backpropagation: calculate the gradients of the loss with respect to model parameters.
            loss.backward()
            # Clip the gradients to a maximum norm of 1.0. This is the safety valve.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update the model's weights using the calculated gradients.
            optimizer.step()
            
            # Add the loss for this batch to the running total for the epoch.
            train_total_loss += loss.item()
        
        # Calculate the average training loss for the entire epoch.
        avg_train_loss = train_total_loss / len(train_loader)

        # VALIDATION PHASE
        # Set the model to evaluation mode. This disables layers like Dropout (if they existed).
        model.eval()
        # Variable to accumulate the validation loss.
        val_recon_loss = 0.0
        
        # Disable gradient calculation to speed up validation and save memory.
        with torch.no_grad():
            # Loop through each batch in the validation loader.
            for data in tqdm(val_loader, desc=f"Epoch {epoch}/{cfg.EPOCHS} [Validation]"):
                data = data.to(device)
                # Perform a forward pass to get the reconstructed images.
                _, data_recon = model(data)
                # Calculate and accumulate the MSE loss for this batch.
                val_recon_loss += F.mse_loss(data_recon, data).item()
                # Update the SSIM metric with the original and reconstructed images for this batch.
                ssim_val_metric.update(data_recon, data)

        # Calculate the average validation MSE loss for the epoch.
        avg_val_loss = val_recon_loss / len(val_loader)
        # Compute the final SSIM score over all batches in the validation set.
        epoch_ssim = ssim_val_metric.compute()
        # Reset the SSIM metric calculator for the next epoch.
        ssim_val_metric.reset()
        
        # The learning rate scheduler step has been removed in this version.

        # LOGGING AND SAVING
        # Print a summary of the epoch's performance.
        print(
            f"Epoch: {epoch}/{cfg.EPOCHS} | "
            f"Train Total Loss: {avg_train_loss:.4f} | "
            f"Val Recon Loss (MSE): {avg_val_loss:.4f} | "
            f"Val SSIM: {epoch_ssim:.4f}"
        )
        
        # Store the metrics for this epoch in the history dictionary for final plotting.
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_ssim'].append(epoch_ssim.item())

        # Check if the current model has the best SSIM score seen so far.
        if epoch_ssim > best_ssim:
            best_ssim = epoch_ssim
            # If so, save the model's state dictionary to a file.
            model_path = os.path.join(cfg.CHECKPOINT_DIR, "vqvae_best_model.pth")
            torch.save(model.state_dict(), model_path)
            print(f"✨ New best model saved with SSIM: {best_ssim:.4f} ✨")

        # Periodically save a sample reconstruction image based on the frequency in the config.
        if epoch % cfg.SAVE_IMAGE_EPOCH == 0:
            save_reconstruction_image(model, fixed_val_images, f"epoch_{epoch:03d}", cfg.CHECKPOINT_DIR)

        # Check if the best SSIM has reached the early stopping target.
        if cfg.EARLY_STOP_SSIM:
            if best_ssim >= cfg.EARLY_STOP_SSIM:
                print(f"\n--- Early stopping triggered! ---")
                print(f"Validation SSIM ({best_ssim:.4f}) has reached the target ({cfg.EARLY_STOP_SSIM}).")
                early_stop_triggered = True
                break # Exit the main training loop.

    # FINAL ACTIONS AFTER TRAINING
    # Determine a label for the final image based on whether training finished or stopped early.
    final_epoch_label = f"epoch_{epoch:03d}"
    if early_stop_triggered:
        final_epoch_label += "_final_early_stop"
    else:
        final_epoch_label += "_final"
    
    print(f"\nSaving final reconstruction image for epoch {epoch}...")
    # Load the best performing model's weights before generating the final image.
    model.load_state_dict(torch.load(os.path.join(cfg.CHECKPOINT_DIR, "vqvae_best_model.pth"), weights_only=True))
    # Call the helper function to save the final labeled image.
    save_reconstruction_image(model, fixed_val_images, final_epoch_label, cfg.CHECKPOINT_DIR)

    # Generate and save the plot of the entire training history.
    plot_path = os.path.join(cfg.CHECKPOINT_DIR, f"{cfg.PROJECT_NAME}_training_progress.png")
    plot_training_progress(history, save_path=plot_path)

    # Print a final summary message.
    print("\nTraining complete!")
    print(f"Best validation SSIM achieved: {best_ssim:.4f}")