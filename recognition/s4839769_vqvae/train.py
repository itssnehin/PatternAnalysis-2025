"""
Main script for training the VQ-VAE model on the HipMRI dataset.
"""
import os
import torch
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from torchvision.utils import save_image

# Import our custom modules
from config import cfg
from modules import VQVAE
from dataset import get_dataloaders

def setup_environment():
    """Sets up the environment for training."""
    torch.manual_seed(cfg.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(cfg.SEED)
    
    # Create the directory for saving model checkpoints and reconstructed images
    os.makedirs(cfg.CHECKPOINT_DIR, exist_ok=True)
    print(f"Using device: {cfg.DEVICE}")
    print(f"Checkpoint directory created at: {cfg.CHECKPOINT_DIR}")

def train_model():
    """The core training function."""
    setup_environment()

    # 1. Initialize Model, Optimizer, and DataLoaders
    model = VQVAE(
        in_channels=cfg.IN_CHANNELS,
        hidden_channels=cfg.HIDDEN_CHANNELS,
        out_channels=cfg.IN_CHANNELS,
        num_res_blocks=cfg.NUM_RES_BLOCKS,
        res_channels=cfg.RES_CHANNELS,
        num_embeddings=cfg.NUM_EMBEDDINGS,
        embedding_dim=cfg.EMBEDDING_DIM,
        commitment_cost=cfg.COMMITMENT_COST
    ).to(cfg.DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE)
    
    train_loader, val_loader = get_dataloaders()
    
    print(f"Model initialized with {sum(p.numel() for p in model.parameters()):,} parameters.")

    # Get a fixed batch from the validation set to monitor progress
    fixed_val_images = next(iter(val_loader)).to(cfg.DEVICE)

    # 2. The Training Loop
    for epoch in range(cfg.EPOCHS):
        model.train()  # Set the model to training mode
        
        # Use tqdm for a nice progress bar
        loop = tqdm(train_loader, leave=True)
        loop.set_description(f"Epoch [{epoch+1}/{cfg.EPOCHS}]")
        
        total_recon_loss = 0
        total_vq_loss = 0

        for batch_idx, data in enumerate(loop):
            data = data.to(cfg.DEVICE)
            optimizer.zero_grad()

            # Forward pass
            vq_loss, data_recon = model(data)
            
            # Calculate the reconstruction loss (MSE is a common choice)
            recon_loss = F.mse_loss(data_recon, data)
            
            # The total loss is the sum of reconstruction loss and VQ loss
            total_loss = recon_loss + vq_loss
            
            # Backward pass and optimization
            total_loss.backward()
            optimizer.step()
            
            # Update running losses for logging
            total_recon_loss += recon_loss.item()
            total_vq_loss += vq_loss.item()
            
            # Update the progress bar with live loss values
            if batch_idx % cfg.LOG_FREQ == 0:
                loop.set_postfix(
                    recon_loss=f"{recon_loss.item():.4f}", 
                    vq_loss=f"{vq_loss.item():.4f}"
                )

        # 3. Logging and Visualization at the end of each epoch
        avg_recon_loss = total_recon_loss / len(train_loader)
        avg_vq_loss = total_vq_loss / len(train_loader)
        print(f"\nEpoch {epoch+1} Summary: Avg Recon Loss: {avg_recon_loss:.4f}, Avg VQ Loss: {avg_vq_loss:.4f}")

        # Save reconstructed images periodically
        if (epoch + 1) % cfg.SAVE_IMAGE_EPOCH == 0:
            model.eval()  # Set the model to evaluation mode
            with torch.no_grad():
                # Get reconstructions for our fixed validation batch
                _, recon_images = model(fixed_val_images)
                
                # Combine original and reconstructed images for comparison
                comparison = torch.cat([fixed_val_images[:8], recon_images[:8]])
                
                save_path = os.path.join(cfg.CHECKPOINT_DIR, f"reconstruction_epoch_{epoch+1}.png")
                save_image(comparison.cpu(), save_path, nrow=8)
                print(f"Saved sample reconstructions to {save_path}")

    # 4. Save the final trained model
    final_model_path = os.path.join(cfg.CHECKPOINT_DIR, "vqvae_final_model.pth")
    torch.save(model.state_dict(), final_model_path)
    print(f"\nTraining complete. Final model saved to {final_model_path}")

if __name__ == '__main__':
    train_model()