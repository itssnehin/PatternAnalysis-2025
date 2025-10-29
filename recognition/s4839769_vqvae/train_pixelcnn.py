# train_pixelcnn.py

"""
The training script for the PixelCNN prior model.
- Loads a pre-trained VQ-VAE model.
- Freezes the VQ-VAE and uses it to generate a dataset of discrete latent codes.
- Trains the PixelCNN to model the distribution of these codes.
- Saves the best PixelCNN model based on its loss.
"""
import os
import torch
import torch.nn.functional as F
from tqdm import tqdm

from config import cfg
from modules import VQVAE, PixelCNN # Import both models
from dataset import get_dataloaders

def train_pixelcnn():
    device = cfg.DEVICE
    print(f"Using device: {device}")

    # --- 1. Load the PRE-TRAINED VQ-VAE ---
    vqvae_model_path = os.path.join(cfg.CHECKPOINT_DIR, "vqvae_best_model.pth")
    if not os.path.exists(vqvae_model_path):
        print(f"ERROR: VQ-VAE model not found at {vqvae_model_path}. Please run 'main.py train' first.")
        return

    vqvae_model = VQVAE(
        in_channels=cfg.IN_CHANNELS, hidden_channels=cfg.HIDDEN_CHANNELS,
        out_channels=cfg.IN_CHANNELS, num_res_blocks=cfg.NUM_RES_BLOCKS,
        res_channels=cfg.RES_CHANNELS, num_embeddings=cfg.NUM_EMBEDDINGS,
        embedding_dim=cfg.EMBEDDING_DIM, commitment_cost=cfg.COMMITMENT_COST
    ).to(device)
    vqvae_model.load_state_dict(torch.load(vqvae_model_path, weights_only=True))
    vqvae_model.eval() # Freeze the VQ-VAE
    print(f"Successfully loaded and froze VQ-VAE model.")

    # --- 2. Initialize PixelCNN, Optimizer, and Data ---
    pixelcnn_model = PixelCNN(num_embeddings=cfg.NUM_EMBEDDINGS).to(device)
    optimizer = torch.optim.Adam(pixelcnn_model.parameters(), lr=cfg.LEARNING_RATE)
    
    # We only need the training data loader
    train_loader, _ = get_dataloaders()
    
    # --- 3. Training Loop for PixelCNN ---
    best_loss = float('inf')
    # The number of epochs for PixelCNN can be different, let's use a config value
    # or a hardcoded value for simplicity. 50 epochs is a good start.
    num_epochs = 50 
    
    print("Starting PixelCNN training...")
    for epoch in range(1, num_epochs + 1):
        pixelcnn_model.train()
        total_loss = 0
        
        for data in tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}"):
            data = data.to(device)
            optimizer.zero_grad()
            
            # Use the VQ-VAE to get the discrete code indices (our targets)
            with torch.no_grad():
                target_indices = vqvae_model.get_code_indices(data) # Shape: [B, H, W]
            
            # Get the PixelCNN's predictions for these indices
            output_logits = pixelcnn_model(target_indices) # Shape: [B, C, H, W]
            
            # The loss is Cross-Entropy between the predictions and the actual next index
            loss = F.cross_entropy(output_logits, target_indices)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch: {epoch}/{num_epochs} | Average Loss: {avg_loss:.4f}")
        
        # Save the best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            model_path = os.path.join(cfg.CHECKPOINT_DIR, "pixelcnn_best_model.pth")
            torch.save(pixelcnn_model.state_dict(), model_path)
            print(f"✨ New best PixelCNN model saved with loss: {best_loss:.4f} ✨")
            
    print("\nPixelCNN training complete!")