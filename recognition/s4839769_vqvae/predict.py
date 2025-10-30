# predict.py

"""
Demonstrates the usage of a trained VQ-VAE model.
- Reconstructs a batch of validation images and reports SSIM.
- Generates a batch of new images from a random latent prior.
- Saves a final, labeled comparison grid showing all three image types.
"""
import os
import torch
from torchvision.utils import make_grid
from torchmetrics import StructuralSimilarityIndexMeasure
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as transforms

from config import cfg
from modules import VQVAE
from dataset import get_dataloaders

def predict():
    """Loads a trained model, generates reconstructions and new images, and reports SSIM."""
    device = cfg.DEVICE
    print(f"Using device: {device}")

    # --- 1. Load the Best Trained Model ---
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

    # --- 2. Get Data for Reconstruction ---
    try:
        # get_dataloaders now returns three items. We only need the third one (test_loader).
        _, _, test_loader = get_dataloaders()
        original_images = next(iter(test_loader))[:8].to(device) # We'll use 8 images
    except (ValueError, FileNotFoundError) as e:
        print(f"ERROR: Could not load test data. {e}")
        return

        
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=2.0).to(device)
        
    # --- 3. Generate Reconstructions and Calculate SSIM ---
    print("Generating reconstructions...")
    with torch.no_grad():
        _, reconstructed_images = model(original_images)
        batch_ssim = ssim_metric(reconstructed_images, original_images)

    # --- 4. Generate New Images from Scratch ---
    print("Generating new images from a random prior...")
    with torch.no_grad():
        latent_h = cfg.IMAGE_SIZE // 4
        latent_w = cfg.IMAGE_SIZE // 4
        num_generations = 8
        random_indices = torch.randint(
            low=0, high=cfg.NUM_EMBEDDINGS,
            size=(num_generations, latent_h, latent_w), device=device
        )
        quantized_latents = model.vq_layer.embedding(random_indices)
        quantized_latents = quantized_latents.permute(0, 3, 1, 2).contiguous()
        generated_images = model.decoder(quantized_latents)

    # --- 5. Create and Save the Labeled Comparison Image (NEW LOGIC) ---
    print("Creating labeled comparison image...")
    
    all_images_tensor = torch.cat([original_images, reconstructed_images, generated_images])
    grid_tensor = make_grid(all_images_tensor.cpu(), nrow=8, normalize=True)
    grid_pil = transforms.ToPILImage()(grid_tensor)
    
    label_width = 200 # Adjusted space
    canvas = Image.new('RGB', (grid_pil.width + label_width, grid_pil.height), 'white')
    canvas.paste(grid_pil, (label_width, 0))
    
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("arial.ttf", size=24) # Smaller font
    except IOError:
        font = ImageFont.load_default()

    row_height = grid_pil.height // 3
    labels = ["Originals:", "Reconstructed:", "Generated:"]
    y_positions = [(row_height * i) + (row_height // 2) - 12 for i in range(len(labels))]
    
    for i, label in enumerate(labels):
        draw.text((10, y_positions[i]), label, fill="black", font=font)
        
    output_dir = 'predictions'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{cfg.PROJECT_NAME}_generation_result_labeled.png")
    canvas.save(output_path)

    
    print("-" * 50)
    print("Prediction and Generation complete!")
    print(f"SSIM for reconstructed batch: {batch_ssim.item():.4f}")
    print(f"Saved labeled comparison image to: {output_path}")
    print("-" * 50)

if __name__ == '__main__':
    predict()