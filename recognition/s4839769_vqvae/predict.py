# predict.py

"""
Performs a comprehensive evaluation of the trained VQ-VAE model on the entire test set.
- Calculates the overall average SSIM across all test images.
- Identifies and saves the 10 best and 10 worst reconstructions.
- Generates a batch of new images from the PixelCNN prior.
"""
import os
import torch
from torchvision.utils import make_grid
from torchmetrics.image import StructuralSimilarityIndexMeasure
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as transforms
from tqdm import tqdm
import numpy as np

from config import cfg
from modules import VQVAE, PixelCNN
from dataset import get_dataloaders

def save_comparison_grid(originals, reconstructions, title, save_path, num_images=10):
    """A helper function to save a labeled grid of original vs. reconstructed images."""
    # Ensure we only take up to num_images
    originals = originals[:num_images]
    reconstructions = reconstructions[:num_images]
    
    all_images_tensor = torch.cat([torch.stack(originals), torch.stack(reconstructions)])
    grid_tensor = make_grid(all_images_tensor.cpu(), nrow=num_images, normalize=True)
    grid_pil = transforms.ToPILImage()(grid_tensor)
    
    label_width = 180
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
        
    # Add a main title
    title_font = ImageFont.truetype("arialbd.ttf", size=30) if 'font' in locals() else font
    draw.text((label_width + 10, 10), title, fill="black", font=title_font)

    canvas.save(save_path)
    print(f"Saved comparison grid to: {save_path}")

def predict():
    """Main evaluation and generation function."""
    device = cfg.DEVICE
    print(f"Using device: {device}")
    output_dir = 'predictions'
    os.makedirs(output_dir, exist_ok=True)

    # --- 1. Load Both Trained Models ---
    vqvae_model_path = os.path.join(cfg.CHECKPOINT_DIR, "vqvae_best_model.pth")
    pixelcnn_model_path = os.path.join(cfg.CHECKPOINT_DIR, "pixelcnn_best_model.pth")
    
    if not os.path.exists(vqvae_model_path) or not os.path.exists(pixelcnn_model_path):
        print("ERROR: One or both model files not found. Please run 'train' and 'train_pixelcnn' modes first.")
        return

    vqvae_model = VQVAE(
        in_channels=cfg.IN_CHANNELS, hidden_channels=cfg.HIDDEN_CHANNELS, out_channels=cfg.IN_CHANNELS,
        num_res_blocks=cfg.NUM_RES_BLOCKS, res_channels=cfg.RES_CHANNELS, num_embeddings=cfg.NUM_EMBEDDINGS,
        embedding_dim=cfg.EMBEDDING_DIM, commitment_cost=cfg.COMMITMENT_COST
    ).to(device)
    vqvae_model.load_state_dict(torch.load(vqvae_model_path, weights_only=True))
    vqvae_model.eval()

    pixelcnn_model = PixelCNN(num_embeddings=cfg.NUM_EMBEDDINGS).to(device)
    pixelcnn_model.load_state_dict(torch.load(pixelcnn_model_path, weights_only=True))
    pixelcnn_model.eval()
    print("Successfully loaded both VQ-VAE and PixelCNN models.")

    # --- 2. Full Test Set Evaluation (NEW & IMPROVED) ---
    print("\n--- Starting Full Test Set Evaluation ---")
    _, _, test_loader = get_dataloaders()
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=2.0).to(device)
    
    results = [] # To store {'original': tensor, 'recon': tensor, 'ssim': float}

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating on Test Set"):
            batch = batch.to(device)
            _, reconstructions = vqvae_model(batch)
            
            # Iterate through each image in the batch to calculate individual SSIM
            for i in range(batch.size(0)):
                original_img = batch[i].unsqueeze(0)
                recon_img = reconstructions[i].unsqueeze(0)
                
                # Calculate SSIM for this single image
                score = ssim_metric(recon_img, original_img).item()
                
                # Store the results (move tensors to CPU to save GPU memory)
                results.append({
                    'original': original_img.squeeze(0).cpu(),
                    'recon': recon_img.squeeze(0).cpu(),
                    'ssim': score
                })

    # --- 3. Process and Report Results ---
    # Calculate overall average SSIM
    all_ssim_scores = [res['ssim'] for res in results]
    overall_avg_ssim = np.mean(all_ssim_scores)

    # Sort the results by SSIM score (lowest to highest)
    results.sort(key=lambda x: x['ssim'])

    # Get the 10 worst and 10 best
    worst_10 = results[:10]
    best_10 = results[-10:]
    
    # Save the visual results
    save_comparison_grid(
        [item['original'] for item in best_10],
        [item['recon'] for item in best_10],
        "Top 10 Best Reconstructions (by SSIM)",
        os.path.join(output_dir, "best_10_reconstructions.png")
    )
    save_comparison_grid(
        [item['original'] for item in worst_10],
        [item['recon'] for item in worst_10],
        "Top 10 Worst Reconstructions (by SSIM)",
        os.path.join(output_dir, "worst_10_reconstructions.png")
    )

    print("\n--- Test Set Evaluation Complete ---")
    print(f"Overall Average SSIM across {len(results)} test images: {overall_avg_ssim:.4f}")
    
    # --- 4. Generate New Images with PixelCNN (Unchanged) ---
    print("\n--- Starting Generation with PixelCNN (this will be slow) ---")
    with torch.no_grad():
        latent_h, latent_w = cfg.IMAGE_SIZE // 4, cfg.IMAGE_SIZE // 4
        num_generations = 8
        latent_map = torch.zeros(num_generations, latent_h, latent_w, dtype=torch.int64).to(device)
        
        for h in tqdm(range(latent_h), desc="Generating Rows"):
            for w in range(latent_w):
                output_logits = pixelcnn_model(latent_map)
                probs = torch.softmax(output_logits[:, :, h, w], dim=1)
                latent_map[:, h, w] = torch.multinomial(probs, 1).squeeze(-1)
                
        quantized_latents = vqvae_model.vq_layer.embedding(latent_map).permute(0, 3, 1, 2)
        generated_images = vqvae_model.decoder(quantized_latents)

    # Save the generated images
    save_image(generated_images.cpu(), os.path.join(output_dir, "pixelcnn_generated_images.png"), nrow=8, normalize=True)
    print(f"Saved PixelCNN generated images to: {os.path.join(output_dir, 'pixelcnn_generated_images.png')}")
    print("-" * 50)

if __name__ == '__main__':
    predict()