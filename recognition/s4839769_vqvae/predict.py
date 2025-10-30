# ======================================================================================
# AI Declaration
#
# This script was developed with the assistance of an AI language model.
#
# Tasks performed by the AI:
# - Wrote the initial script to load a trained model and reconstruct a single batch.
# - Implemented the autoregressive sampling loop to generate new images using the PixelCNN.
# - Implemented the logic to calculate per-image SSIM, sort the results, and identify the
#   10 best and 10 worst performing images.
#
# Prompt:
# "Create a Python script using PyTorch and Pillow to evaluate a trained VQ-VAE.
# It should iterate through the entire test set, calculate the SSIM for each image,
# and store the results. After, it should find the 10 best and 10 worst images based on SSIM.
# Finally, create and save a PNG that shows a grid of the 'best' original images
# and their reconstructions, with a main title above the grid and the 'subject ID' label
# centered above each individual image."
#
# LLM Used: Gemini 2.5 Pro
# ======================================================================================

"""
Performs a comprehensive evaluation of the trained VQ-VAE model on the entire test set.
- Calculates the overall average SSIM across all test images.
- Identifies and saves the 10 best and 10 worst reconstructions with per-image labels.
- Generates a batch of new images from the PixelCNN prior.
"""
import os
import torch
from torchvision.utils import make_grid, save_image
from torchmetrics.image import StructuralSimilarityIndexMeasure
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as transforms
from tqdm import tqdm
import numpy as np

from config import cfg
from modules import VQVAE, PixelCNN
from dataset import get_dataloaders

def save_comparison_grid(originals, reconstructions, filenames, subject_labels, title, save_path, num_images=10):
    """Saves a labeled grid comparing original and reconstructed images.

    This function creates a composite image with a main title, side labels ("Originals:", "Reconstructed:"),
    and per-image labels (subject ID and filename) for detailed analysis.

    Args:
        originals (list[torch.Tensor]): A list of original image tensors.
        reconstructions (list[torch.Tensor]): A list of reconstructed image tensors.
        filenames (list[str]): A list of filenames for each image.
        subject_labels (list[str]): A list of subject ID labels for each image.
        title (str): The main title to be displayed above the image grid.
        save_path (str): The file path where the final image will be saved.
        num_images (int, optional): The number of images to display in the grid. Defaults to 10.
    """
    originals = originals[:num_images]
    reconstructions = reconstructions[:num_images]
    filenames = filenames[:num_images]
    subject_labels = subject_labels[:num_images]
    
    all_images_tensor = torch.cat([torch.stack(originals), torch.stack(reconstructions)])
    grid_tensor = make_grid(all_images_tensor.cpu(), nrow=num_images, normalize=True, pad_value=1.0)
    grid_pil = transforms.ToPILImage()(grid_tensor)
    
    title_height = 60
    label_height = 60 # Make this taller to fit two lines of text
    sidelabel_width = 180
    
    canvas_width = grid_pil.width + sidelabel_width
    canvas_height = grid_pil.height + title_height + label_height
    canvas = Image.new('RGB', (canvas_width, canvas_height), 'white')
    
    canvas.paste(grid_pil, (sidelabel_width, title_height + label_height))
    
    draw = ImageDraw.Draw(canvas)
    try:
        title_font = ImageFont.truetype("arialbd.ttf", size=30)
        font = ImageFont.truetype("arial.ttf", size=24)
        label_font = ImageFont.truetype("arialbd.ttf", size=16) # Bold for subject
        micro_font = ImageFont.truetype("arial.ttf", size=12)  # Small for filename
    except IOError:
        title_font = font = label_font = micro_font = ImageFont.load_default()

    # Draw main title
    text_bbox = draw.textbbox((0, 0), title, font=title_font)
    text_width = text_bbox[2] - text_bbox[0]
    title_x = sidelabel_width + (grid_pil.width - text_width) / 2
    draw.text((title_x, 15), title, fill="black", font=title_font)

    # Draw per-image Subject Label
    image_width = grid_tensor.shape[2] + 2
    for i in range(num_images):
        # Subject Label (the "class")
        subj_label = subject_labels[i]
        label_bbox = draw.textbbox((0, 0), subj_label, font=label_font)
        label_width = label_bbox[2] - label_bbox[0]
        label_x = sidelabel_width + (i * image_width) + (image_width - label_width) / 2
        draw.text((label_x, title_height + 5), subj_label, fill="black", font=label_font)
        
        # Filename (for specific slice info)
        #fname = filenames[i].replace('.nii.gz', '')
        #fname_bbox = draw.textbbox((0, 0), fname, font=micro_font)
        #fname_width = fname_bbox[2] - fname_bbox[0]
        #fname_x = sidelabel_width + (i * image_width) + (image_width - fname_width) / 2
        #draw.text((fname_x, title_height + 30), fname, fill="gray", font=micro_font)

    # Draw side row labels
    row_height = grid_pil.height // 2
    side_labels = ["Originals:", "Reconstructed:"]
    y_positions = [title_height + label_height + (row_height * i) + (row_height // 2) - 12 for i in range(len(side_labels))]
    for i, label in enumerate(side_labels):
        draw.text((10, y_positions[i]), label, fill="black", font=font)

    canvas.save(save_path)
    print(f"Saved comparison grid to: {save_path}")


def predict():
    """Performs the main evaluation and generation pipeline.

    This function loads the trained VQ-VAE and PixelCNN models and performs three main tasks:
    1.  Evaluates the VQ-VAE's reconstruction performance on the entire test set, calculating an overall SSIM.
    2.  Generates and saves visualizations of the 10 best and 10 worst reconstructions.
    3.  Uses the PixelCNN to autoregressively generate a batch of new, novel images.
    """
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

    vqvae_model = VQVAE(in_channels=cfg.IN_CHANNELS, hidden_channels=cfg.HIDDEN_CHANNELS, out_channels=cfg.IN_CHANNELS,
                        num_res_blocks=cfg.NUM_RES_BLOCKS, res_channels=cfg.RES_CHANNELS, num_embeddings=cfg.NUM_EMBEDDINGS,
                        embedding_dim=cfg.EMBEDDING_DIM, commitment_cost=cfg.COMMITMENT_COST).to(device)
    vqvae_model.load_state_dict(torch.load(vqvae_model_path, weights_only=True))
    vqvae_model.eval()

    pixelcnn_model = PixelCNN(num_embeddings=cfg.NUM_EMBEDDINGS).to(device)
    pixelcnn_model.load_state_dict(torch.load(pixelcnn_model_path, weights_only=True))
    pixelcnn_model.eval()
    print("Successfully loaded both VQ-VAE and PixelCNN models.")

    # --- 2. Full Test Set Evaluation ---
    print("\n--- Starting Full Test Set Evaluation ---")
    _, _, test_loader = get_dataloaders()
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=2.0).to(device)
    
    results = [] 

    with torch.no_grad():
        # --- KEY CHANGE: Unpack all three items from the loader ---
        for batch_images, batch_filenames, batch_labels in tqdm(test_loader, desc="Evaluating on Test Set"):
            batch_images = batch_images.to(device)
            _, reconstructions = vqvae_model(batch_images)
            
            for i in range(batch_images.size(0)):
                original_img = batch_images[i].unsqueeze(0)
                recon_img = reconstructions[i].unsqueeze(0)
                score = ssim_metric(recon_img, original_img).item()
                
                #  Store the subject label in the results
                results.append({
                    'original': original_img.squeeze(0).cpu(),
                    'recon': recon_img.squeeze(0).cpu(),
                    'ssim': score,
                    'filename': batch_filenames[i],
                    'label': batch_labels[i] 
                })

    # Process and Report Results
    all_ssim_scores = [res['ssim'] for res in results]
    overall_avg_ssim = np.mean(all_ssim_scores)
    results.sort(key=lambda x: x['ssim'])
    worst_10 = results[:10]
    best_10 = results[-10:]
    
    # Pass the subject labels to the saving function
    save_comparison_grid(
        [item['original'] for item in best_10], [item['recon'] for item in best_10],
        [item['filename'] for item in best_10], [item['label'] for item in best_10],
        "Top 10 Best Reconstructions (by SSIM)",
        os.path.join(output_dir, "best_10_reconstructions.png")
    )
    save_comparison_grid(
        [item['original'] for item in worst_10], [item['recon'] for item in worst_10],
        [item['filename'] for item in worst_10], [item['label'] for item in worst_10],
        "Top 10 Worst Reconstructions (by SSIM)",
        os.path.join(output_dir, "worst_10_reconstructions.png")
    )


    print("\n--- Test Set Evaluation Complete ---")
    print(f"Overall Average SSIM across {len(results)} test images: {overall_avg_ssim:.4f}")
    
    # --- 4. Generate New Images with PixelCNN ---
    print("\n--- Starting Generation with PixelCNN ---")
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

    save_image(generated_images.cpu(), os.path.join(output_dir, "pixelcnn_generated_images.png"), nrow=8, normalize=True)
    print(f"Saved PixelCNN generated images to: {os.path.join(output_dir, 'pixelcnn_generated_images.png')}")
    print("-" * 50)

if __name__ == '__main__':
    predict()