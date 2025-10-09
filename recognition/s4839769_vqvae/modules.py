"""
Core components of the Vector-Quantized Variational Autoencoder (VQ-VAE) model.
Contains the Encoder, Decoder, VectorQuantizer, and the main VQVAE module.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    """
    The Encoder network compresses the input image into a lower-dimensional latent space.
    """
    def __init__(self, in_channels, hidden_channels, num_res_blocks, res_channels):
        super(Encoder, self).__init__()
        self.conv_in = nn.Conv2d(in_channels, hidden_channels // 2, kernel_size=4, stride=2, padding=1)
        
        self.layers = nn.ModuleList([
            nn.Conv2d(hidden_channels // 2, hidden_channels, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, stride=1, padding=1),
            # Residual Blocks
            # You can add more complex residual blocks here if needed
        ])
        
        self.conv_out = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.conv_in(x)
        for layer in self.layers:
            x = layer(x)
        return self.conv_out(x)

class Decoder(nn.Module):
    """
    The Decoder network reconstructs the image from the quantized latent space.
    """
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(Decoder, self).__init__()
        
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, stride=1, padding=1),
            # You can add more complex residual blocks here if needed
            nn.ConvTranspose2d(hidden_channels, hidden_channels // 2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(hidden_channels//2, out_channels, kernel_size=4, stride=2, padding=1)
        )

    def forward(self, x):
        return self.layers(x)

class VectorQuantizer(nn.Module):
    """
    The core Vector Quantizer layer.
    This layer takes the continuous output of the encoder and maps it to the closest
    vector in a learned discrete codebook.
    """
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VectorQuantizer, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        
        # Initialize the codebook embeddings
        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embedding.weight.data.uniform_(-1./self.num_embeddings, 1./self.num_embeddings)

    def forward(self, inputs):
        # inputs shape: [B, C, H, W]
        # Rearrange input to [B, H, W, C]
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape
        
        # Flatten input to [B*H*W, C]
        flat_input = inputs.view(-1, self.embedding_dim)
        
        # Calculate distances between each input vector and all embedding vectors
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self.embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self.embedding.weight.t()))
            
        # Find the closest embedding for each input vector
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # Quantize the input by replacing each vector with its closest embedding
        quantized = torch.matmul(encodings, self.embedding.weight).view(input_shape)
        
        # --- Calculate Loss ---
        # The VQ loss has two parts:
        # 1. e_loss (codebook loss): Pushes the embeddings to be closer to the encoder outputs.
        e_loss = F.mse_loss(quantized.detach(), inputs)
        # 2. q_loss (commitment loss): Pushes the encoder outputs to be closer to their chosen embeddings.
        q_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_loss + self.commitment_cost * e_loss
        
        # Use Straight-Through Estimator to allow gradients to flow back to the encoder
        quantized = inputs + (quantized - inputs).detach()
        
        # Reshape quantized back to [B, C, H, W]
        quantized = quantized.permute(0, 3, 1, 2).contiguous()
        
        return loss, quantized

class VQVAE(nn.Module):
    """
    The complete VQ-VAE model, combining the Encoder, VectorQuantizer, and Decoder.
    """
    def __init__(self, in_channels, hidden_channels, out_channels, num_res_blocks, res_channels,
                num_embeddings, embedding_dim, commitment_cost):
        super(VQVAE, self).__init__()
        
        self.encoder = Encoder(in_channels, hidden_channels, num_res_blocks, res_channels)
        self.vq_layer = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)
        self.decoder = Decoder(embedding_dim, hidden_channels, out_channels)

    def forward(self, x):
        # 1. Encode the input image
        z = self.encoder(x)
        
        # 2. Quantize the latent representation
        vq_loss, quantized_z = self.vq_layer(z)
        
        # 3. Decode the quantized representation to reconstruct the image
        x_recon = self.decoder(quantized_z)
        
        return vq_loss, x_recon