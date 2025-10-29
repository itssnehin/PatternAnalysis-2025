
"""
Core components of the Vector-Quantized Variational Autoencoder (VQ-VAE) model.
Contains the Encoder, Decoder, VectorQuantizer, and the main VQVAE module.
This version includes Residual Blocks for a more powerful architecture.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """
    A simple Residual Block with two convolutional layers.
    """
    def __init__(self, in_channels, res_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, res_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(res_channels, out_channels, kernel_size=1, stride=1, bias=False)
        )

    def forward(self, x):
        return x + self.block(x)

class Encoder(nn.Module):
    """
    The Encoder network with Residual Blocks to compress the input image.
    """
    def __init__(self, in_channels, hidden_channels, num_res_blocks, res_channels):
        super(Encoder, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels // 2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels // 2, hidden_channels, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, stride=1, padding=1),
            # Add a sequence of Residual Blocks
            *[ResidualBlock(hidden_channels, res_channels, hidden_channels) for _ in range(num_res_blocks)],
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.layers(x)

class Decoder(nn.Module):
    """
    The Decoder network with Residual Blocks to reconstruct the image.
    """
    def __init__(self, in_channels, hidden_channels, out_channels, num_res_blocks, res_channels):
        super(Decoder, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, stride=1, padding=1),
            # Add a sequence of Residual Blocks
            *[ResidualBlock(hidden_channels, res_channels, hidden_channels) for _ in range(num_res_blocks)],
            nn.ReLU(inplace=True),
            # Upsample twice to return to original size
            nn.ConvTranspose2d(hidden_channels, hidden_channels // 2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(hidden_channels // 2, out_channels, kernel_size=4, stride=2, padding=1)
        )

    def forward(self, x):
        return self.layers(x)

class VectorQuantizer(nn.Module):
    """
    The core Vector Quantizer layer. No changes needed here.
    """
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VectorQuantizer, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        
        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embedding.weight.data.uniform_(-1./self.num_embeddings, 1./self.num_embeddings)

    def forward(self, inputs):
        inputs_permuted = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs_permuted.shape
        flat_input = inputs_permuted.view(-1, self.embedding_dim)
        
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self.embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self.embedding.weight.t()))
            
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        quantized = torch.matmul(encodings, self.embedding.weight).view(input_shape)
        
        e_loss = F.mse_loss(quantized.detach(), inputs_permuted)
        q_loss = F.mse_loss(quantized, inputs_permuted.detach())
        loss = q_loss + self.commitment_cost * e_loss
        
        quantized = inputs_permuted + (quantized - inputs_permuted).detach()
        quantized = quantized.permute(0, 3, 1, 2).contiguous()
        
        return loss, quantized

class VQVAE(nn.Module):
    """
    The complete VQ-VAE model. We update this to pass parameters to the new Decoder.
    """
    def __init__(self, in_channels, hidden_channels, out_channels, num_res_blocks, res_channels,
                num_embeddings, embedding_dim, commitment_cost):
        super(VQVAE, self).__init__()
        
        self.encoder = Encoder(in_channels, hidden_channels, num_res_blocks, res_channels)
        self.vq_layer = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)
        # --- UPDATED DECODER CALL ---
        self.decoder = Decoder(embedding_dim, hidden_channels, out_channels, num_res_blocks, res_channels)

    def forward(self, x):
        z = self.encoder(x)
        vq_loss, quantized_z = self.vq_layer(z)
        x_recon = self.decoder(quantized_z)
        return vq_loss, x_recon