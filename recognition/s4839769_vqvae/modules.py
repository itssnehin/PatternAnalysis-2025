# modules.py

"""
Core components of the VQ-VAE and PixelCNN models.
- VQ-VAE: Encoder, Decoder, VectorQuantizer, Residual Blocks.
- PixelCNN: Masked Convolution and the main PixelCNN architecture.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- VQ-VAE Components (Largely Unchanged) ---

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, res_channels, out_channels):
        """Initializes a Residual Block.

        Args:
            in_channels (int): Number of input channels.
            res_channels (int): Number of channels in the internal convolutional layer.
            out_channels (int): Number of output channels.
        """
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, res_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(res_channels, out_channels, kernel_size=1, stride=1, bias=False)
        )
    def forward(self, x):
        """Defines the forward pass for the Residual Block.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after adding the residual connection.
        """
        return x + self.block(x)

class Encoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_res_blocks, res_channels):
        """Initializes the Encoder network.

        Args:
            in_channels (int): Number of channels in the input image.
            hidden_channels (int): Number of channels in the main convolutional layers.
            num_res_blocks (int): The number of residual blocks to include in the stack.
            res_channels (int): Number of channels in the internal layers of the residual blocks.
        """
        super(Encoder, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels // 2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels // 2, hidden_channels, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, stride=1, padding=1),
            *[ResidualBlock(hidden_channels, res_channels, hidden_channels) for _ in range(num_res_blocks)],
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        """Defines the forward pass for the Encoder.

        Args:
            x (torch.Tensor): The input image tensor.

        Returns:
            torch.Tensor: The continuous latent representation 'z_e'.
        """
        return self.layers(x)

class Decoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_res_blocks, res_channels):
        """Initializes the Decoder network.

        Args:
            in_channels (int): Number of channels in the quantized latent input.
            hidden_channels (int): Number of channels in the main convolutional layers.
            out_channels (int): Number of channels in the reconstructed output image.
            num_res_blocks (int): The number of residual blocks to include in the stack.
            res_channels (int): Number of channels in the internal layers of the residual blocks.
        """
        super(Decoder, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, stride=1, padding=1),
            *[ResidualBlock(hidden_channels, res_channels, hidden_channels) for _ in range(num_res_blocks)],
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(hidden_channels, hidden_channels // 2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(hidden_channels // 2, out_channels, kernel_size=4, stride=2, padding=1)
        )
    def forward(self, x):
        """Defines the forward pass for the Decoder.

        Args:
            x (torch.Tensor): The quantized latent representation 'z_q'.

        Returns:
            torch.Tensor: The reconstructed image tensor.
        """
        return self.layers(x)

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        """Initializes the Vector Quantizer layer.

        Args:
            num_embeddings (int): The number of vectors in the codebook.
            embedding_dim (int): The dimensionality of each codebook vector.
            commitment_cost (float): The 'beta' weight for the commitment loss term.
        """
        super(VectorQuantizer, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embedding.weight.data.uniform_(-1./self.num_embeddings, 1./self.num_embeddings)

    def forward(self, inputs):
        """Performs the vector quantization step.

        Maps the continuous encoder output to the nearest discrete codebook vectors.

        Args:
            inputs (torch.Tensor): The continuous latent representation from the encoder.

        Returns:
            tuple: A tuple containing:
                - torch.Tensor: The total VQ loss (codebook loss + commitment loss).
                - torch.Tensor: The quantized latent representation.
        """
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
    def __init__(self, in_channels, hidden_channels, out_channels, num_res_blocks, res_channels,
                num_embeddings, embedding_dim, commitment_cost):
        """Initializes the complete VQ-VAE model.

        Args:
            in_channels (int): Number of channels in the input image.
            hidden_channels (int): Base number of channels for the encoder/decoder.
            out_channels (int): Number of channels in the output image.
            num_res_blocks (int): Number of residual blocks in the encoder/decoder.
            res_channels (int): Number of channels within the residual blocks.
            num_embeddings (int): The number of vectors in the codebook.
            embedding_dim (int): The dimensionality of each codebook vector.
            commitment_cost (float): The 'beta' weight for the commitment loss.
        """
        super(VQVAE, self).__init__()
        self.encoder = Encoder(in_channels, hidden_channels, num_res_blocks, res_channels)
        self.vq_layer = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)
        self.decoder = Decoder(embedding_dim, hidden_channels, out_channels, num_res_blocks, res_channels)

    def forward(self, x):
        """Initializes the complete VQ-VAE model.

        Args:
            in_channels (int): Number of channels in the input image.
            hidden_channels (int): Base number of channels for the encoder/decoder.
            out_channels (int): Number of channels in the output image.
            num_res_blocks (int): Number of residual blocks in the encoder/decoder.
            res_channels (int): Number of channels within the residual blocks.
            num_embeddings (int): The number of vectors in the codebook.
            embedding_dim (int): The dimensionality of each codebook vector.
            commitment_cost (float): The 'beta' weight for the commitment loss.
        """
        z = self.encoder(x)
        vq_loss, quantized_z = self.vq_layer(z)
        x_recon = self.decoder(quantized_z)
        return vq_loss, x_recon

    def get_code_indices(self, x):
        """Encodes an image and returns the discrete codebook indices.

        This helper function is used to generate targets for training the PixelCNN prior.

        Args:
            x (torch.Tensor): The input image tensor.

        Returns:
            torch.Tensor: A tensor of shape [B, H, W] containing the codebook index for
                          each position in the latent map.
        """
        """Encodes an image and returns the discrete codebook indices."""
        z = self.encoder(x)
        z_permuted = z.permute(0, 2, 3, 1).contiguous()
        flat_z = z_permuted.view(-1, self.vq_layer.embedding_dim)
        
        distances = (torch.sum(flat_z**2, dim=1, keepdim=True) 
                    + torch.sum(self.vq_layer.embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_z, self.vq_layer.embedding.weight.t()))
            
        indices = torch.argmin(distances, dim=1)
        return indices.view(z.shape[0], z.shape[2], z.shape[3]) # Reshape to [B, H, W]

# --- NEW PIXELCNN COMPONENTS ---

class MaskedConv2d(nn.Conv2d):
    """
    A Convolutional layer with a mask to respect the autoregressive property.
    For the first layer (type 'A'), it masks the center pixel.
    For subsequent layers (type 'B'), it allows the center pixel to see itself.
    """
    def __init__(self, mask_type, *args, **kwargs):
        """Initializes a masked convolutional layer for autoregressive models.

        Args:
            mask_type (str): The type of mask to apply. Must be 'A' (for the first layer,
                             blocking the center pixel) or 'B' (for subsequent layers,
                             allowing the center pixel).
            *args: Variable length argument list passed to nn.Conv2d.
            **kwargs: Arbitrary keyword arguments passed to nn.Conv2d.
        """
        super(MaskedConv2d, self).__init__(*args, **kwargs)
        assert mask_type in {'A', 'B'}
        self.register_buffer('mask', self.weight.data.clone())
        
        h, w = self.kernel_size
        self.mask.fill_(1)
        self.mask[:, :, h // 2, w // 2 + (mask_type == 'B'):] = 0
        self.mask[:, :, h // 2 + 1:] = 0

    def forward(self, x):
        """Performs a forward pass with the mask applied to the weights.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output of the masked convolution.
        """
        self.weight.data *= self.mask
        return super(MaskedConv2d, self).forward(x)

class PixelCNN(nn.Module):
    """
    The PixelCNN model that learns the prior distribution over the discrete latent space.
    """
    def __init__(self, num_embeddings, hidden_dim=256, num_layers=7):
        """Initializes the PixelCNN prior model.

        Args:
            num_embeddings (int): The number of possible discrete codes (size of the VQ-VAE codebook).
            hidden_dim (int, optional): The number of channels in the convolutional layers. Defaults to 256.
            num_layers (int, optional): The total number of masked convolutional layers. Defaults to 7.
        """
        super(PixelCNN, self).__init__()
        
        # The PixelCNN needs an embedding layer for the input indices
        self.embedding = nn.Embedding(num_embeddings, hidden_dim)
        
        layers = [MaskedConv2d('A', hidden_dim, hidden_dim, kernel_size=7, padding=3)]
        for _ in range(num_layers - 1):
            layers.append(nn.ReLU(True))
            layers.append(MaskedConv2d('B', hidden_dim, hidden_dim, kernel_size=7, padding=3))
        
        # A final conv layer to produce the output logits
        layers.extend([
            nn.ReLU(True),
            nn.Conv2d(hidden_dim, num_embeddings, kernel_size=1)
        ])
        
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """Defines the forward pass for the PixelCNN.

        Args:
            x (torch.Tensor): The input tensor of discrete code indices, shape [B, H, W].

        Returns:
            torch.Tensor: The output logits of shape [B, num_embeddings, H, W], representing
                          the predicted probability distribution for each position.
        """
        # x is the input tensor of code indices, shape [B, H, W]
        x = self.embedding(x.long()).permute(0, 3, 1, 2) # To [B, C, H, W]
        return self.net(x)