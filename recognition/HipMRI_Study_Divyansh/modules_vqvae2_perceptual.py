import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import numpy as np

class ResidualBlock(nn.Module):
    """
    A standard residual block with two convolutional layers.
    It uses InstanceNorm for normalization.
    """
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.InstanceNorm2d(channels),
            nn.ReLU(True),
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.InstanceNorm2d(channels)
        )
    
    def forward(self, x):
        """
        Adds the input tensor to the output of the block (skip connection).
        """
        return x + self.block(x)
    
class VectorQuantizer(nn.Module):
    """
    The Vector Quantizer (VQ) layer.
    This layer takes a tensor of continuous features and maps it to a
    finite number of discrete embeddings.
    """
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        super(VectorQuantizer, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        
        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embedding.weight.data.uniform_(-1/self.num_embeddings, 1/self.num_embeddings)
    
    def forward(self, inputs):
        # Reshape input from [B, C, H, W] to [B*H*W, C]
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape
        flat_input = inputs.view(-1, self.embedding_dim)
        
        # Calculate distances to embedding vectors
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                     + torch.sum(self.embedding.weight**2, dim=1)
                     - 2 * torch.matmul(flat_input, self.embedding.weight.t()))
        
        # Find the closest embedding for each vector
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # Get the quantized vectors
        quantized = torch.matmul(encodings, self.embedding.weight).view(input_shape)
        
        # VQ Loss calculation
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss
        
        # Straight-through estimator
        quantized = inputs + (quantized - inputs).detach()
        
        # Perplexity (a measure of codebook usage)
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        # Reshape back to [B, C, H, W]
        quantized = quantized.permute(0, 3, 1, 2).contiguous()
        
        return loss, quantized, perplexity, encoding_indices

class BottomUpEncoder(nn.Module):
    """Bottom-up encoder for VQVAE-2 that extracts features at multiple scales."""
    def __init__(self, in_channels=1, base_channels=64, num_levels=3, num_residual_blocks=2):
        super(BottomUpEncoder, self).__init__()
        self.num_levels = num_levels
        
        # Initial convolution
        self.initial_conv = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 4, 2, 1),
            nn.InstanceNorm2d(base_channels),
            nn.ReLU(True)
        )
        
        # Bottom-up blocks
        self.blocks = nn.ModuleList()
        current_channels = base_channels
        
        for i in range(num_levels):
            level_blocks = []
            for _ in range(num_residual_blocks):
                level_blocks.append(ResidualBlock(current_channels))
            
            # Downsample for the next level (except for the last one)
            if i < num_levels - 1:
                level_blocks.extend([
                    nn.Conv2d(current_channels, current_channels * 2, 4, 2, 1),
                    nn.InstanceNorm2d(current_channels * 2),
                    nn.ReLU(True)
                ])
                current_channels *= 2
            
            self.blocks.append(nn.Sequential(*level_blocks))
    
    def forward(self, x):
        features = []
        x = self.initial_conv(x)
        
        for block in self.blocks:
            x = block(x)
            features.append(x)
        
        return features

class TopDownEncoder(nn.Module):
    """Top-down encoder that refines features using skip connections."""
    def __init__(self, channels_list, num_residual_blocks=2):
        super(TopDownEncoder, self).__init__()
        self.blocks = nn.ModuleList()
        
        # Process from the smallest feature map to the largest
        for i, channels in enumerate(channels_list[::-1]):
            block_layers = []
            
            # Upsample from the previous level (except for the very first top-level block)
            if i > 0:
                block_layers.extend([
                    nn.ConvTranspose2d(prev_channels, channels, 4, 2, 1),
                    nn.InstanceNorm2d(channels),
                    nn.ReLU(True)
                ])
            
            for _ in range(num_residual_blocks):
                block_layers.append(ResidualBlock(channels))
            
            self.blocks.append(nn.Sequential(*block_layers))
            prev_channels = channels
    
    def forward(self, bottom_up_features):
        # Reverse the features to start from the top (smallest spatial dimension)
        features = bottom_up_features[::-1]
        x = features[0]
        
        outputs = []
        for i, block in enumerate(self.blocks):
            x = block(x)
            outputs.append(x)
            
            # Add skip connection from the corresponding bottom-up feature map
            if i < len(features) - 1:
                x = x + features[i + 1]
        
        # Return features in original order (largest to smallest)
        return outputs[::-1]

print("modules_vqvae2_perceptual.py loaded.")