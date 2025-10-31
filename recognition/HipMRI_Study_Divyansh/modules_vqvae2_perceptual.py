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

class Decoder(nn.Module):
    """Decoder that reconstructs the image from quantized features."""
    def __init__(self, embedding_dims, out_channels=1, num_residual_blocks=2):
        super(Decoder, self).__init__()
        
        # The decoder architecture starts with the highest-level (most compressed)
        # feature's embedding dimension.
        current_channels = embedding_dims[0]
        
        layers = []
        
        # Process the initial feature map
        layers.extend([
            ResidualBlock(current_channels),
            nn.Conv2d(current_channels, 128, 3, 1, 1),
            nn.InstanceNorm2d(128),
            nn.ReLU(True),
            ResidualBlock(128)
        ])
        current_channels = 128
        
        # Final upsampling block to restore original image size
        layers.extend([
            nn.ConvTranspose2d(current_channels, 64, 4, 2, 1), # Upsamples to 128x128
            nn.InstanceNorm2d(64),
            nn.ReLU(True),
            ResidualBlock(64),
            nn.Conv2d(64, out_channels, 3, 1, 1),
            nn.Tanh() # Tanh activation to scale output to [-1, 1]
        ])
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, quantized_features):
        # The VQVAE-2 decoder typically only uses the features
        # from the highest level of the hierarchy for reconstruction.
        x = quantized_features[0]
        return self.network(x)

class EnhancedVQVAE2(nn.Module):
    """Assembles the full VQ-VAE 2 model with hierarchical encoders and quantizers."""
    def __init__(self, in_channels=1, base_channels=64, num_levels=3, 
                 embedding_dims=[256, 128, 64], num_embeddings_list=[512, 512, 512],
                 commitment_cost=0.25, num_residual_blocks=2, device='cuda'):
        super(EnhancedVQVAE2, self).__init__()
        
        self.num_levels = num_levels
        
        # Encoders
        self.bottom_up_encoder = BottomUpEncoder(
            in_channels, base_channels, num_levels, num_residual_blocks
        )
        
        channel_sizes = [base_channels * (2 ** i) for i in range(num_levels)]
        self.top_down_encoder = TopDownEncoder(channel_sizes, num_residual_blocks)
        
        # Quantizers
        self.quantizer_conv_layers = nn.ModuleList()
        self.quantizers = nn.ModuleList()
        
        for i, (embed_dim, num_emb) in enumerate(zip(embedding_dims, num_embeddings_list)):
            self.quantizer_conv_layers.append(nn.Conv2d(channel_sizes[i], embed_dim, 1))
            self.quantizers.append(VectorQuantizer(num_emb, embed_dim, commitment_cost))
        
        # Decoder
        self.decoder = Decoder(embedding_dims, in_channels, num_residual_blocks)
    
    def forward(self, x):
        # Encode
        bottom_up_features = self.bottom_up_encoder(x)
        top_down_features = self.top_down_encoder(bottom_up_features)
        
        # Quantize at each level
        total_vq_loss = 0
        total_perplexity = 0
        quantized_features = []
        
        for feature, quant_conv, quantizer in zip(top_down_features, self.quantizer_conv_layers, self.quantizers):
            projected = quant_conv(feature)
            vq_loss, quantized, perplexity, _ = quantizer(projected)
            
            total_vq_loss += vq_loss
            total_perplexity += perplexity
            quantized_features.append(quantized)
        
        # Decode
        reconstructions = self.decoder(quantized_features)
        
        return reconstructions, total_vq_loss, total_perplexity / self.num_levels

print("modules_vqvae2_perceptual.py loaded.")