import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import models
from utils import AdvancedVisualizations

class ResidualBlock(nn.Module):
    """Residual block with instance normalization"""
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.InstanceNorm2d(channels),
            nn.ReLU(True),
            nn.Dropout(0.1),  # ADD DROPOUT
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.InstanceNorm2d(channels)
        )
    
    def forward(self, x):
        return x + self.block(x)

class Encoder(nn.Module):
    """VQVAE Encoder"""
    def __init__(self, in_channels=1, hidden_dims=[64, 128, 256], num_residual_layers=2):
        super(Encoder, self).__init__()
        
        layers = []
        current_dim = in_channels
        
        # Initial convolution
        layers.extend([
            nn.Conv2d(current_dim, hidden_dims[0], 4, 2, 1),
            nn.InstanceNorm2d(hidden_dims[0]),
            nn.ReLU(True)
        ])
        current_dim = hidden_dims[0]
        
        # Downsampling blocks
        for hidden_dim in hidden_dims[1:]:
            layers.extend([
                nn.Conv2d(current_dim, hidden_dim, 4, 2, 1),
                nn.InstanceNorm2d(hidden_dim),
                nn.ReLU(True)
            ])
            current_dim = hidden_dim
        
        # Residual blocks
        for _ in range(num_residual_layers):
            layers.append(ResidualBlock(current_dim))
        
        self.network = nn.Sequential(*layers)
        self.final_conv = nn.Conv2d(current_dim, current_dim, 3, 1, 1)
    
    def forward(self, x):
        x = self.network(x)
        return self.final_conv(x)

class VectorQuantizer(nn.Module):
    """Vector Quantization layer"""
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        super(VectorQuantizer, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        
        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embedding.weight.data.uniform_(-1/self.num_embeddings, 1/self.num_embeddings)
    
    def forward(self, inputs):
        # Convert inputs from BCHW -> BHWC
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape
        
        # Flatten input
        flat_input = inputs.view(-1, self.embedding_dim)
        
        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self.embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self.embedding.weight.t()))
        
        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # Quantize and unflatten
        quantized = torch.matmul(encodings, self.embedding.weight).view(input_shape)
        
        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss
        
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        # Convert quantized from BHWC -> BCHW
        quantized = quantized.permute(0, 3, 1, 2).contiguous()
        
        return loss, quantized, perplexity, encoding_indices

class Decoder(nn.Module):
    """VQVAE Decoder"""
    def __init__(self, in_channels, hidden_dims=[256, 128, 64], out_channels=1, num_residual_layers=2):
        super(Decoder, self).__init__()
        
        layers = []
        current_dim = in_channels
        
        # Residual blocks
        for _ in range(num_residual_layers):
            layers.append(ResidualBlock(current_dim))
        
        # Upsampling blocks (reverse of encoder)
        for hidden_dim in hidden_dims[1:][::-1]:
            layers.extend([
                nn.ConvTranspose2d(current_dim, hidden_dim, 4, 2, 1),
                nn.InstanceNorm2d(hidden_dim),
                nn.ReLU(True)
            ])
            current_dim = hidden_dim
        
        # Final upsampling
        layers.extend([
            nn.ConvTranspose2d(current_dim, out_channels, 4, 2, 1),
            nn.Tanh()  # Normalize output to [-1, 1]
        ])
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

class VQVAE(nn.Module):
    """Complete VQVAE model"""
    def __init__(self, in_channels=1, hidden_dims=[64, 128, 256], num_embeddings=512, 
                 embedding_dim=256, commitment_cost=0.25, num_residual_layers=2):
        super(VQVAE, self).__init__()
        
        self.encoder = Encoder(in_channels, hidden_dims, num_residual_layers)
        self.pre_quantization_conv = nn.Conv2d(hidden_dims[-1], embedding_dim, 1)
        self.vector_quantizer = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)
        self.decoder = Decoder(embedding_dim, hidden_dims, in_channels, num_residual_layers)
    
    def forward(self, x):
        z = self.encoder(x)
        z = self.pre_quantization_conv(z)
        loss, quantized, perplexity, _ = self.vector_quantizer(z)
        x_recon = self.decoder(quantized)
        
        return x_recon, loss, perplexity
    
    def encode(self, x):
        z = self.encoder(x)
        z = self.pre_quantization_conv(z)
        _, quantized, _, encoding_indices = self.vector_quantizer(z)
        return quantized, encoding_indices
    
    def decode(self, z):
        return self.decoder(z)

def calculate_ssim(x, y, data_range=2.0):
    """Calculate Structural Similarity Index"""
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2
    
    mu_x = F.avg_pool2d(x, 3, 1, 1)
    mu_y = F.avg_pool2d(y, 3, 1, 1)
    
    sigma_x = F.avg_pool2d(x ** 2, 3, 1, 1) - mu_x ** 2
    sigma_y = F.avg_pool2d(y ** 2, 3, 1, 1) - mu_y ** 2
    sigma_xy = F.avg_pool2d(x * y, 3, 1, 1) - mu_x * mu_y
    
    ssim_numerator = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    ssim_denominator = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)
    
    return torch.mean(ssim_numerator / ssim_denominator)