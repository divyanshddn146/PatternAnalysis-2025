import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os
import time
from tqdm import tqdm
import argparse

from modules_vqvae2_perceptual import EnhancedVQVAE2, calculate_ssim

class EnhancedVQVAE2Trainer:
    """A trainer class to handle the training and validation of the VQ-VAE model."""
    def __init__(self, model, train_loader, val_loader, device, learning_rate=1.5e-4):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # Using AdamW optimizer for better weight decay handling
        self.optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=learning_rate, 
            weight_decay=1e-5,
            betas=(0.9, 0.98)
        )
        
        # Loss weights to balance the different components of the total loss
        self.lambda_rec = 1.0        # MSE reconstruction
        self.lambda_perceptual = 0.8 # Perceptual loss
        self.lambda_ssim = 0.5       # SSIM loss
        
        # Lists to store metrics for later plotting
        self.train_losses = []
        self.val_losses = []
        self.train_ssim = []
        self.val_ssim = []
        self.perplexities = []
        self.loss_components_history = []
    
    def train_epoch(self):
        """Runs a single training epoch."""
        self.model.train()
        total_loss = 0
        total_ssim = 0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc="Enhanced VQVAE-2 Training")
        for data, _ in pbar:
            data = data.to(self.device)
            self.optimizer.zero_grad()
            
            # Forward pass through the model
            reconstructions, vq_loss, perplexity = self.model(data)
            
            # Use the enhanced loss calculation method from the model
            total_loss_batch, loss_components = self.model.calculate_enhanced_loss(
                data, reconstructions, vq_loss,
                self.lambda_rec, self.lambda_perceptual, self.lambda_ssim
            )
            
            # Backward pass and optimization
            total_loss_batch.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5) # Gradient clipping
            self.optimizer.step()
            
            # Calculate SSIM for monitoring
            ssim = calculate_ssim(reconstructions, data)
            
            total_loss += total_loss_batch.item()
            total_ssim += ssim.item()
            num_batches += 1
            
            # Update the progress bar
            pbar.set_postfix({
                'Total Loss': f'{total_loss_batch.item():.4f}',
                'SSIM': f'{ssim.item():.4f}',
                'MSE': f'{loss_components["mse"]:.4f}',
                'Perceptual': f'{loss_components["perceptual"]:.4f}'
            })
        
        return total_loss / num_batches, total_ssim / num_batches

    def validate_epoch(self):
        """Runs a single validation epoch."""
        self.model.eval()
        total_loss = 0
        total_ssim = 0
        num_batches = 0
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc="Enhanced VQVAE-2 Validation")
            for data, _ in pbar:
                data = data.to(self.device)
                
                reconstructions, vq_loss, _ = self.model(data)
                
                # Calculate the same enhanced loss for validation
                total_loss_val, _ = self.model.calculate_enhanced_loss(
                    data, reconstructions, vq_loss,
                    self.lambda_rec, self.lambda_perceptual, self.lambda_ssim
                )
                
                ssim = calculate_ssim(reconstructions, data)
                
                total_loss += total_loss_val.item()
                total_ssim += ssim.item()
                num_batches += 1
                
                pbar.set_postfix({
                    'Val Loss': f'{total_loss_val.item():.4f}',
                    'Val SSIM': f'{ssim.item():.4f}'
                })
        
        return total_loss / num_batches, total_ssim / num_batches

def main():
    parser = argparse.ArgumentParser(description='Enhanced VQVAE-2 Training')
    parser.add_argument('--data_dir', type=str, default='HipMRI_Study_open', help='Directory for the dataset.')
    parser.add_argument('--batch_size', type=int, default=6, help='Input batch size for training.')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train for.')
    parser.add_argument('--lr', type=float, default=1.5e-4, help='Learning rate.')
    parser.add_argument('--image_size', type=int, default=128, help='The height and width of the input image.')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume training from.')
    
    args = parser.parse_args()
    
    print("Starting VQ-VAE 2 Training...")
    print(f"Arguments: {args}")
    
    # Configuration dictionary will be defined here.
    # Data loaders will be created here.
    # Model will be initialized here.
    # Trainer will be instantiated and training will commence here.

if __name__ == "__main__":
    main()