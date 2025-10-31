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
        
        self.optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=learning_rate, 
            weight_decay=1e-5,
            betas=(0.9, 0.98)
        )
        
        self.lambda_rec = 1.0
        self.lambda_perceptual = 0.8
        self.lambda_ssim = 0.5
        
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
        
        # NOTE: Slightly modified to also return perplexity for tracking
        total_perplexity = 0
        
        pbar = tqdm(self.train_loader, desc="Enhanced VQVAE-2 Training")
        for data, _ in pbar:
            data = data.to(self.device)
            self.optimizer.zero_grad()
            
            reconstructions, vq_loss, perplexity = self.model(data)
            
            total_loss_batch, loss_components = self.model.calculate_enhanced_loss(
                data, reconstructions, vq_loss,
                self.lambda_rec, self.lambda_perceptual, self.lambda_ssim
            )
            
            total_loss_batch.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            self.optimizer.step()
            
            ssim = calculate_ssim(reconstructions, data)
            
            total_loss += total_loss_batch.item()
            total_ssim += ssim.item()
            total_perplexity += perplexity.item() # Track perplexity
            num_batches += 1
            
            pbar.set_postfix({
                'Total Loss': f'{total_loss_batch.item():.4f}',
                'SSIM': f'{ssim.item():.4f}',
                'MSE': f'{loss_components["mse"]:.4f}',
                'Perceptual': f'{loss_components["perceptual"]:.4f}'
            })
        
        return total_loss / num_batches, total_ssim / num_batches, total_perplexity / num_batches

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
    
    def train(self, num_epochs, config, save_dir='enhanced_vqvae2_checkpoints', resume_checkpoint=None):
        os.makedirs(save_dir, exist_ok=True)
        
        best_ssim = 0.0
        start_epoch = 0
        
        # Resume from checkpoint if provided
        if resume_checkpoint and os.path.exists(resume_checkpoint):
            checkpoint = torch.load(resume_checkpoint)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_ssim = checkpoint.get('best_ssim', 0.0)
            
            # Load training history
            self.train_losses = self._ensure_list(checkpoint.get('train_losses', []))
            self.val_losses = self._ensure_list(checkpoint.get('val_losses', []))
            self.train_ssim = self._ensure_list(checkpoint.get('train_ssim', []))
            self.val_ssim = self._ensure_list(checkpoint.get('val_ssim', []))
            self.perplexities = self._ensure_list(checkpoint.get('perplexities', []))
            
            print(f"✅ Resumed training from epoch {start_epoch}")
            print(f"   Previous best SSIM: {best_ssim:.4f}")

        print("🚀 Starting Enhanced VQVAE-2 Training...")
        
        for epoch in range(start_epoch, num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            
            train_loss, train_ssim, perplexity = self.train_epoch()
            val_loss, val_ssim = self.validate_epoch()
            
            # Store metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_ssim.append(train_ssim)
            self.val_ssim.append(val_ssim)
            self.perplexities.append(perplexity)
            
            print(f"Train Loss: {train_loss:.4f}, Train SSIM: {train_ssim:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val SSIM: {val_ssim:.4f}")
            print(f"Perplexity: {perplexity:.2f}")
            
            # Save best model checkpoint based on validation SSIM
            if val_ssim > best_ssim:
                best_ssim = val_ssim
                
                checkpoint_data = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_ssim': val_ssim,
                    'best_ssim': best_ssim,
                    'config': config,
                    # Save history for resuming
                    'train_losses': self.train_losses,
                    'val_losses': self.val_losses,
                    'train_ssim': self.train_ssim,
                    'val_ssim': self.val_ssim,
                    'perplexities': self.perplexities,
                }
                
                torch.save(checkpoint_data, os.path.join(save_dir, 'best_model.pth'))
                print(f"✅ Saved best model with SSIM: {val_ssim:.4f} (epoch {epoch+1})")
                
        return best_ssim

    def _ensure_list(self, data):
        """Helper to ensure loaded metric data is a list."""
        if isinstance(data, (int, float)):
            return [data]
        return data if isinstance(data, list) else []

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