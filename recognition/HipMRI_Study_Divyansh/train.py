import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os
import time
from tqdm import tqdm
import argparse

from modules import VQVAE, calculate_ssim
from dataset import create_data_loaders_keras, explore_h5_structure
from utils import AdvancedVisualizations

class VQVAETrainer:
    def __init__(self, model, train_loader, val_loader, device, learning_rate=1e-3):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.reconstruction_criterion = nn.MSELoss()
        
        self.train_losses = []
        self.val_losses = []
        self.train_ssim = []
        self.val_ssim = []
        self.perplexities = []
    
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        total_ssim = 0
        total_perplexity = 0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc="Training")
        for batch_idx, (data, _) in enumerate(pbar):
            data = data.to(self.device)
            self.optimizer.zero_grad()
            
            # Forward pass
            reconstructions, vq_loss, perplexity = self.model(data)
            
            # Calculate losses
            reconstruction_loss = self.reconstruction_criterion(reconstructions, data)
            loss = reconstruction_loss + vq_loss
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            # Calculate SSIM
            ssim = calculate_ssim(reconstructions, data)
            
            total_loss += loss.item()
            total_ssim += ssim.item()
            total_perplexity += perplexity.item()
            num_batches += 1
            
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'SSIM': f'{ssim.item():.4f}',
                'Perplexity': f'{perplexity.item():.2f}'
            })
        
        return total_loss / num_batches, total_ssim / num_batches, total_perplexity / num_batches
    
    def validate_epoch(self):
        self.model.eval()
        total_loss = 0
        total_ssim = 0
        num_batches = 0
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc="Validation")
            for data, _ in pbar:
                data = data.to(self.device)
                
                reconstructions, vq_loss, _ = self.model(data)
                reconstruction_loss = self.reconstruction_criterion(reconstructions, data)
                loss = reconstruction_loss + vq_loss
                
                ssim = calculate_ssim(reconstructions, data)
                
                total_loss += loss.item()
                total_ssim += ssim.item()
                num_batches += 1
                
                pbar.set_postfix({
                    'Val Loss': f'{loss.item():.4f}',
                    'Val SSIM': f'{ssim.item():.4f}'
                })
        
        return total_loss / num_batches, total_ssim / num_batches
    
    def train(self, num_epochs, config, save_dir='checkpoints'):
        os.makedirs(save_dir, exist_ok=True)
        
        best_ssim = 0.0
        patience = 30
        patience_counter = 0
        
        print("Starting training...")
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            start_time = time.time()
            
            # Training
            train_loss, train_ssim, perplexity = self.train_epoch()
            
            # Validation
            val_loss, val_ssim = self.validate_epoch()
            
            epoch_time = time.time() - start_time
            
            # Store metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_ssim.append(train_ssim)
            self.val_ssim.append(val_ssim)
            self.perplexities.append(perplexity)
            
            print(f"Train Loss: {train_loss:.4f}, Train SSIM: {train_ssim:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val SSIM: {val_ssim:.4f}")
            print(f"Perplexity: {perplexity:.2f}, Time: {epoch_time:.2f}s")
            
            # Early stopping based on SSIM
            if val_ssim > best_ssim:
                best_ssim = val_ssim
                patience_counter = 0
                
                # Save best model
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_ssim': val_ssim,
                    'train_ssim': train_ssim,
                    'loss': val_loss,
                    'config': config
                }, os.path.join(save_dir, 'best_model.pth'))
                print(f"✅ Saved best model with SSIM: {val_ssim:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"🛑 Early stopping after {patience} epochs without improvement")
                    break
            
            # Save checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_ssim': val_ssim,
                    'train_ssim': train_ssim,
                    'loss': val_loss
                }, os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pth'))
        
        # Plot training history
        self.plot_training_history(save_dir)
        
        # Generate advanced metrics
        self.plot_advanced_metrics(config, save_dir)
        
        return best_ssim
    
    def plot_advanced_metrics(self, config, save_dir):
        """Generate advanced training metrics"""
        visualizer = AdvancedVisualizations(self.model, self.device)
        
        # Comprehensive training curves
        visualizer.plot_training_curves_comprehensive(self, 
                                                    save_path=os.path.join(save_dir, 'training_curves_comprehensive.png'))
        
        # Create training summary
        final_val_ssim = self.val_ssim[-1] if self.val_ssim else 0
        best_val_ssim = max(self.val_ssim) if self.val_ssim else 0
        
        print(f"\n📈 Training Summary:")
        print(f"   Final Validation SSIM: {final_val_ssim:.4f}")
        print(f"   Best Validation SSIM: {best_val_ssim:.4f}")
        print(f"   Epochs trained: {len(self.train_losses)}")
        print(f"   Target achieved: {'✅' if best_val_ssim > 0.6 else '❌'}")
    
    def plot_training_history(self, save_dir):
        plt.figure(figsize=(15, 5))
        
        # Plot losses
        plt.subplot(1, 3, 1)
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.val_losses, label='Val Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # Plot SSIM
        plt.subplot(1, 3, 2)
        plt.plot(self.train_ssim, label='Train SSIM')
        plt.plot(self.val_ssim, label='Val SSIM')
        plt.axhline(y=0.6, color='r', linestyle='--', label='Target SSIM (0.6)')
        plt.title('Training and Validation SSIM')
        plt.xlabel('Epoch')
        plt.ylabel('SSIM')
        plt.legend()
        plt.grid(True)
        
        # Plot perplexity
        plt.subplot(1, 3, 3)
        plt.plot(self.perplexities, label='Perplexity')
        plt.title('Codebook Perplexity')
        plt.xlabel('Epoch')
        plt.ylabel('Perplexity')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'training_history.png'), dpi=300, bbox_inches='tight')
        plt.close()

def main():
    parser = argparse.ArgumentParser(description='VQVAE Training for Prostate MRI')
    parser.add_argument('--data_dir', type=str, default='HipMRI_Study_open', help='Root data directory')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--epochs', type=int, default=120, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--image_size', type=int, default=128, help='Image size')
    parser.add_argument('--multimodal', action='store_true', help='Use multimodal input (image + segmentation)')
    
    args = parser.parse_args()
    
    # Configuration
    # In main() function, change config to:
   # BALANCED MEDIUM MODEL (between small and large)
    config = {
    'data_dir': args.data_dir,
    'batch_size': 12,
    'image_size': (args.image_size, args.image_size),
    'num_epochs': args.epochs,
    'learning_rate': 5e-4,  # Lower LR for medium model
    'hidden_dims': [96, 192, 384],    # 50% increase from small
    'num_embeddings': 768,            # Moderate increase
    'embedding_dim': 384,             # Moderate increase
    'commitment_cost': 0.2,           # Slightly lower
    'num_residual_layers': 3,         # One extra layer
    'use_multimodal': args.multimodal
}
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Explore data structure first
    print("Exploring data structure...")
    explore_h5_structure(config['data_dir'])
    
    # Create data loaders
    try:
        train_loader, val_loader, test_loader, in_channels = create_data_loaders_keras(
            config['data_dir'],
            batch_size=config['batch_size'],
            target_size=config['image_size'],
            use_multimodal=config['use_multimodal']
        )
        
        config['in_channels'] = in_channels
        print(f"Input channels: {in_channels}")
        
    except Exception as e:
        print(f"Error loading data: {e}")
        print("Please check the data directory and file structure")
        return
    
    # Create model
    model = VQVAE(
        in_channels=config['in_channels'],
        hidden_dims=config['hidden_dims'],
        num_embeddings=config['num_embeddings'],
        embedding_dim=config['embedding_dim'],
        commitment_cost=config['commitment_cost'],
        num_residual_layers=config['num_residual_layers']
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create trainer
    trainer = VQVAETrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=config['learning_rate']
    )
    
    # Train model
    best_ssim = trainer.train(
        num_epochs=config['num_epochs'],
        config=config,
        save_dir='checkpoints'
    )
    
    print(f"\n{'='*50}")
    print(f"Training completed! Best validation SSIM: {best_ssim:.4f}")
    if best_ssim > 0.6:
        print("✅ Target SSIM > 0.6 achieved!")
    else:
        print("❌ Target SSIM > 0.6 not achieved. Consider tuning hyperparameters.")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()