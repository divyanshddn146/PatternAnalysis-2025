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

from modules import EnhancedVQVAE2, calculate_ssim

class EnhancedVQVAE2Trainer:
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
        
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=200, eta_min=1e-6)
        self.plateau_scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', factor=0.5, patience=15)
        
        # Loss weights (tuned for medical imaging)
        self.lambda_rec = 1.0      # MSE reconstruction
        self.lambda_perceptual = 0.8  # Perceptual loss  
        self.lambda_ssim = 0.5     # SSIM loss
        
        self.train_losses = []
        self.val_losses = []
        self.train_ssim = []
        self.val_ssim = []
        self.perplexities = []
        self.loss_components_history = []
    
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        total_ssim = 0
        total_perplexity = 0
        num_batches = 0
        
        epoch_loss_components = {
    'mse': 0, 'perceptual': 0, 'ssim_loss': 0, 'ssim_value': 0, 'vq': 0, 'total': 0
}
        
        pbar = tqdm(self.train_loader, desc="Enhanced VQVAE-2 Training")
        for batch_idx, (data, _) in enumerate(pbar):
            data = data.to(self.device)
            self.optimizer.zero_grad()
            
            # Forward pass
            reconstructions, vq_loss, perplexity = self.model(data)
            
            # Enhanced loss calculation
            total_loss_batch, loss_components = self.model.calculate_enhanced_loss(
                data, reconstructions, vq_loss,
                self.lambda_rec, self.lambda_perceptual, self.lambda_ssim
            )
            
            # Backward pass
            total_loss_batch.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            self.optimizer.step()
            
            # Calculate SSIM for monitoring
            ssim = calculate_ssim(reconstructions, data)
            
            total_loss += total_loss_batch.item()
            total_ssim += ssim.item()
            total_perplexity += perplexity.item()
            num_batches += 1
            
            # Accumulate loss components
            for key, value in loss_components.items():
                if key in epoch_loss_components:
                    epoch_loss_components[key] += value
            
            pbar.set_postfix({
                'Total Loss': f'{total_loss_batch.item():.4f}',
                'SSIM': f'{ssim.item():.4f}',
                'MSE': f'{loss_components["mse"]:.4f}',
                'Perceptual': f'{loss_components["perceptual"]:.4f}'
            })
        
        # Average loss components
        for key in epoch_loss_components:
            epoch_loss_components[key] /= num_batches
        
        self.loss_components_history.append(epoch_loss_components)
        self.scheduler.step()
        
        return total_loss / num_batches, total_ssim / num_batches, total_perplexity / num_batches
    
    def validate_epoch(self):
        self.model.eval()
        total_loss = 0
        total_ssim = 0
        num_batches = 0
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc="Enhanced VQVAE-2 Validation")
            for data, _ in pbar:
                data = data.to(self.device)
                
                reconstructions, vq_loss, _ = self.model(data)
                
                # Calculate enhanced loss for validation
                total_loss_val, loss_components = self.model.calculate_enhanced_loss(
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
        patience = 35
        patience_counter = 0
        start_epoch = 0
        
        # Resume from checkpoint if provided
        if resume_checkpoint and os.path.exists(resume_checkpoint):
            checkpoint = torch.load(resume_checkpoint)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_ssim = checkpoint.get('best_ssim', checkpoint.get('val_ssim', 0.0))
            
            # FIX: Ensure loaded metrics are lists, not single values
            self.train_losses = self._ensure_list(checkpoint.get('train_losses', []))
            self.val_losses = self._ensure_list(checkpoint.get('val_losses', []))
            self.train_ssim = self._ensure_list(checkpoint.get('train_ssim', []))
            self.val_ssim = self._ensure_list(checkpoint.get('val_ssim', []))
            self.perplexities = self._ensure_list(checkpoint.get('perplexities', []))
            self.loss_components_history = self._ensure_list(checkpoint.get('loss_components_history', []))
            
            print(f"✅ Resumed training from epoch {start_epoch}")
            print(f"   Previous best SSIM: {best_ssim:.4f}")
            print(f"   Loaded history: {len(self.train_losses)} epochs")
        
        print("🚀 Starting Enhanced VQVAE-2 Training for 85%+ SSIM Target")
        print(f"Loss weights - MSE: {self.lambda_rec}, Perceptual: {self.lambda_perceptual}, SSIM: {self.lambda_ssim}")
        
        for epoch in range(start_epoch, num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            start_time = time.time()
            
            # Training
            train_loss, train_ssim, perplexity = self.train_epoch()
            
            # Validation
            val_loss, val_ssim = self.validate_epoch()
            
            # Learning rate scheduling
            self.plateau_scheduler.step(val_ssim)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            epoch_time = time.time() - start_time
            
            # Store metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_ssim.append(train_ssim)
            self.val_ssim.append(val_ssim)
            self.perplexities.append(perplexity)
            
            # Print loss components
            current_components = self.loss_components_history[-1] if self.loss_components_history else {
                'mse': 0, 'perceptual': 0, 'vq': 0
            }
            print(f"Train Loss: {train_loss:.4f}, Train SSIM: {train_ssim:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val SSIM: {val_ssim:.4f}")
            print(f"Loss Components - MSE: {current_components['mse']:.4f}, "
                  f"Perceptual: {current_components['perceptual']:.4f}, "
                  f"VQ: {current_components['vq']:.4f}")
            print(f"Perplexity: {perplexity:.2f}, LR: {current_lr:.2e}, Time: {epoch_time:.2f}s")
            
            # Save best model checkpoint
            if val_ssim > best_ssim:
                best_ssim = val_ssim
                patience_counter = 0
                
                checkpoint_data = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_ssim': val_ssim,
                    'train_ssim': train_ssim,
                    'best_ssim': best_ssim,
                    'config': config,
                    'loss_components': current_components,
                    # Save training history for resuming
                    'train_losses': self.train_losses,
                    'val_losses': self.val_losses,
                    'train_ssim': self.train_ssim,
                    'val_ssim': self.val_ssim,
                    'perplexities': self.perplexities,
                    'loss_components_history': self.loss_components_history
                }
                
                torch.save(checkpoint_data, os.path.join(save_dir, 'best_model.pth'))
                print(f"✅ Saved best model with SSIM: {val_ssim:.4f} (epoch {epoch+1})")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"🛑 Early stopping after {patience} epochs without improvement")
                    break
        
        # Save final checkpoint with complete training history
        final_checkpoint = {
            'epoch': start_epoch + len(self.train_losses) - 1,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_ssim': self.val_ssim[-1] if self.val_ssim else 0.0,
            'train_ssim': self.train_ssim[-1] if self.train_ssim else 0.0,
            'best_ssim': best_ssim,
            'config': config,
            # Complete training history
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_ssim': self.train_ssim,
            'val_ssim': self.val_ssim,
            'perplexities': self.perplexities,
            'loss_components_history': self.loss_components_history
        }
        torch.save(final_checkpoint, os.path.join(save_dir, 'final_model_complete_history.pth'))
        print(f"💾 Saved final model with complete {len(self.train_losses)} epochs of training history")
        
        # Plot enhanced training analysis
        self.plot_enhanced_analysis(save_dir)
        
        return best_ssim
    
    def _ensure_list(self, data):
        """Ensure data is a list, convert single values to list"""
        if isinstance(data, (int, float)):
            return [data]
        elif isinstance(data, list):
            return data
        else:
            return []
    
    def plot_enhanced_analysis(self, save_dir):
        """Enhanced plotting with loss component analysis"""
        if not self.train_losses:  # No training data
            return
            
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Loss curves
        axes[0, 0].plot(self.train_losses, label='Train Loss', linewidth=2)
        axes[0, 0].plot(self.val_losses, label='Val Loss', linewidth=2)
        axes[0, 0].set_title('Enhanced Training Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # SSIM curves
        axes[0, 1].plot(self.train_ssim, label='Train SSIM', linewidth=2)
        axes[0, 1].plot(self.val_ssim, label='Val SSIM', linewidth=2)
        axes[0, 1].axhline(y=0.85, color='g', linestyle='--', label='Target SSIM (0.85)', alpha=0.7)
        axes[0, 1].axhline(y=0.9, color='purple', linestyle='--', label='Goal SSIM (0.9)', alpha=0.7)
        axes[0, 1].set_title('Structural Similarity Index (SSIM)')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('SSIM')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Loss components
        if self.loss_components_history:
            mse_losses = [comp['mse'] for comp in self.loss_components_history]
            perceptual_losses = [comp['perceptual'] for comp in self.loss_components_history]
            vq_losses = [comp['vq'] for comp in self.loss_components_history]
            
            axes[0, 2].plot(mse_losses, label='MSE Loss', linewidth=2)
            axes[0, 2].plot(perceptual_losses, label='Perceptual Loss', linewidth=2)
            axes[0, 2].plot(vq_losses, label='VQ Loss', linewidth=2)
            axes[0, 2].set_title('Loss Components')
            axes[0, 2].set_xlabel('Epoch')
            axes[0, 2].set_ylabel('Loss')
            axes[0, 2].legend()
            axes[0, 2].grid(True, alpha=0.3)
        
        # Perplexity
        axes[1, 0].plot(self.perplexities, 'g-', linewidth=2)
        axes[1, 0].set_title('Codebook Perplexity')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Perplexity')
        axes[1, 0].grid(True, alpha=0.3)
        
        # SSIM vs Loss
        axes[1, 1].scatter(self.val_losses, self.val_ssim, c=range(len(self.val_losses)), 
                          cmap='viridis', alpha=0.7, s=30)
        axes[1, 1].set_xlabel('Validation Loss')
        axes[1, 1].set_ylabel('Validation SSIM')
        axes[1, 1].set_title('Loss vs SSIM Correlation')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Loss component ratio
        if self.loss_components_history:
            mse_ratio = [comp['mse'] / comp['total'] for comp in self.loss_components_history]
            perceptual_ratio = [comp['perceptual'] / comp['total'] for comp in self.loss_components_history]
            
            axes[1, 2].plot(mse_ratio, label='MSE Ratio', linewidth=2)
            axes[1, 2].plot(perceptual_ratio, label='Perceptual Ratio', linewidth=2)
            axes[1, 2].set_title('Loss Component Ratios')
            axes[1, 2].set_xlabel('Epoch')
            axes[1, 2].set_ylabel('Ratio')
            axes[1, 2].legend()
            axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'enhanced_training_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()

def main():
    parser = argparse.ArgumentParser(description='Enhanced VQVAE-2 Training')
    parser.add_argument('--data_dir', type=str, default='HipMRI_Study_open')
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1.5e-4)
    parser.add_argument('--image_size', type=int, default=128)
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    # ENHANCED CONFIG FOR 85%+ SSIM
    config = {
        'data_dir': args.data_dir,
        'batch_size': args.batch_size,
        'image_size': (args.image_size, args.image_size),
        'num_epochs': args.epochs,
        'learning_rate': args.lr,
        # Enhanced VQVAE-2 configuration
        'base_channels': 64,
        'num_levels': 3,
        'embedding_dims': [256, 128, 64],
        'num_embeddings_list': [512, 512, 512],
        'commitment_cost': 0.1,
        'num_residual_blocks': 2
    }
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    from dataset import create_data_loaders_keras
    train_loader, val_loader, test_loader, in_channels = create_data_loaders_keras(
        config['data_dir'],
        batch_size=config['batch_size'],
        target_size=config['image_size']
    )
    
    # Create Enhanced VQVAE-2 model
    model = EnhancedVQVAE2(
        in_channels=in_channels,
        base_channels=config['base_channels'],
        num_levels=config['num_levels'],
        embedding_dims=config['embedding_dims'],
        num_embeddings_list=config['num_embeddings_list'],
        commitment_cost=config['commitment_cost'],
        num_residual_blocks=config['num_residual_blocks'],
        device=device
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"🚀 Enhanced VQVAE-2 Model parameters: {total_params:,}")
    print("🎯 Using Perceptual + SSIM + MSE loss combination")
    
    # Train
    trainer = EnhancedVQVAE2Trainer(model, train_loader, val_loader, device, config['learning_rate'])
    best_ssim = trainer.train(config['num_epochs'], config, resume_checkpoint=args.resume)
    
    print(f"\n{'='*60}")
    print(f"🎉 ENHANCED VQVAE-2 TRAINING COMPLETED!")
    print(f"   Best SSIM: {best_ssim:.4f}")
    if best_ssim > 0.9:
        print("   🏆 EXCELLENT: SSIM > 0.9!")
    elif best_ssim > 0.85:
        print("   🎯 TARGET ACHIEVED: SSIM > 0.85!")
    elif best_ssim > 0.8:
        print("   ✅ VERY GOOD: SSIM > 0.8!")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()