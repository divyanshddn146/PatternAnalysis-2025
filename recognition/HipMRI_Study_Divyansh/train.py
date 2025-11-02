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
        """
        Initialize the trainer with model, data loaders, and training configuration.
        This sets up everything we need: optimizers, schedulers, and loss tracking lists.
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # AdamW optimizer with weight decay to prevent overfitting
        # The beta values (0.9, 0.98) are tuned for stable training of VQ-VAE models
        self.optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=learning_rate, 
            weight_decay=1e-5,
            betas=(0.9, 0.98)
        )
        
        # Two-stage learning rate scheduling for better convergence
        # Cosine annealing gradually reduces LR over training
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=200, eta_min=1e-6)
        # Plateau scheduler reduces LR when validation SSIM stops improving
        self.plateau_scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', factor=0.5, patience=15)
        
        # Loss weights - these are carefully tuned for medical imaging quality
        # MSE ensures basic pixel accuracy
        self.lambda_rec = 1.0      # MSE reconstruction
        # Perceptual loss (via VGG16) makes images look natural and sharp
        self.lambda_perceptual = 0.8  # Perceptual loss  
        # SSIM preserves structural information critical for medical diagnosis
        self.lambda_ssim = 0.5     # SSIM loss
        
        # Lists to track training progress over epochs
        self.train_losses = []
        self.val_losses = []
        self.train_ssim = []
        self.val_ssim = []
        self.perplexities = []
        self.loss_components_history = []
    
    def train_epoch(self):
        """
        Train the model for one complete epoch through the training data.
        Returns average loss, SSIM, and perplexity for this epoch.
        """
        self.model.train()
        total_loss = 0
        total_ssim = 0
        total_perplexity = 0
        num_batches = 0
        
        # Dictionary to accumulate all loss components for analysis
        epoch_loss_components = {
            'mse': 0, 'perceptual': 0, 'ssim_loss': 0, 'ssim_value': 0, 'vq': 0, 'total': 0
        }
        
        pbar = tqdm(self.train_loader, desc="Enhanced VQVAE-2 Training")
        for batch_idx, (data, _) in enumerate(pbar):
            data = data.to(self.device)
            self.optimizer.zero_grad()
            
            # Forward pass through the model
            # Returns: reconstructed images, VQ loss (codebook training), and perplexity (codebook usage)
            reconstructions, vq_loss, perplexity = self.model(data)
            
            # Calculate our enhanced multi-component loss
            # This combines MSE, perceptual loss (VGG16), SSIM, and VQ loss
            total_loss_batch, loss_components = self.model.calculate_enhanced_loss(
                data, reconstructions, vq_loss,
                self.lambda_rec, self.lambda_perceptual, self.lambda_ssim
            )
            
            # Backward pass and optimization
            total_loss_batch.backward()
            # Gradient clipping prevents exploding gradients in deep networks
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            self.optimizer.step()
            
            # Calculate SSIM separately for monitoring (not used in backprop)
            ssim = calculate_ssim(reconstructions, data)
            
            # Accumulate metrics for epoch averages
            total_loss += total_loss_batch.item()
            total_ssim += ssim.item()
            total_perplexity += perplexity.item()
            num_batches += 1
            
            # Accumulate loss components for detailed analysis
            for key, value in loss_components.items():
                if key in epoch_loss_components:
                    epoch_loss_components[key] += value
            
            # Update progress bar with current batch metrics
            pbar.set_postfix({
                'Total Loss': f'{total_loss_batch.item():.4f}',
                'SSIM': f'{ssim.item():.4f}',
                'MSE': f'{loss_components["mse"]:.4f}',
                'Perceptual': f'{loss_components["perceptual"]:.4f}'
            })
        
        # Average all loss components across batches
        for key in epoch_loss_components:
            epoch_loss_components[key] /= num_batches
        
        # Store this epoch's loss breakdown for plotting later
        self.loss_components_history.append(epoch_loss_components)
        
        # Step the cosine annealing scheduler
        self.scheduler.step()
        
        return total_loss / num_batches, total_ssim / num_batches, total_perplexity / num_batches
    
    def validate_epoch(self):
        """
        Evaluate the model on validation data without updating weights.
        This gives us an unbiased estimate of how well the model generalizes.
        """
        self.model.eval()
        total_loss = 0
        total_ssim = 0
        num_batches = 0
        
        # No gradient computation needed for validation - saves memory
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc="Enhanced VQVAE-2 Validation")
            for data, _ in pbar:
                data = data.to(self.device)
                
                # Forward pass only
                reconstructions, vq_loss, _ = self.model(data)
                
                # Calculate the same enhanced loss for fair comparison
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
        """
        Main training loop. Trains for num_epochs, saving checkpoints and handling early stopping.
        Can resume from a previous checkpoint if provided.
        """
        os.makedirs(save_dir, exist_ok=True)
        
        best_ssim = 0.0
        patience = 35  # Early stopping: stop if no improvement for 35 epochs
        patience_counter = 0
        start_epoch = 0
        
        # Resume from checkpoint if we're continuing a previous training run
        if resume_checkpoint and os.path.exists(resume_checkpoint):
            checkpoint = torch.load(resume_checkpoint)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_ssim = checkpoint.get('best_ssim', checkpoint.get('val_ssim', 0.0))
            
            # Load training history - this ensures our plots show the complete training curve
            # The _ensure_list helper fixes any corrupted data
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
        
        # Main training loop - each iteration is one complete pass through the data
        for epoch in range(start_epoch, num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            start_time = time.time()
            
            # Train for one epoch
            train_loss, train_ssim, perplexity = self.train_epoch()
            
            # Validate to check generalization
            val_loss, val_ssim = self.validate_epoch()
            
            # Adjust learning rate based on validation performance
            self.plateau_scheduler.step(val_ssim)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            epoch_time = time.time() - start_time
            
            # Record all metrics for this epoch
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_ssim.append(train_ssim)
            self.val_ssim.append(val_ssim)
            self.perplexities.append(perplexity)
            
            # Print detailed metrics to console
            current_components = self.loss_components_history[-1] if self.loss_components_history else {
                'mse': 0, 'perceptual': 0, 'vq': 0
            }
            print(f"Train Loss: {train_loss:.4f}, Train SSIM: {train_ssim:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val SSIM: {val_ssim:.4f}")
            print(f"Loss Components - MSE: {current_components['mse']:.4f}, "
                  f"Perceptual: {current_components['perceptual']:.4f}, "
                  f"VQ: {current_components['vq']:.4f}")
            print(f"Perplexity: {perplexity:.2f}, LR: {current_lr:.2e}, Time: {epoch_time:.2f}s")
            
            # Save checkpoint if this is the best model so far
            if val_ssim > best_ssim:
                best_ssim = val_ssim
                patience_counter = 0  # Reset early stopping counter
                
                # Create a comprehensive checkpoint with everything needed to resume
                checkpoint_data = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_ssim': val_ssim,
                    'train_ssim': train_ssim,
                    'best_ssim': best_ssim,
                    'config': config,
                    'loss_components': current_components,
                    # Save complete training history so we can resume and continue plotting
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
                # No improvement - increment early stopping counter
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"🛑 Early stopping after {patience} epochs without improvement")
                    break
        
        # Save final checkpoint with complete training history for analysis
        final_checkpoint = {
            'epoch': start_epoch + len(self.train_losses) - 1,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_ssim': self.val_ssim[-1] if self.val_ssim else 0.0,
            'train_ssim': self.train_ssim[-1] if self.train_ssim else 0.0,
            'best_ssim': best_ssim,
            'config': config,
            # Complete training history for comprehensive analysis and plotting
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_ssim': self.train_ssim,
            'val_ssim': self.val_ssim,
            'perplexities': self.perplexities,
            'loss_components_history': self.loss_components_history
        }
        torch.save(final_checkpoint, os.path.join(save_dir, 'final_model_complete_history.pth'))
        print(f"💾 Saved final model with complete {len(self.train_losses)} epochs of training history")
        
        # Generate comprehensive training analysis plots
        self.plot_enhanced_analysis(save_dir)
        
        return best_ssim
    
    def _ensure_list(self, data):
        """
        Helper function to ensure loaded checkpoint data is in list format.
        Sometimes checkpoints get corrupted or saved as single values - this fixes that.
        """
        if isinstance(data, (int, float)):
            return [data]
        elif isinstance(data, list):
            return data
        else:
            return []
    
    def plot_enhanced_analysis(self, save_dir):
        """
        Create a comprehensive 6-panel visualization of training progress.
        Shows loss curves, SSIM progress, loss components, perplexity, and more.
        """
        if not self.train_losses:  # No training data to plot
            return
            
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Panel 1: Main loss curves - shows training and validation loss over time
        axes[0, 0].plot(self.train_losses, label='Train Loss', linewidth=2)
        axes[0, 0].plot(self.val_losses, label='Val Loss', linewidth=2)
        axes[0, 0].set_title('Enhanced Training Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Panel 2: SSIM curves - our main quality metric
        # The horizontal lines show our target thresholds
        axes[0, 1].plot(self.train_ssim, label='Train SSIM', linewidth=2)
        axes[0, 1].plot(self.val_ssim, label='Val SSIM', linewidth=2)
        axes[0, 1].axhline(y=0.85, color='g', linestyle='--', label='Target SSIM (0.85)', alpha=0.7)
        axes[0, 1].axhline(y=0.9, color='purple', linestyle='--', label='Goal SSIM (0.9)', alpha=0.7)
        axes[0, 1].set_title('Structural Similarity Index (SSIM)')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('SSIM')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Panel 3: Individual loss components - helps diagnose training issues
        # VQ loss should dominate early, then stabilize
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
        
        # Panel 4: Codebook perplexity - measures how many codebook entries are being used
        # Should stabilize at a reasonable value (not collapse to 1, not stay at max)
        axes[1, 0].plot(self.perplexities, 'g-', linewidth=2)
        axes[1, 0].set_title('Codebook Perplexity')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Perplexity')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Panel 5: Correlation plot - shows if loss reduction actually improves SSIM
        # Color gradient shows progression over time
        axes[1, 1].scatter(self.val_losses, self.val_ssim, c=range(len(self.val_losses)), 
                          cmap='viridis', alpha=0.7, s=30)
        axes[1, 1].set_xlabel('Validation Loss')
        axes[1, 1].set_ylabel('Validation SSIM')
        axes[1, 1].set_title('Loss vs SSIM Correlation')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Panel 6: Loss component ratios - shows what the model is optimizing for
        # High perceptual ratio means focus on visual quality over pixel-perfect accuracy
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
    """
    Main entry point for training. Handles command-line arguments and orchestrates the training process.
    """
    parser = argparse.ArgumentParser(description='Enhanced VQVAE-2 Training')
    parser.add_argument('--data_dir', type=str, default='HipMRI_Study_open')
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1.5e-4)
    parser.add_argument('--image_size', type=int, default=128)
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    # Model configuration - these hyperparameters are tuned for 128x128 medical images
    # num_levels=3 gives us a hierarchical representation at 3 different scales
    config = {
        'data_dir': args.data_dir,
        'batch_size': args.batch_size,
        'image_size': (args.image_size, args.image_size),
        'num_epochs': args.epochs,
        'learning_rate': args.lr,
        # Enhanced VQVAE-2 architecture parameters
        'base_channels': 64,  # Starting channel count - doubles at each downsampling
        'num_levels': 3,  # Number of hierarchical levels
        'embedding_dims': [256, 128, 64],  # Codebook dimensions at each level
        'num_embeddings_list': [512, 512, 512],  # Codebook size at each level
        'commitment_cost': 0.1,  # Weight for commitment loss
        'num_residual_blocks': 2  # Residual blocks per level
    }
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load the MRI dataset
    from dataset import create_data_loaders_keras
    train_loader, val_loader, test_loader, in_channels = create_data_loaders_keras(
        config['data_dir'],
        batch_size=config['batch_size'],
        target_size=config['image_size']
    )
    
    # Create the Enhanced VQVAE-2 model with our configuration
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
    
    # Print model size for reference
    total_params = sum(p.numel() for p in model.parameters())
    print(f"🚀 Enhanced VQVAE-2 Model parameters: {total_params:,}")
    print("🎯 Using Perceptual + SSIM + MSE loss combination")
    
    # Initialize trainer and start training
    trainer = EnhancedVQVAE2Trainer(model, train_loader, val_loader, device, config['learning_rate'])
    best_ssim = trainer.train(config['num_epochs'], config, resume_checkpoint=args.resume)
    
    # Print final results with achievement levels
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