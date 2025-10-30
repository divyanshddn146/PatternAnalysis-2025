import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import torch.nn.functional as F

class AdvancedVisualizations:
    def __init__(self, model, device):
        self.model = model
        self.device = device
    
    def plot_latent_space_2d(self, dataloader, num_samples=1000, save_path='latent_space_2d.png'):
        """Visualize 2D projection of latent space using t-SNE"""
        self.model.eval()
        
        all_codes = []
        all_labels = []
        
        with torch.no_grad():
            for batch_idx, (data, _) in enumerate(dataloader):
                if batch_idx * dataloader.batch_size >= num_samples:
                    break
                    
                data = data.to(self.device)
                quantized, encoding_indices = self.model.encode(data)
                
                # Flatten encoding indices
                codes = encoding_indices.view(-1).cpu().numpy()
                all_codes.extend(codes)
                all_labels.extend([batch_idx] * len(codes))
        
        # Convert to numpy arrays
        all_codes = np.array(all_codes)
        all_labels = np.array(all_labels)
        
        # Use t-SNE for 2D visualization
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        codes_2d = tsne.fit_transform(all_codes.reshape(-1, 1))
        
        plt.figure(figsize=(12, 10))
        scatter = plt.scatter(codes_2d[:, 0], codes_2d[:, 1], c=all_labels, 
                            cmap='viridis', alpha=0.6, s=10)
        plt.colorbar(scatter, label='Batch Index')
        plt.title('t-SNE Visualization of Discrete Latent Space')
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        plt.grid(True, alpha=0.3)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return codes_2d
    
    def plot_codebook_usage(self, dataloader, save_path='codebook_usage.png'):
        """Visualize codebook embedding usage statistics"""
        self.model.eval()
        
        codebook_size = self.model.vector_quantizer.num_embeddings
        usage_count = torch.zeros(codebook_size)
        
        with torch.no_grad():
            for data, _ in dataloader:
                data = data.to(self.device)
                _, encoding_indices = self.model.encode(data)
                
                # Count usage of each code
                unique, counts = torch.unique(encoding_indices, return_counts=True)
                usage_count[unique] += counts.cpu()
        
        # Plot usage distribution
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 2, 1)
        plt.bar(range(codebook_size), usage_count.numpy(), alpha=0.7)
        plt.xlabel('Codebook Index')
        plt.ylabel('Usage Count')
        plt.title('Codebook Usage Distribution')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        sorted_usage = torch.sort(usage_count, descending=True)[0]
        plt.plot(sorted_usage.numpy(), 'r-', linewidth=2)
        plt.xlabel('Rank')
        plt.ylabel('Usage Count')
        plt.title('Codebook Usage (Sorted)')
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        usage_rate = (usage_count > 0).float().mean()
        print(f"Codebook Usage Rate: {usage_rate.item()*100:.2f}%")
        print(f"Most used code: {torch.argmax(usage_count).item()} (count: {torch.max(usage_count).item()})")
        print(f"Least used code: {torch.argmin(usage_count).item()} (count: {torch.min(usage_count).item()})")
        
        return usage_count
    
    def plot_training_curves_comprehensive(self, trainer, save_path='training_curves_comprehensive.png'):
        """Comprehensive training analysis with multiple subplots"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Loss curves
        axes[0, 0].plot(trainer.train_losses, label='Train Loss', linewidth=2)
        axes[0, 0].plot(trainer.val_losses, label='Val Loss', linewidth=2)
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # SSIM curves
        axes[0, 1].plot(trainer.train_ssim, label='Train SSIM', linewidth=2)
        axes[0, 1].plot(trainer.val_ssim, label='Val SSIM', linewidth=2)
        axes[0, 1].axhline(y=0.6, color='r', linestyle='--', label='Target SSIM (0.6)')
        axes[0, 1].set_title('Structural Similarity Index (SSIM)')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('SSIM')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Perplexity
        axes[0, 2].plot(trainer.perplexities, 'g-', linewidth=2)
        axes[0, 2].set_title('Codebook Perplexity')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('Perplexity')
        axes[0, 2].grid(True, alpha=0.3)
        
        # Loss derivatives (smoothed)
        train_loss_deriv = np.gradient(trainer.train_losses)
        val_loss_deriv = np.gradient(trainer.val_losses)
        axes[1, 0].plot(train_loss_deriv, label='Train Loss Derivative', alpha=0.7)
        axes[1, 0].plot(val_loss_deriv, label='Val Loss Derivative', alpha=0.7)
        axes[1, 0].axhline(y=0, color='k', linestyle='-', alpha=0.3)
        axes[1, 0].set_title('Loss Derivatives (Convergence Analysis)')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('d(Loss)/dEpoch')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # SSIM improvement rate
        ssim_improvement = np.gradient(trainer.val_ssim)
        axes[1, 1].plot(ssim_improvement, 'purple', linewidth=2)
        axes[1, 1].axhline(y=0, color='k', linestyle='-', alpha=0.3)
        axes[1, 1].set_title('SSIM Improvement Rate')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('d(SSIM)/dEpoch')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Loss vs SSIM correlation
        axes[1, 2].scatter(trainer.val_losses, trainer.val_ssim, 
                          c=range(len(trainer.val_losses)), cmap='viridis', alpha=0.6)
        axes[1, 2].set_xlabel('Validation Loss')
        axes[1, 2].set_ylabel('Validation SSIM')
        axes[1, 2].set_title('Loss vs SSIM Correlation')
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_reconstruction_quality_analysis(self, dataloader, num_samples=5, save_path='reconstruction_quality.png'):
        """Detailed analysis of reconstruction quality"""
        self.model.eval()
        
        data_iter = iter(dataloader)
        images, _ = next(data_iter)
        images = images[:num_samples].to(self.device)
        
        with torch.no_grad():
            reconstructions, _, _ = self.model(images)
        
        # Denormalize
        images = (images + 1) / 2
        reconstructions = (reconstructions + 1) / 2
        
        fig, axes = plt.subplots(4, num_samples, figsize=(20, 16))
        
        if num_samples == 1:
            axes = axes.reshape(4, 1)
        
        for i in range(num_samples):
            # Original
            axes[0, i].imshow(images[i].cpu().squeeze(), cmap='gray')
            axes[0, i].set_title(f'Original {i+1}')
            axes[0, i].axis('off')
            
            # Reconstruction
            axes[1, i].imshow(reconstructions[i].cpu().squeeze(), cmap='gray')
            ssim = calculate_ssim(reconstructions[i:i+1], images[i:i+1]).item()
            axes[1, i].set_title(f'Reconstruction\nSSIM: {ssim:.3f}')
            axes[1, i].axis('off')
            
            # Absolute difference
            diff = torch.abs(images[i] - reconstructions[i])
            im = axes[2, i].imshow(diff.cpu().squeeze(), cmap='hot')
            axes[2, i].set_title(f'Absolute Difference\nMax: {diff.max().item():.3f}')
            axes[2, i].axis('off')
            plt.colorbar(im, ax=axes[2, i], fraction=0.046)
            
            # Error histogram
            axes[3, i].hist(diff.cpu().flatten().numpy(), bins=50, alpha=0.7, edgecolor='black')
            axes[3, i].set_title(f'Error Distribution\nMean: {diff.mean().item():.4f}')
            axes[3, i].set_xlabel('Error Magnitude')
            axes[3, i].set_ylabel('Frequency')
            axes[3, i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

def calculate_ssim(x, y, data_range=1.0):
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