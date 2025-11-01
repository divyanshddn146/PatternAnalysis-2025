# advanced_visualizations.py
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import torch.nn.functional as F
from tqdm import tqdm

# We need this to calculate SSIM, as it's part of the plot
from modules_vqvae2_perceptual import calculate_ssim


class AdvancedVisualizations:
    def __init__(self, model, device):
        self.model = model
        self.device = device
    
    def plot_latent_space_2d(self, dataloader, level=0, num_samples=1000, save_path='latent_space_2d.png'):
        """
        Visualize 2D projection of the *embedding vectors* from a specific level.
        Uses t-SNE to show which codebook vectors are being used and how they cluster.
        """
        self.model.eval()
        
        all_codes = []
        all_labels = []
        
        print(f"Generating t-SNE plot for level {level}. This may take a while...")
        
        # --- THIS ENTIRE BLOCK HAS BEEN REPLACED ---
        with torch.no_grad():
            for batch_idx, (data, _) in enumerate(tqdm(dataloader, desc="Collecting Latent Vectors")):
                data = data.to(self.device)
                _, encoding_indices_list = self.model.encode(data)
                indices = encoding_indices_list[level].view(-1).cpu()
                
                codebook = self.model.quantizers[level].embedding.weight
                codes = codebook[indices].detach().cpu().numpy()
                
                all_codes.append(codes)
                all_labels.extend([batch_idx] * len(codes))
                
                # --- THIS IS THE FIX ---
                # Check the total number of *vectors* collected so far
                # We use .shape[0] on the concatenated array to get the true count
                if np.concatenate(all_codes, axis=0).shape[0] >= num_samples:
                    break
                # --- END OF FIX ---
        
        all_codes = np.concatenate(all_codes, axis=0)
        all_labels = np.array(all_labels)
        
        # --- ADD THIS BLOCK TO TRIM THE DATA ---
        # Ensure we have exactly num_samples
        if all_codes.shape[0] > num_samples:
            all_codes = all_codes[:num_samples]
            all_labels = all_labels[:num_samples]
        # --- END OF BLOCK ---
        # --- END OF REPLACED BLOCK ---
        
        # Use t-SNE for 2D visualization (n_iter is now max_iter)
        tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=300)
        codes_2d = tsne.fit_transform(all_codes)
        
        plt.figure(figsize=(12, 10))
        scatter = plt.scatter(codes_2d[:, 0], codes_2d[:, 1], c=all_labels, 
                            cmap='viridis', alpha=0.6, s=10)
        plt.colorbar(scatter, label='Data Batch Index')
        plt.title(f't-SNE Visualization of Latent Space Embeddings (Level {level})')
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        plt.grid(True, alpha=0.3)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return codes_2d

    def plot_codebook_usage(self, dataloader, level=0, save_path='codebook_usage.png'):
        """Visualize codebook embedding usage statistics for a specific level."""
        self.model.eval()
        
        # Access the specific quantizer from your model's list
        codebook_size = self.model.quantizers[level].num_embeddings
        usage_count = torch.zeros(codebook_size)
        
        with torch.no_grad():
            for data, _ in tqdm(dataloader, desc=f"Calculating Codebook Usage (Level {level})"):
                data = data.to(self.device)
                
                # Get the list of encoding indices
                _, encoding_indices_list = self.model.encode(data)
                
                # Get the indices for the specified level
                encoding_indices = encoding_indices_list[level]
                
                # Count usage of each code
                unique, counts = torch.unique(encoding_indices, return_counts=True)
                usage_count[unique.cpu()] += counts.cpu()
        
        # Plot usage distribution
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 2, 1)
        plt.bar(range(codebook_size), usage_count.numpy(), alpha=0.7)
        plt.xlabel('Codebook Index')
        plt.ylabel('Usage Count')
        plt.title(f'Codebook Usage Distribution (Level {level})')
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
        print(f"--- Codebook Usage (Level {level}) ---")
        print(f"Codebook Usage Rate: {usage_rate.item()*100:.2f}% ({torch.sum(usage_count > 0)} / {codebook_size})")
        print(f"Most used code: {torch.argmax(usage_count).item()} (count: {torch.max(usage_count).item():.0f})")
        print(f"Least used code: {torch.argmin(usage_count).item()} (count: {torch.min(usage_count).item():.0f})")
        
        return usage_count
    
    def plot_reconstruction_quality_analysis(self, dataloader, num_samples=5, save_path='reconstruction_quality.png'):
        """
        Detailed analysis of reconstruction quality.
        This function worked correctly as-is, but we must use a denormalized
        SSIM calculation.
        """
        self.model.eval()
        
        data_iter = iter(dataloader)
        images, _ = next(data_iter)
        images = images[:num_samples].to(self.device)
        
        with torch.no_grad():
            reconstructions, _, _ = self.model(images)
        
        # Denormalize from [-1, 1] to [0, 1] for visualization and SSIM
        images_norm = (images + 1) / 2
        reconstructions_norm = (reconstructions + 1) / 2
        
        fig, axes = plt.subplots(4, num_samples, figsize=(4 * num_samples, 16))
        
        if num_samples == 1:
            axes = axes.reshape(4, 1)
        
        for i in range(num_samples):
            # Original
            axes[0, i].imshow(images_norm[i].cpu().squeeze(), cmap='gray')
            axes[0, i].set_title(f'Original {i+1}')
            axes[0, i].axis('off')
            
            # Reconstruction
            axes[1, i].imshow(reconstructions_norm[i].cpu().squeeze(), cmap='gray')
            
            # Calculate SSIM on the [0, 1] normalized images
            ssim = calculate_ssim(reconstructions_norm[i:i+1], images_norm[i:i+1], data_range=1.0).item()
            axes[1, i].set_title(f'Reconstruction\nSSIM: {ssim:.4f}')
            axes[1, i].axis('off')
            
            # Absolute difference
            diff = torch.abs(images_norm[i] - reconstructions_norm[i])
            im = axes[2, i].imshow(diff.cpu().squeeze(), cmap='hot', vmin=0, vmax=1)
            axes[2, i].set_title(f'Absolute Difference\nMax: {diff.max().item():.3f}')
            axes[2, i].axis('off')
            
            # Error histogram
            axes[3, i].hist(diff.cpu().flatten().numpy(), bins=50, range=(0, 1), alpha=0.7)
            axes[3, i].set_title(f'Error Distribution\nMean: {diff.mean().item():.4f}')
            axes[3, i].set_xlabel('Error Magnitude')
            axes[3, i].set_ylabel('Frequency')
            axes[3, i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()