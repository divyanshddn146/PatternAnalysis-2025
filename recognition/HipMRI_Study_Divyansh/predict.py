import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
from torchvision.utils import make_grid
import h5py
import glob

from modules import VQVAE
from dataset import ProstateMRIDataset, create_data_loaders_keras
from utils import AdvancedVisualizations

def load_model(checkpoint_path, device, config):
    """Load trained model from checkpoint"""
    model = VQVAE(
        in_channels=config['in_channels'],
        hidden_dims=config['hidden_dims'],
        num_embeddings=config['num_embeddings'],
        embedding_dim=config['embedding_dim'],
        commitment_cost=config['commitment_cost'],
        num_residual_layers=config['num_residual_layers']
    )
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"Loaded model from epoch {checkpoint['epoch']}")
    print(f"Validation SSIM: {checkpoint['val_ssim']:.4f}")
    
    return model, checkpoint

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

def visualize_reconstructions(model, dataloader, device, num_samples=8, save_path='reconstruction_comparison.png'):
    """Visualize original and reconstructed images"""
    model.eval()
    
    # Get a batch of data
    data_iter = iter(dataloader)
    images, _ = next(data_iter)
    images = images[:num_samples].to(device)
    
    with torch.no_grad():
        reconstructions, vq_loss, perplexity = model(images)
    
    # Denormalize from [-1, 1] to [0, 1]
    images = (images + 1) / 2
    reconstructions = (reconstructions + 1) / 2
    
    # Calculate SSIM for each image
    ssim_scores = []
    for i in range(num_samples):
        ssim = calculate_ssim(
            reconstructions[i:i+1], 
            images[i:i+1], 
            data_range=1.0
        )
        ssim_scores.append(ssim.item())
    
    # Create plot
    fig, axes = plt.subplots(3, num_samples, figsize=(20, 8))
    
    if num_samples == 1:
        axes = axes.reshape(3, 1)
    
    for i in range(num_samples):
        # Original
        axes[0, i].imshow(images[i].cpu().squeeze(), cmap='gray', vmin=0, vmax=1)
        axes[0, i].set_title(f'Original {i+1}')
        axes[0, i].axis('off')
        
        # Reconstruction
        axes[1, i].imshow(reconstructions[i].cpu().squeeze(), cmap='gray', vmin=0, vmax=1)
        axes[1, i].set_title(f'Recon (SSIM: {ssim_scores[i]:.3f})')
        axes[1, i].axis('off')
        
        # Difference
        diff = torch.abs(images[i] - reconstructions[i])
        axes[2, i].imshow(diff.cpu().squeeze(), cmap='hot', vmin=0, vmax=0.5)
        axes[2, i].set_title(f'Difference')
        axes[2, i].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Average SSIM: {np.mean(ssim_scores):.4f}")
    print(f"VQ Loss: {vq_loss.item():.4f}")
    print(f"Perplexity: {perplexity.item():.2f}")
    
    return ssim_scores, reconstructions

def generate_samples(model, device, num_samples=16, latent_size=(8, 8), save_path='generated_samples.png'):
    """Generate new samples by sampling from the codebook"""
    model.eval()
    
    with torch.no_grad():
        # Sample random codes from the codebook
        codebook_size = model.vector_quantizer.num_embeddings
        embedding_dim = model.vector_quantizer.embedding_dim
        
        # Random indices
        random_indices = torch.randint(0, codebook_size, (num_samples, *latent_size))
        random_indices = random_indices.flatten().to(device)
        
        # Get embeddings
        encodings = torch.zeros(random_indices.shape[0], codebook_size, device=device)
        encodings.scatter_(1, random_indices.unsqueeze(1), 1)
        quantized = torch.matmul(encodings, model.vector_quantizer.embedding.weight)
        
        # Reshape to spatial dimensions
        quantized = quantized.view(num_samples, *latent_size, embedding_dim)
        quantized = quantized.permute(0, 3, 1, 2).contiguous()
        
        # Decode
        generated_images = model.decode(quantized)
        generated_images = (generated_images + 1) / 2  # Denormalize
    
    # Create grid
    grid = make_grid(generated_images, nrow=4, normalize=False)
    
    plt.figure(figsize=(12, 12))
    if generated_images.shape[1] == 1:  # Grayscale
        plt.imshow(grid.permute(1, 2, 0).cpu().squeeze(), cmap='gray')
    else:  # RGB or multi-channel
        plt.imshow(grid.permute(1, 2, 0).cpu())
    
    plt.title(f'Generated Prostate MRI Samples (n={num_samples})')
    plt.axis('off')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    return generated_images

def evaluate_model(model, test_loader, device):
    """Comprehensive model evaluation"""
    model.eval()
    
    total_ssim = 0
    total_loss = 0
    total_perplexity = 0
    num_batches = 0
    
    ssim_scores = []
    
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(device)
            reconstructions, vq_loss, perplexity = model(data)
            
            # Calculate losses
            reconstruction_loss = F.mse_loss(reconstructions, data)
            loss = reconstruction_loss + vq_loss
            
            # Calculate SSIM
            ssim = calculate_ssim(reconstructions, data, data_range=2.0)
            
            total_ssim += ssim.item()
            total_loss += loss.item()
            total_perplexity += perplexity.item()
            num_batches += 1
            
            # Store individual SSIM scores
            batch_ssim = calculate_ssim(reconstructions, data, data_range=2.0)
            ssim_scores.extend([batch_ssim.item()] * data.shape[0])
    
    avg_ssim = total_ssim / num_batches
    avg_loss = total_loss / num_batches
    avg_perplexity = total_perplexity / num_batches
    
    # Calculate SSIM statistics
    ssim_scores = np.array(ssim_scores)
    ssim_std = np.std(ssim_scores)
    ssim_above_06 = np.mean(ssim_scores > 0.6) * 100
    
    print(f"\n📊 Model Evaluation Results:")
    print(f"Average SSIM: {avg_ssim:.4f} ± {ssim_std:.4f}")
    print(f"SSIM > 0.6: {ssim_above_06:.1f}% of samples")
    print(f"Average Loss: {avg_loss:.4f}")
    print(f"Average Perplexity: {avg_perplexity:.2f}")
    print(f"Samples with SSIM > 0.6: {np.sum(ssim_scores > 0.6)}/{len(ssim_scores)}")
    
    # Plot SSIM distribution
    plt.figure(figsize=(10, 6))
    plt.hist(ssim_scores, bins=20, alpha=0.7, edgecolor='black')
    plt.axvline(x=0.6, color='red', linestyle='--', label='Target SSIM (0.6)')
    plt.axvline(x=avg_ssim, color='blue', linestyle='--', label=f'Mean SSIM ({avg_ssim:.3f})')
    plt.xlabel('SSIM Score')
    plt.ylabel('Frequency')
    plt.title('Distribution of SSIM Scores on Test Set')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('ssim_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return avg_ssim, ssim_above_06

def generate_comprehensive_report(model, test_loader, device, save_dir='comprehensive_report'):
    """Generate comprehensive test report with all graphs"""
    os.makedirs(save_dir, exist_ok=True)
    
    visualizer = AdvancedVisualizations(model, device)
    
    print("📊 Generating Comprehensive Test Report...")
    
    # 1. Latent space visualization
    print("1. Visualizing latent space...")
    visualizer.plot_latent_space_2d(test_loader, save_path=os.path.join(save_dir, 'latent_space_2d.png'))
    
    # 2. Codebook usage analysis
    print("2. Analyzing codebook usage...")
    visualizer.plot_codebook_usage(test_loader, save_path=os.path.join(save_dir, 'codebook_usage.png'))
    
    # 3. Reconstruction quality analysis
    print("3. Analyzing reconstruction quality...")
    visualizer.plot_reconstruction_quality_analysis(test_loader, save_path=os.path.join(save_dir, 'reconstruction_quality.png'))
    
    # 4. Generate samples
    print("4. Generating new samples...")
    generated = generate_samples(model, device, save_path=os.path.join(save_dir, 'generated_samples.png'))
    
    # 5. Performance metrics
    print("5. Calculating performance metrics...")
    avg_ssim, ssim_above_06 = evaluate_model(model, test_loader, device)
    
    # Create summary report
    create_summary_report(model, test_loader, device, avg_ssim, ssim_above_06, save_dir)
    
    print(f"✅ Comprehensive report saved to: {save_dir}")

def create_summary_report(model, test_loader, device, avg_ssim, ssim_above_06, save_dir):
    """Create a text summary report"""
    report_path = os.path.join(save_dir, 'model_report.txt')
    
    with open(report_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("          VQVAE PROSTATE MRI MODEL REPORT\n")
        f.write("="*60 + "\n\n")
        
        f.write("MODEL ARCHITECTURE:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Encoder layers: {len(model.encoder.network)} layers\n")
        f.write(f"Decoder layers: {len(model.decoder.network)} layers\n")
        f.write(f"Codebook size: {model.vector_quantizer.num_embeddings}\n")
        f.write(f"Embedding dim: {model.vector_quantizer.embedding_dim}\n")
        f.write(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}\n\n")
        
        f.write("PERFORMANCE METRICS:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Average SSIM: {avg_ssim:.4f}\n")
        f.write(f"Samples with SSIM > 0.6: {ssim_above_06:.1f}%\n")
        f.write(f"Target Achievement: {'✅ SUCCESS' if avg_ssim > 0.6 else '❌ NEEDS IMPROVEMENT'}\n\n")
        
        f.write("DATA STATISTICS:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Test samples: {len(test_loader.dataset)}\n")
        f.write(f"Batch size: {test_loader.batch_size}\n")
        f.write(f"Image size: {test_loader.dataset.target_size}\n\n")
        
        f.write("GENERATED FILES:\n")
        f.write("-" * 30 + "\n")
        f.write("1. latent_space_2d.png - t-SNE visualization of discrete codes\n")
        f.write("2. codebook_usage.png - Codebook utilization statistics\n")
        f.write("3. reconstruction_quality.png - Detailed error analysis\n")
        f.write("4. generated_samples.png - Synthetic MRI samples\n")
        f.write("5. ssim_distribution.png - SSIM score distribution\n")
        f.write("6. reconstruction_comparison.png - Original vs Reconstruction\n")
    
    print(f"📄 Summary report: {report_path}")

def main():
    parser = argparse.ArgumentParser(description='VQVAE Prostate MRI Prediction')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--data_dir', type=str, default='HipMRI_Study_open', help='Root data directory')
    parser.add_argument('--num_samples', type=int, default=8, help='Number of samples to visualize')
    parser.add_argument('--generate', action='store_true', help='Generate new samples')
    parser.add_argument('--evaluate', action='store_true', help='Run comprehensive evaluation')
    parser.add_argument('--comprehensive', action='store_true', help='Generate comprehensive test report')
    
    args = parser.parse_args()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model and config
    if not os.path.exists(args.checkpoint):
        print(f"❌ Checkpoint not found: {args.checkpoint}")
        return
    
    # First load checkpoint to get config
    checkpoint = torch.load(args.checkpoint, map_location=device)
    config = checkpoint.get('config', {
        'in_channels': 1,
        'hidden_dims': [64, 128, 256],
        'num_embeddings': 512,
        'embedding_dim': 256,
        'commitment_cost': 0.25,
        'num_residual_layers': 2
    })
    
    model, checkpoint = load_model(args.checkpoint, device, config)
    
    # Create test dataset
    try:
        _, _, test_loader, _ = create_data_loaders_keras(
            args.data_dir,
            batch_size=args.num_samples,
            target_size=(128, 128),
            use_multimodal=config.get('use_multimodal', False)
        )
    except Exception as e:
        print(f"Error loading test data: {e}")
        return
    
    if args.comprehensive:
        print("\n📈 Generating Comprehensive Test Report...")
        generate_comprehensive_report(model, test_loader, device)
    else:
        print("\n1. 🖼️  Visualizing Reconstructions")
        ssim_scores, reconstructions = visualize_reconstructions(
            model, test_loader, device, args.num_samples
        )
        
        if args.generate:
            print("\n2. 🎨 Generating New Samples")
            generated_images = generate_samples(model, device, num_samples=16)
            print("✅ Generated samples saved to 'generated_samples.png'")
        
        if args.evaluate:
            print("\n3. 📈 Comprehensive Evaluation")
            # Create full test loader for evaluation
            _, _, full_test_loader, _ = create_data_loaders_keras(
                args.data_dir,
                batch_size=16,
                target_size=(128, 128),
                use_multimodal=config.get('use_multimodal', False)
            )
            
            avg_ssim, ssim_above_06 = evaluate_model(model, full_test_loader, device)
            
            print(f"\n🎯 Final Assessment:")
            if avg_ssim > 0.6 and ssim_above_06 > 50:
                print("✅ SUCCESS: Model achieved target performance!")
                print(f"   - Average SSIM: {avg_ssim:.4f} > 0.6")
                print(f"   - {ssim_above_06:.1f}% of samples have SSIM > 0.6")
            else:
                print("❌ Model needs improvement")
                print(f"   - Average SSIM: {avg_ssim:.4f} (target: > 0.6)")
                print(f"   - Only {ssim_above_06:.1f}% of samples have SSIM > 0.6")

if __name__ == "__main__":
    main()