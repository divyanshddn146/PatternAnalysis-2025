"""
This script loads a trained VQ-VAE model and performs comprehensive evaluation:
1. Quantitative metrics on the held-out test set (SSIM, MSE, perceptual loss)
2. Qualitative visualizations on the validation set (reconstructions, codebook usage, latent space)

The test set provides unbiased performance metrics, while validation set is used for visualizations
since we've already used it during training to select the best model.
"""

import torch
import os
from tqdm import tqdm
from modules import EnhancedVQVAE2
from utils import AdvancedVisualizations

# Import the data loading function from our dataset module
try:
    from dataset import create_data_loaders_keras
except ImportError:
    print("Error: Could not find 'dataset.py' or 'create_data_loaders_keras' function.")
    print("Please make sure your data loading script is in the same directory.")
    exit()

# --- Configuration ---
# Path to the trained model checkpoint with complete training history
CHECKPOINT_PATH = 'enhanced_vqvae2_checkpoints/final_model_complete_history.pth'
NUM_SAMPLES = 5  # Number of images to show in the reconstruction comparison plot
TSNE_SAMPLES = 1000  # Number of samples for t-SNE visualization (t-SNE is computationally expensive)
# ---------------------

def evaluate_test_set(model, test_loader, device):
    """
    Evaluates the trained model on the held-out test set to get final, unbiased performance metrics.
    
    This is the most important evaluation because the test set was never seen during training,
    so these metrics tell us how well the model will perform on new, real-world data.
    
    Args:
        model: The trained VQ-VAE model
        test_loader: DataLoader containing the test set
        device: torch.device (CPU or CUDA)
    
    Prints:
        Average test loss, SSIM, MSE, and perceptual loss across the entire test set
    """
    model.eval()  # Set to evaluation mode - disables dropout, batchnorm updates, etc.
    
    # Accumulators for metrics
    total_loss = 0
    total_ssim = 0
    total_mse = 0
    total_perceptual = 0
    num_batches = 0
    
    # These loss weights MUST match the ones used during training in train.py
    # Otherwise our test loss won't be comparable to training/validation loss
    lambda_rec = 1.0
    lambda_perceptual = 0.8
    lambda_ssim = 0.5
    
    print("\n--- Running Final Evaluation on Test Set ---")
    
    # No gradient computation needed - we're only evaluating, not training
    with torch.no_grad():
        pbar = tqdm(test_loader, desc="Testing Model")
        for data, _ in pbar:
            data = data.to(device)
            
            # Forward pass through the model
            reconstructions, vq_loss, _ = model(data)
            
            # Calculate the same enhanced loss we used during training
            # This ensures fair comparison with training/validation metrics
            total_loss_batch, components = model.calculate_enhanced_loss(
                data, reconstructions, vq_loss,
                lambda_rec, lambda_perceptual, lambda_ssim
            )
            
            # Accumulate all metrics
            total_loss += total_loss_batch.item()
            total_ssim += components['ssim_value']
            total_mse += components['mse']
            total_perceptual += components['perceptual']
            num_batches += 1
            
            # Show progress with current batch metrics
            pbar.set_postfix({
                'Test Loss': f'{total_loss_batch.item():.4f}',
                'Test SSIM': f'{components["ssim_value"]:.4f}'
            })

    # Calculate final averages across all test batches
    avg_loss = total_loss / num_batches
    avg_ssim = total_ssim / num_batches
    avg_mse = total_mse / num_batches
    avg_perceptual = total_perceptual / num_batches
    
    # Print the final, unbiased performance metrics
    print("\n--- Test Set Evaluation Complete ---")
    print(f"  Average Test Loss:   {avg_loss:.4f}")
    print(f"  Average Test SSIM:   {avg_ssim:.4f}")
    print(f"  Average MSE Loss:    {avg_mse:.4f}")
    print(f"  Average Perceptual:  {avg_perceptual:.4f}")
    print("----------------------------------------")


def main():
    """
    Main evaluation pipeline:
    1. Load the trained model from checkpoint
    2. Load test and validation datasets
    3. Run quantitative evaluation on test set (unbiased metrics)
    4. Generate qualitative visualizations using validation set
    
    The separation of test (quantitative) and validation (qualitative) ensures:
    - Test metrics are completely unbiased
    - Visualizations use data we've seen during training selection
    """
    
    # --- Step 1: Load the trained model checkpoint ---
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"Error: Checkpoint file not found at '{CHECKPOINT_PATH}'")
        return

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=torch.device('cpu'))
    config = checkpoint['config']  # Get the model configuration used during training
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Recreate the model architecture with the same hyperparameters used during training
    print("Loading trained model...")
    model = EnhancedVQVAE2(
        in_channels=1,  # Grayscale MRI images
        base_channels=config.get('base_channels', 64),
        num_levels=config.get('num_levels', 3),
        embedding_dims=config.get('embedding_dims', [256, 128, 64]),
        num_embeddings_list=config.get('num_embeddings_list', [512, 512, 512]),
        commitment_cost=config.get('commitment_cost', 0.1),
        num_residual_blocks=config.get('num_residual_blocks', 2),
        device=device
    )
    
    # Load the trained weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()  # Set to evaluation mode
    print("Model loaded successfully.")

    # --- Step 2: Load the datasets ---
    print("Loading data...")
    # We need both validation and test loaders:
    # - Test loader: for unbiased quantitative metrics
    # - Validation loader: for qualitative visualizations
    _, val_loader, test_loader, in_channels = create_data_loaders_keras(
        config['data_dir'],
        batch_size=config['batch_size'],
        target_size=config['image_size']
    )
    
    # Update model's input/output layers to match the actual number of channels
    # This handles cases where the checkpoint might have different channel settings
    model.bottom_up_encoder.initial_conv[0] = torch.nn.Conv2d(
        in_channels, 
        model.bottom_up_encoder.initial_conv[0].out_channels, 
        4, 2, 1
    ).to(device)
    model.decoder.network[-2] = torch.nn.Conv2d(
        model.decoder.network[-2].in_channels, 
        in_channels, 
        3, 1, 1
    ).to(device)
    
    # Reload weights after architecture modification
    model.load_state_dict(checkpoint['model_state_dict'])

    # --- Step 3: Quantitative Evaluation on Test Set ---
    # This gives us the final, unbiased performance metrics
    # The test set was never used during training, so these numbers tell us
    # how well the model will perform on completely new data
    evaluate_test_set(model, test_loader, device)

    # --- Step 4: Qualitative Visualizations on Validation Set ---
    # We use validation set here because:
    # 1. It was used to select the best model, so visualizing it is informative
    # 2. It keeps the test set purely for quantitative metrics
    # 3. The patterns should be similar anyway since both come from the same distribution
    
    print("\n--- Running Qualitative Visualizations (on Validation Set) ---")
    
    # Create the visualization object
    viz = AdvancedVisualizations(model=model, device=device)

    # Generate three key analysis plots:
    
    # Plot 1: Reconstruction Quality
    # Shows original images, reconstructions, difference maps, and error histograms
    viz.plot_reconstruction_quality_analysis(
        val_loader, 
        num_samples=NUM_SAMPLES,
        save_path='reconstruction_quality.png'
    )

    # Plot 2: Codebook Usage
    # Shows which codebook entries are being used and how frequently
    # Helps us understand if the model is using its full capacity or "collapsing"
    viz.plot_codebook_usage(
        val_loader, 
        level=0,  # Top-level codemap
        save_path='codebook_usage_level_0.png'
    )
    
    # Plot 3: Latent Space Visualization (t-SNE)
    # Projects the high-dimensional codebook vectors to 2D to visualize clustering
    # Shows if the model has learned structured, meaningful representations
    viz.plot_latent_space_2d(
        val_loader, 
        level=0,  # Top-level codemap
        num_samples=TSNE_SAMPLES,  # Limit samples since t-SNE is slow
        save_path='latent_space_2d_level_0.png'
    )

    print("\nAll analyses complete!")
    print("Check the 'plots/' directory for generated visualizations.")

if __name__ == '__main__':
    main()