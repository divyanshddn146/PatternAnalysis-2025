# run_analysis.py (to be renamed predict.py)
import torch
import os
from tqdm import tqdm
from modules_vqvae2_perceptual import EnhancedVQVAE2
from utils import AdvancedVisualizations

# IMPORTANT: You must import your data loading function
try:
    from dataset import create_data_loaders_keras
except ImportError:
    print("Error: Could not find 'dataset.py' or 'create_data_loaders_keras' function.")
    print("Please make sure your data loading script is in the same directory.")
    exit()

# --- Configuration ---
CHECKPOINT_PATH = 'enhanced_vqvae2_checkpoints/final_model_complete_history.pth'
NUM_SAMPLES = 5 # Number of images for the reconstruction plot
TSNE_SAMPLES = 1000 # How many samples to use for t-SNE (it's slow!)
# ---------------------

# --- NEW FUNCTION FOR TEST SET EVALUATION ---
def evaluate_test_set(model, test_loader, device):
    """
    Runs a full evaluation on the unseen test set and prints final metrics.
    """
    model.eval()
    total_loss = 0
    total_ssim = 0
    total_mse = 0
    total_perceptual = 0
    num_batches = 0
    
    # These loss weights must match the ones used in your train.py
    lambda_rec = 1.0
    lambda_perceptual = 0.8
    lambda_ssim = 0.5
    
    print("\n--- Running Final Evaluation on Test Set ---")
    
    with torch.no_grad():
        pbar = tqdm(test_loader, desc="Testing Model")
        for data, _ in pbar:
            data = data.to(device)
            reconstructions, vq_loss, _ = model(data)
            
            # Calculate the enhanced loss
            total_loss_batch, components = model.calculate_enhanced_loss(
                data, reconstructions, vq_loss,
                lambda_rec, lambda_perceptual, lambda_ssim
            )
            
            total_loss += total_loss_batch.item()
            total_ssim += components['ssim_value']
            total_mse += components['mse']
            total_perceptual += components['perceptual']
            num_batches += 1
            
            pbar.set_postfix({
                'Test Loss': f'{total_loss_batch.item():.4f}',
                'Test SSIM': f'{components["ssim_value"]:.4f}'
            })

    # Calculate averages
    avg_loss = total_loss / num_batches
    avg_ssim = total_ssim / num_batches
    avg_mse = total_mse / num_batches
    avg_perceptual = total_perceptual / num_batches
    
    print("\n--- Test Set Evaluation Complete ---")
    print(f"  Average Test Loss:   {avg_loss:.4f}")
    print(f"  Average Test SSIM:   {avg_ssim:.4f}")
    print(f"  Average MSE Loss:    {avg_mse:.4f}")
    print(f"  Average Perceptual:  {avg_perceptual:.4f}")
    print("----------------------------------------")
# --- END OF NEW FUNCTION ---


def main():
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"Error: Checkpoint file not found at '{CHECKPOINT_PATH}'")
        return

    # Load checkpoint and config
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=torch.device('cpu'))
    config = checkpoint['config']
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Load the trained model
    print("Loading trained model...")
    model = EnhancedVQVAE2(
        in_channels=1, 
        base_channels=config.get('base_channels', 64),
        num_levels=config.get('num_levels', 3),
        embedding_dims=config.get('embedding_dims', [256, 128, 64]),
        num_embeddings_list=config.get('num_embeddings_list', [512, 512, 512]),
        commitment_cost=config.get('commitment_cost', 0.1),
        num_residual_blocks=config.get('num_residual_blocks', 2),
        device=device
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    print("Model loaded successfully.")

    # 2. Load the validation and test data
    print("Loading data...")
    # --- MODIFIED ---
    # We now need all three data loaders
    _, val_loader, test_loader, in_channels = create_data_loaders_keras(
        config['data_dir'],
        batch_size=config['batch_size'],
        target_size=config['image_size']
    )
    # --- END OF MODIFICATION ---
    
    # Ensure model's in_channels is correct
    model.bottom_up_encoder.initial_conv[0] = torch.nn.Conv2d(in_channels, model.bottom_up_encoder.initial_conv[0].out_channels, 4, 2, 1).to(device)
    model.decoder.network[-2] = torch.nn.Conv2d(model.decoder.network[-2].in_channels, in_channels, 3, 1, 1).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])


    # 3. --- NEW STEP ---
    # Run quantitative evaluation on the test set FIRST
    evaluate_test_set(model, test_loader, device)
    # --- END OF NEW STEP ---


    # 4. Create the visualization object
    viz = AdvancedVisualizations(model=model, device=device)

    # 5. Run the qualitative visualizations on the validation set
    print("\n--- Running Qualitative Visualizations (on Validation Set) ---")
    
    viz.plot_reconstruction_quality_analysis(val_loader, num_samples=NUM_SAMPLES,
                                             save_path='reconstruction_quality.png')

    viz.plot_codebook_usage(val_loader, level=0, save_path='codebook_usage_level_0.png')
    
    viz.plot_latent_space_2d(val_loader, level=0, num_samples=TSNE_SAMPLES,
                             save_path='latent_space_2d_level_0.png')

    print("\nAll analyses complete!")

if __name__ == '__main__':
    main()