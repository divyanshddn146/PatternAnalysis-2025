# generate_training_plots.py
import torch
import os
import matplotlib.pyplot as plt
import numpy as np

def generate_training_plots(checkpoint_path):
    """
    Regenerates comprehensive training plots from checkpoint history.
    Generates six analysis plots:
      1. Train vs Validation Loss
      2. SSIM (with goal line)
      3. Perplexity
      4. Loss Components (MSE, Perceptual, VQ)
      5. Loss vs SSIM Correlation
      6. Loss Component Ratios (stacked contribution)
    """

    if not os.path.exists(checkpoint_path):
        print(f"❌ Error: Checkpoint not found at {checkpoint_path}")
        return

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    save_dir = "C:/Users/divya/Desktop/generative/plots"
    os.makedirs(save_dir, exist_ok=True)

    # Extract data
    train_losses = checkpoint.get('train_losses', [])
    val_losses = checkpoint.get('val_losses', [])
    train_ssim = checkpoint.get('train_ssim', [])
    val_ssim = checkpoint.get('val_ssim', [])
    perplexities = checkpoint.get('perplexities', [])
    loss_components_history = checkpoint.get('loss_components_history', [])

    # Extract component breakdown
    mse_vals = [d.get('mse', 0) for d in loss_components_history]
    perceptual_vals = [d.get('perceptual', 0) for d in loss_components_history]
    vq_vals = [d.get('vq', 0) for d in loss_components_history]
    total_vals = [d.get('total', 0) for d in loss_components_history]

    # ---- Plot 1: Training vs Validation Loss ----
    plt.figure(figsize=(8, 5))
    if train_losses: plt.plot(train_losses, label='Train Loss', linewidth=2)
    if val_losses: plt.plot(val_losses, label='Validation Loss', linewidth=2)
    plt.title('Training vs Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'loss_curve.png'))
    plt.close()

    # ---- Plot 2: SSIM ----
    if train_ssim or val_ssim:
        plt.figure(figsize=(8, 5))
        if train_ssim: plt.plot(train_ssim, label='Train SSIM', linewidth=2)
        if val_ssim: plt.plot(val_ssim, label='Validation SSIM', linewidth=2)
        plt.axhline(y=0.90, color='r', linestyle='--', linewidth=1.5, label='Goal SSIM = 0.90')
        plt.title('SSIM over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('SSIM')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'ssim_curve.png'))
        plt.close()

    # ---- Plot 3: Perplexity ----
    if perplexities:
        plt.figure(figsize=(8, 5))
        plt.plot(perplexities, label='Perplexity', color='orange', linewidth=2)
        plt.title('Codebook Perplexity over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Perplexity')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'perplexity_curve.png'))
        plt.close()

    # ---- Plot 4: Loss Components (MSE, Perceptual, VQ) ----
    if loss_components_history:
        plt.figure(figsize=(8, 5))
        plt.plot(mse_vals, label='MSE Loss', linewidth=2)
        plt.plot(perceptual_vals, label='Perceptual Loss', linewidth=2)
        plt.plot(vq_vals, label='VQ Loss', linewidth=2)
        plt.title('Loss Components (MSE, Perceptual, VQ)')
        plt.xlabel('Epoch')
        plt.ylabel('Loss Value')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'loss_components_curve.png'))
        plt.close()

    # ---- Plot 5: Loss vs SSIM ----
    if train_ssim and train_losses:
        min_len = min(len(train_losses), len(train_ssim))
        plt.figure(figsize=(6, 6))
        plt.scatter(train_losses[:min_len], train_ssim[:min_len], c=np.arange(min_len), cmap='viridis', s=30)
        plt.colorbar(label='Epoch')
        plt.title('Loss vs SSIM Correlation')
        plt.xlabel('Training Loss')
        plt.ylabel('SSIM')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'loss_vs_ssim.png'))
        plt.close()

    # ---- Plot 6: Loss Component Ratios (stacked) ----
    if loss_components_history and total_vals:
        total_vals = np.array(total_vals) + 1e-8  # avoid divide by zero
        mse_ratio = np.array(mse_vals) / total_vals
        perceptual_ratio = np.array(perceptual_vals) / total_vals
        vq_ratio = np.array(vq_vals) / total_vals

        plt.figure(figsize=(8, 5))
        plt.plot(mse_ratio, label='MSE Ratio', linewidth=2)
        plt.plot(perceptual_ratio, label='Perceptual Ratio', linewidth=2)
        plt.plot(vq_ratio, label='VQ Ratio', linewidth=2)
        plt.title('Loss Component Ratios over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Relative Contribution')
        plt.legend(loc='upper right')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'loss_component_ratios.png'))
        plt.close()

    print(f"✅ All 6 plots saved in: {save_dir}")
    print("📊 Files generated:")
    print("- loss_curve.png")
    print("- ssim_curve.png")
    print("- perplexity_curve.png")
    print("- loss_components_curve.png")
    print("- loss_vs_ssim.png")
    print("- loss_component_ratios.png")

if __name__ == "__main__":
    generate_training_plots("C:/Users/divya/Desktop/generative/final_model_complete_history.pth")
