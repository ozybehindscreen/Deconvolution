"""
Method 2: Gradient Ascent (Class Model Visualization)

This technique generates synthetic images that maximize the activation of a 
specific class neuron. It reveals what patterns the network has learned to 
associate with each class.

How it works:
1. Start with random noise (or a blank image)
2. Forward pass through the network
3. Compute gradient of target class score with respect to input image
4. Update the image in the direction that increases the class score
5. Apply regularization (e.g., L2 norm, blur) to produce interpretable images
6. Repeat until convergence

Reference: Simonyan et al. (2014) - "Deep Inside Convolutional Networks"
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from tqdm import tqdm
from PIL import Image

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model.cnn_model import CIFAR10CNN, CIFAR10_CLASSES


def get_device():
    """Get the best available device."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_model(model_path='checkpoints/best_model.pth'):
    """Load the trained model."""
    device = get_device()
    model = CIFAR10CNN(num_classes=10)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model, device


def create_initial_image(size=(32, 32), mode='noise', device='cpu'):
    """
    Create an initial image for gradient ascent.
    
    Args:
        size: Image size (H, W)
        mode: 'noise' for random noise, 'gray' for gray image
        device: Computation device
        
    Returns:
        image: Tensor [1, 3, H, W]
    """
    H, W = size
    
    if mode == 'noise':
        # Random noise centered around gray (0.5)
        image = torch.randn(1, 3, H, W) * 0.01 + 0.5
    elif mode == 'gray':
        # Uniform gray
        image = torch.ones(1, 3, H, W) * 0.5
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    return image.to(device).requires_grad_(True)


def normalize_for_model(image, mean=(0.4914, 0.4822, 0.4465), 
                        std=(0.2470, 0.2435, 0.2616)):
    """Normalize image for the model (CIFAR-10 normalization)."""
    mean = torch.tensor(mean).view(1, 3, 1, 1).to(image.device)
    std = torch.tensor(std).view(1, 3, 1, 1).to(image.device)
    return (image - mean) / std


def apply_regularization(image, l2_reg=0.0001):
    """
    Apply L2 regularization to encourage simpler patterns.
    
    Args:
        image: Input image tensor
        l2_reg: L2 regularization strength
        
    Returns:
        reg_loss: L2 regularization loss
    """
    return l2_reg * torch.sum(image ** 2)


def apply_jitter(image, jitter_amount=4):
    """
    Apply random jitter (translation) to prevent high-frequency artifacts.
    
    Args:
        image: Input image tensor [B, C, H, W]
        jitter_amount: Maximum pixels to shift
        
    Returns:
        jittered_image: Translated image
    """
    if jitter_amount == 0:
        return image
    
    _, _, H, W = image.shape
    
    # Random shifts
    shift_x = np.random.randint(-jitter_amount, jitter_amount + 1)
    shift_y = np.random.randint(-jitter_amount, jitter_amount + 1)
    
    # Apply translation using roll
    image = torch.roll(image, shifts=(shift_y, shift_x), dims=(2, 3))
    
    return image, (shift_y, shift_x)


def undo_jitter(image, shifts):
    """Undo the jitter operation."""
    shift_y, shift_x = shifts
    return torch.roll(image, shifts=(-shift_y, -shift_x), dims=(2, 3))


def apply_blur(image, sigma=0.5):
    """
    Apply Gaussian blur to reduce high-frequency patterns.
    
    Args:
        image: Image tensor [B, C, H, W]
        sigma: Blur sigma
        
    Returns:
        blurred_image: Blurred image tensor
    """
    image_np = image.detach().cpu().numpy()
    
    for b in range(image_np.shape[0]):
        for c in range(image_np.shape[1]):
            image_np[b, c] = gaussian_filter(image_np[b, c], sigma=sigma)
    
    return torch.from_numpy(image_np).to(image.device).requires_grad_(True)


def gradient_ascent(model, target_class, device, 
                    num_iterations=500,
                    learning_rate=0.1,
                    l2_reg=0.0001,
                    blur_freq=10,
                    blur_sigma=0.3,
                    jitter_amount=2,
                    init_mode='noise',
                    image_size=(32, 32)):
    """
    Generate a class model visualization using gradient ascent.
    
    Args:
        model: Trained CNN model
        target_class: Class index to visualize
        device: Computation device
        num_iterations: Number of optimization iterations
        learning_rate: Step size for gradient ascent
        l2_reg: L2 regularization strength
        blur_freq: Apply blur every N iterations
        blur_sigma: Gaussian blur sigma
        jitter_amount: Random jitter amount
        init_mode: Initial image mode ('noise' or 'gray')
        image_size: Size of generated image
        
    Returns:
        image: Generated image tensor
        history: List of intermediate images
    """
    model.eval()
    
    # Create initial image
    image = create_initial_image(image_size, mode=init_mode, device=device)
    
    # Optimizer
    optimizer = torch.optim.Adam([image], lr=learning_rate)
    
    # History for visualization
    history = []
    scores = []
    
    pbar = tqdm(range(num_iterations), desc=f'Generating {CIFAR10_CLASSES[target_class]}')
    
    for iteration in pbar:
        optimizer.zero_grad()
        
        # Apply jitter
        jittered_image, shifts = apply_jitter(image, jitter_amount)
        
        # Normalize for model
        normalized = normalize_for_model(jittered_image)
        
        # Forward pass
        output = model(normalized)
        
        # Target: maximize class score (negate for gradient descent optimizer)
        class_score = output[0, target_class]
        
        # Regularization
        reg_loss = apply_regularization(image, l2_reg)
        
        # Total loss (negative because we're maximizing)
        loss = -class_score + reg_loss
        
        # Backward pass
        loss.backward()
        
        # Update image
        optimizer.step()
        
        # Clamp to valid range
        with torch.no_grad():
            image.clamp_(0, 1)
        
        # Apply blur periodically
        if blur_freq > 0 and (iteration + 1) % blur_freq == 0:
            image = apply_blur(image, sigma=blur_sigma)
        
        # Record history
        scores.append(class_score.item())
        if iteration % (num_iterations // 10) == 0:
            history.append(image.detach().clone())
        
        pbar.set_postfix({'score': f'{class_score.item():.4f}'})
    
    # Final image
    history.append(image.detach().clone())
    
    return image.detach(), history, scores


def visualize_class_model(image, class_name, history=None, 
                          scores=None, save_path=None):
    """
    Visualize the generated class model.
    """
    if history is not None and len(history) > 1:
        # Show evolution
        num_steps = min(len(history), 6)
        fig, axes = plt.subplots(2, num_steps, figsize=(num_steps * 2.5, 5))
        
        # Top row: evolution
        step_indices = np.linspace(0, len(history) - 1, num_steps).astype(int)
        for i, idx in enumerate(step_indices):
            img = history[idx].squeeze().permute(1, 2, 0).cpu().numpy()
            img = np.clip(img, 0, 1)
            axes[0, i].imshow(img)
            axes[0, i].set_title(f'Step {idx * 10}')
            axes[0, i].axis('off')
        
        # Bottom left: final image (larger)
        final_img = image.squeeze().permute(1, 2, 0).cpu().numpy()
        final_img = np.clip(final_img, 0, 1)
        
        # Merge bottom row for final image
        gs = axes[1, 0].get_gridspec()
        for ax in axes[1, :3]:
            ax.remove()
        axbig = fig.add_subplot(gs[1, :3])
        axbig.imshow(final_img)
        axbig.set_title(f'Final: {class_name}', fontsize=14, fontweight='bold')
        axbig.axis('off')
        
        # Score evolution
        for ax in axes[1, 3:]:
            ax.remove()
        axscore = fig.add_subplot(gs[1, 3:])
        if scores:
            axscore.plot(scores, color='blue', linewidth=2)
            axscore.set_xlabel('Iteration')
            axscore.set_ylabel('Class Score')
            axscore.set_title('Score Evolution')
            axscore.grid(True, alpha=0.3)
        
        plt.suptitle(f'Class Model Visualization: {class_name}', 
                     fontsize=14, fontweight='bold')
    else:
        # Just show the final image
        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        img = image.squeeze().permute(1, 2, 0).cpu().numpy()
        img = np.clip(img, 0, 1)
        ax.imshow(img)
        ax.set_title(f'{class_name}', fontsize=14, fontweight='bold')
        ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    
    plt.show()
    plt.close()


def generate_all_class_models(model, device, save_dir='outputs/gradient_ascent'):
    """
    Generate class model visualizations for all CIFAR-10 classes.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    all_images = []
    
    print("\n" + "="*60)
    print("  GENERATING CLASS MODEL VISUALIZATIONS")
    print("="*60)
    
    for class_idx in range(10):
        print(f"\n{'='*50}")
        print(f"Class {class_idx}: {CIFAR10_CLASSES[class_idx]}")
        print('='*50)
        
        # Generate visualization
        image, history, scores = gradient_ascent(
            model, class_idx, device,
            num_iterations=300,
            learning_rate=0.1,
            l2_reg=0.0001,
            blur_freq=5,
            blur_sigma=0.3,
            jitter_amount=2
        )
        
        all_images.append(image)
        
        # Save individual visualization
        save_path = os.path.join(save_dir, f'class_{class_idx}_{CIFAR10_CLASSES[class_idx]}.png')
        visualize_class_model(image, CIFAR10_CLASSES[class_idx], 
                             history, scores, save_path)
    
    # Create grid of all classes
    create_class_grid(all_images, save_dir)
    
    return all_images


def create_class_grid(images, save_dir):
    """
    Create a grid showing all class visualizations.
    """
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    
    for idx, (image, ax) in enumerate(zip(images, axes.flatten())):
        img = image.squeeze().permute(1, 2, 0).cpu().numpy()
        img = np.clip(img, 0, 1)
        ax.imshow(img)
        ax.set_title(CIFAR10_CLASSES[idx], fontsize=12, fontweight='bold')
        ax.axis('off')
    
    plt.suptitle('Class Model Visualizations (Gradient Ascent)', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, 'all_classes_grid.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nClass grid saved to {save_path}")
    plt.show()
    plt.close()


def compare_initializations(model, device, target_class=3, 
                            save_path='outputs/gradient_ascent/init_comparison.png'):
    """
    Compare different initialization methods.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    initializations = ['noise', 'gray']
    results = []
    
    print(f"\nComparing initializations for class: {CIFAR10_CLASSES[target_class]}")
    
    for init_mode in initializations:
        print(f"\nTrying {init_mode} initialization...")
        image, _, _ = gradient_ascent(
            model, target_class, device,
            num_iterations=300,
            init_mode=init_mode
        )
        results.append((init_mode, image))
    
    # Visualize comparison
    fig, axes = plt.subplots(1, len(results), figsize=(len(results) * 4, 4))
    
    for ax, (init_mode, image) in zip(axes, results):
        img = image.squeeze().permute(1, 2, 0).cpu().numpy()
        img = np.clip(img, 0, 1)
        ax.imshow(img)
        ax.set_title(f'Init: {init_mode}')
        ax.axis('off')
    
    plt.suptitle(f'Initialization Comparison: {CIFAR10_CLASSES[target_class]}', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Comparison saved to {save_path}")
    plt.show()
    plt.close()


def main():
    """Main function to run gradient ascent visualization."""
    print("="*60)
    print("  GRADIENT ASCENT (CLASS MODEL VISUALIZATION)")
    print("="*60)
    print("""
    This method synthesizes images that maximize the activation
    of a specific class neuron, revealing what patterns the
    network associates with each class.
    """)
    
    # Load model
    print("\n📦 Loading trained model...")
    try:
        model, device = load_model()
        print(f"Model loaded successfully on {device}")
    except FileNotFoundError:
        print("❌ Model not found! Please run train_cnn.py first.")
        return
    
    # Generate all class models
    print("\n🎨 Generating class model visualizations...")
    generate_all_class_models(model, device)
    
    # Compare initializations
    print("\n🔄 Comparing initialization methods...")
    compare_initializations(model, device, target_class=3)
    
    print("\n✅ Gradient ascent visualization complete!")
    print("Check the 'outputs/gradient_ascent/' directory for results.")


if __name__ == '__main__':
    main()
