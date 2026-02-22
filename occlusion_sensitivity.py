"""
Method 1: Occlusion Sensitivity

This technique helps understand which parts of an image are important for classification
by systematically occluding (covering) different regions and observing the change in 
classification probability.

How it works:
1. Take an input image and its predicted class
2. Slide an occluding patch across the image
3. For each position, record the probability of the original prediction
4. Regions where occlusion causes the largest drop in probability are most important

Reference: Zeiler & Fergus (2014) - "Visualizing and Understanding Convolutional Networks"
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
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


def get_transform():
    """Get the standard transform for CIFAR-10."""
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])


def denormalize(tensor, mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616)):
    """Denormalize a tensor for visualization."""
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    return tensor * std + mean


def occlusion_sensitivity(model, image, target_class, device, 
                          patch_size=4, stride=1, occlusion_value=0.5):
    """
    Compute occlusion sensitivity map for an image.
    
    Args:
        model: Trained CNN model
        image: Input image tensor [1, C, H, W]
        target_class: Class index to analyze
        device: Computation device
        patch_size: Size of the occluding patch
        stride: Stride for sliding the patch
        occlusion_value: Value to fill the occluded region (gray = 0.5)
        
    Returns:
        sensitivity_map: [H, W] heatmap showing sensitivity to occlusion
    """
    model.eval()
    image = image.to(device)
    
    _, C, H, W = image.shape
    
    # Get baseline prediction
    with torch.no_grad():
        baseline_output = model(image)
        baseline_prob = F.softmax(baseline_output, dim=1)[0, target_class].item()
    
    # Initialize sensitivity map
    sensitivity_map = np.zeros((H, W))
    count_map = np.zeros((H, W))
    
    # Slide the occluding patch
    for y in tqdm(range(0, H - patch_size + 1, stride), desc='Computing occlusion sensitivity'):
        for x in range(0, W - patch_size + 1, stride):
            # Create occluded image
            occluded = image.clone()
            occluded[:, :, y:y+patch_size, x:x+patch_size] = occlusion_value
            
            # Get prediction for occluded image
            with torch.no_grad():
                output = model(occluded)
                prob = F.softmax(output, dim=1)[0, target_class].item()
            
            # Record the drop in probability
            # Higher drop = more important region
            drop = baseline_prob - prob
            
            # Add to sensitivity map (accumulate for overlapping patches)
            sensitivity_map[y:y+patch_size, x:x+patch_size] += drop
            count_map[y:y+patch_size, x:x+patch_size] += 1
    
    # Average overlapping regions
    count_map[count_map == 0] = 1  # Avoid division by zero
    sensitivity_map = sensitivity_map / count_map
    
    return sensitivity_map, baseline_prob


def visualize_occlusion_sensitivity(image, sensitivity_map, class_name, 
                                    baseline_prob, save_path=None):
    """
    Visualize the occlusion sensitivity map overlaid on the original image.
    """
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    # Original image
    img_np = denormalize(image.squeeze(0)).permute(1, 2, 0).numpy()
    img_np = np.clip(img_np, 0, 1)
    
    axes[0].imshow(img_np)
    axes[0].set_title(f'Original Image\nClass: {class_name}')
    axes[0].axis('off')
    
    # Sensitivity map (raw)
    im1 = axes[1].imshow(sensitivity_map, cmap='hot')
    axes[1].set_title('Sensitivity Map\n(Higher = More Important)')
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    
    # Overlay
    axes[2].imshow(img_np)
    # Resize sensitivity map if needed
    sensitivity_resized = sensitivity_map
    im2 = axes[2].imshow(sensitivity_resized, cmap='jet', alpha=0.5)
    axes[2].set_title(f'Overlay\nBaseline Prob: {baseline_prob:.3f}')
    axes[2].axis('off')
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    
    # Thresholded important regions
    threshold = np.percentile(sensitivity_map, 75)
    important_mask = sensitivity_map > threshold
    masked_img = img_np.copy()
    for c in range(3):
        channel = masked_img[:, :, c]
        channel[~important_mask] = channel[~important_mask] * 0.3
    
    axes[3].imshow(masked_img)
    axes[3].set_title('Most Important Regions\n(Top 25%)')
    axes[3].axis('off')
    
    plt.suptitle('Occlusion Sensitivity Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    
    plt.show()
    plt.close()


def analyze_multiple_images(model, device, num_images=5, save_dir='outputs/occlusion'):
    """
    Analyze multiple images from the test set.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Load test dataset
    transform = get_transform()
    test_dataset = datasets.CIFAR10(root='./data', train=False, 
                                     download=True, transform=transform)
    
    # Select diverse images (one from each class)
    class_indices = {i: [] for i in range(10)}
    for idx, (_, label) in enumerate(test_dataset):
        class_indices[label].append(idx)
    
    # Analyze images
    for class_idx in range(min(num_images, 10)):
        if class_indices[class_idx]:
            img_idx = class_indices[class_idx][0]
            image, label = test_dataset[img_idx]
            image = image.unsqueeze(0)
            
            print(f"\n{'='*50}")
            print(f"Analyzing {CIFAR10_CLASSES[label]} (class {label})")
            print('='*50)
            
            # Compute occlusion sensitivity
            sensitivity_map, baseline_prob = occlusion_sensitivity(
                model, image, label, device,
                patch_size=4, stride=1
            )
            
            # Visualize
            save_path = os.path.join(save_dir, f'occlusion_{CIFAR10_CLASSES[label]}.png')
            visualize_occlusion_sensitivity(
                image, sensitivity_map, 
                CIFAR10_CLASSES[label], baseline_prob,
                save_path
            )


def demonstrate_occlusion_effect(model, device, save_path='outputs/occlusion/demo.png'):
    """
    Demonstrate how occlusion affects classification by showing 
    the image with progressive occlusion of important regions.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Load a sample image
    transform = get_transform()
    test_dataset = datasets.CIFAR10(root='./data', train=False, 
                                     download=True, transform=transform)
    
    # Get a well-classified image
    for idx in range(len(test_dataset)):
        image, label = test_dataset[idx]
        image = image.unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(image)
            prob = F.softmax(output, dim=1)[0, label].item()
        
        if prob > 0.9:  # Find a confidently classified image
            break
    
    print(f"Selected image: {CIFAR10_CLASSES[label]} (prob: {prob:.3f})")
    
    # Compute sensitivity
    sensitivity_map, baseline_prob = occlusion_sensitivity(
        model, image, label, device, patch_size=4, stride=1
    )
    
    # Progressive occlusion of important regions
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    percentiles = [0, 10, 25, 50, 75, 90, 95, 99, 100]
    
    image_np = denormalize(image.cpu().squeeze(0)).permute(1, 2, 0).numpy()
    image_np = np.clip(image_np, 0, 1)
    
    for i, pct in enumerate(percentiles[:10]):
        ax = axes[i // 5, i % 5]
        
        if pct == 0:
            ax.imshow(image_np)
            ax.set_title(f'Original\nProb: {baseline_prob:.3f}')
        else:
            threshold = np.percentile(sensitivity_map, 100 - pct)
            mask = sensitivity_map >= threshold
            
            occluded_img = image.clone()
            mask_tensor = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).float().to(device)
            occluded_img = occluded_img * (1 - mask_tensor) + 0.5 * mask_tensor
            
            with torch.no_grad():
                output = model(occluded_img)
                prob = F.softmax(output, dim=1)[0, label].item()
            
            occluded_np = denormalize(occluded_img.cpu().squeeze(0)).permute(1, 2, 0).numpy()
            occluded_np = np.clip(occluded_np, 0, 1)
            
            ax.imshow(occluded_np)
            ax.set_title(f'Top {pct}% occluded\nProb: {prob:.3f}')
        
        ax.axis('off')
    
    # Fill remaining subplot if any
    if len(percentiles) < 10:
        for i in range(len(percentiles), 10):
            axes[i // 5, i % 5].axis('off')
    
    plt.suptitle(f'Progressive Occlusion of Important Regions ({CIFAR10_CLASSES[label]})', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Demo saved to {save_path}")
    plt.show()
    plt.close()


def main():
    """Main function to run occlusion sensitivity analysis."""
    print("="*60)
    print("  OCCLUSION SENSITIVITY ANALYSIS")
    print("="*60)
    print("""
    This method reveals which parts of an image are crucial for 
    classification by systematically covering different regions 
    and observing changes in the model's predictions.
    """)
    
    # Load model
    print("\n📦 Loading trained model...")
    try:
        model, device = load_model()
        print(f"Model loaded successfully on {device}")
    except FileNotFoundError:
        print("❌ Model not found! Please run train_cnn.py first.")
        return
    
    # Analyze multiple images
    print("\n🔍 Analyzing images...")
    analyze_multiple_images(model, device, num_images=5)
    
    # Demonstrate progressive occlusion
    print("\n📊 Creating occlusion demonstration...")
    demonstrate_occlusion_effect(model, device)
    
    print("\n✅ Occlusion sensitivity analysis complete!")
    print("Check the 'outputs/occlusion/' directory for results.")


if __name__ == '__main__':
    main()
