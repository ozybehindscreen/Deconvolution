"""
Method 3: Class Activation Maps (CAM)

This technique generates heatmaps that highlight the regions of an image
that were most important for the model's classification decision.

How it works:
1. Get the feature maps from the last convolutional layer
2. Get the weights from the fully connected layer for the target class
3. Compute weighted combination of feature maps using these weights
4. Upsample the resulting heatmap to the input image size
5. Overlay on the original image to visualize important regions

Requirements:
- The network must have a Global Average Pooling (GAP) layer after the last conv layer
- Followed by a single fully connected layer for classification

Reference: Zhou et al. (2016) - "Learning Deep Features for Discriminative Localization"
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from PIL import Image
import cv2

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


class CAMGenerator:
    """
    Class Activation Map generator using hooks.
    
    This class uses PyTorch hooks to capture activations from the
    last convolutional layer during forward pass.
    """
    
    def __init__(self, model, target_layer=None):
        """
        Initialize CAM generator.
        
        Args:
            model: The CNN model
            target_layer: Name of the target layer (default: conv4)
        """
        self.model = model
        self.activations = None
        self.gradients = None
        
        # Register hooks on the target layer
        if target_layer is None:
            target_layer = model.conv4  # Last conv layer
        
        self.hook_handles = []
        self.hook_handles.append(
            target_layer.register_forward_hook(self._save_activation)
        )
        self.hook_handles.append(
            target_layer.register_full_backward_hook(self._save_gradient)
        )
    
    def _save_activation(self, module, input, output):
        """Hook to save activations during forward pass."""
        self.activations = output.detach()
    
    def _save_gradient(self, module, grad_in, grad_out):
        """Hook to save gradients during backward pass."""
        self.gradients = grad_out[0].detach()
    
    def remove_hooks(self):
        """Remove the hooks."""
        for handle in self.hook_handles:
            handle.remove()
    
    def generate_cam(self, class_idx):
        """
        Generate Class Activation Map for a specific class.
        
        Args:
            class_idx: Target class index
            
        Returns:
            cam: Normalized CAM heatmap [H, W]
        """
        # Get the weights for the target class from FC layer
        weights = self.model.fc.weight[class_idx]  # [512]
        
        # Compute weighted combination of activation maps
        # activations shape: [1, 512, H, W]
        cam = torch.zeros(self.activations.shape[2:], device=self.activations.device)
        
        for i, w in enumerate(weights):
            cam += w * self.activations[0, i, :, :]
        
        # Apply ReLU (we only want positive contributions)
        cam = F.relu(cam)
        
        # Normalize to [0, 1]
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        
        return cam.cpu().numpy()
    
    def generate_grad_cam(self, class_idx, image):
        """
        Generate Gradient-weighted CAM (Grad-CAM).
        
        This is an extension of CAM that uses gradients for weighting,
        allowing it to work with any CNN architecture.
        
        Args:
            class_idx: Target class index
            image: Input image tensor
            
        Returns:
            grad_cam: Normalized Grad-CAM heatmap [H, W]
        """
        # Forward pass
        self.model.zero_grad()
        output = self.model(image)
        
        # Backward pass for target class
        output[0, class_idx].backward(retain_graph=True)
        
        # Get gradient weights (global average pooling of gradients)
        weights = self.gradients.mean(dim=(2, 3))  # [1, 512]
        
        # Compute weighted combination
        grad_cam = torch.zeros(self.activations.shape[2:], device=self.activations.device)
        
        for i, w in enumerate(weights[0]):
            grad_cam += w * self.activations[0, i, :, :]
        
        # ReLU
        grad_cam = F.relu(grad_cam)
        
        # Normalize
        grad_cam = grad_cam - grad_cam.min()
        if grad_cam.max() > 0:
            grad_cam = grad_cam / grad_cam.max()
        
        return grad_cam.cpu().numpy()


def upsample_cam(cam, target_size):
    """
    Upsample CAM to match the input image size.
    
    Args:
        cam: CAM heatmap [H, W]
        target_size: (target_H, target_W)
        
    Returns:
        upsampled_cam: Upsampled heatmap
    """
    cam = cam.astype(np.float32)
    return cv2.resize(cam, target_size[::-1], interpolation=cv2.INTER_LINEAR)


def apply_colormap(cam, colormap=cv2.COLORMAP_JET):
    """
    Apply a colormap to the CAM.
    
    Args:
        cam: Normalized CAM [0, 1]
        colormap: OpenCV colormap
        
    Returns:
        colored_cam: RGB colored CAM
    """
    cam_uint8 = (cam * 255).astype(np.uint8)
    colored = cv2.applyColorMap(cam_uint8, colormap)
    return cv2.cvtColor(colored, cv2.COLOR_BGR2RGB) / 255.0


def overlay_cam(image, cam, alpha=0.5):
    """
    Overlay CAM on the original image.
    
    Args:
        image: Original image [H, W, 3]
        cam: Colored CAM [H, W, 3]
        alpha: Transparency of the overlay
        
    Returns:
        overlay: Blended image
    """
    return (1 - alpha) * image + alpha * cam


def visualize_cam(image, cam, class_name, pred_class, confidence, 
                  save_path=None, method='CAM'):
    """
    Visualize the CAM with the original image.
    """
    # Upsample CAM
    H, W = image.shape[1], image.shape[2]
    cam_upsampled = upsample_cam(cam, (H, W))
    
    # Convert image to numpy
    img_np = denormalize(image).permute(1, 2, 0).cpu().numpy()
    img_np = np.clip(img_np, 0, 1)
    
    # Create colored CAM
    colored_cam = apply_colormap(cam_upsampled)
    
    # Create overlay
    overlay = overlay_cam(img_np, colored_cam, alpha=0.4)
    
    # Plot
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    # Original image
    axes[0].imshow(img_np)
    axes[0].set_title(f'Original Image\nTrue: {class_name}')
    axes[0].axis('off')
    
    # CAM heatmap
    im = axes[1].imshow(cam_upsampled, cmap='jet')
    axes[1].set_title(f'{method} Heatmap')
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    
    # Colored CAM
    axes[2].imshow(colored_cam)
    axes[2].set_title(f'Colored {method}')
    axes[2].axis('off')
    
    # Overlay
    axes[3].imshow(overlay)
    axes[3].set_title(f'Overlay\nPred: {pred_class} ({confidence:.2%})')
    axes[3].axis('off')
    
    plt.suptitle(f'{method} Visualization', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    
    plt.show()
    plt.close()


def compare_cam_methods(model, image, true_class, device, save_path=None):
    """
    Compare standard CAM and Grad-CAM.
    """
    model.eval()
    image = image.to(device)
    if image.dim() == 3:
        image = image.unsqueeze(0)
    
    # Forward pass
    with torch.no_grad():
        output = model(image)
        probs = F.softmax(output, dim=1)
        pred_class = output.argmax(dim=1).item()
        confidence = probs[0, pred_class].item()
    
    # Generate CAMs
    cam_gen = CAMGenerator(model)
    
    # Standard CAM (for predicted class)
    cam = cam_gen.generate_cam(pred_class)
    
    # Grad-CAM (for predicted class)
    image.requires_grad_(True)
    grad_cam = cam_gen.generate_grad_cam(pred_class, image)
    
    cam_gen.remove_hooks()
    
    # Upsample
    H, W = image.shape[2], image.shape[3]
    cam_upsampled = upsample_cam(cam, (H, W))
    grad_cam_upsampled = upsample_cam(grad_cam, (H, W))
    
    # Convert image
    img_np = denormalize(image.squeeze(0).detach()).permute(1, 2, 0).cpu().numpy()
    img_np = np.clip(img_np, 0, 1)
    
    # Colored CAMs
    cam_colored = apply_colormap(cam_upsampled)
    grad_cam_colored = apply_colormap(grad_cam_upsampled)
    
    # Overlays
    cam_overlay = overlay_cam(img_np, cam_colored, alpha=0.4)
    grad_cam_overlay = overlay_cam(img_np, grad_cam_colored, alpha=0.4)
    
    # Plot comparison
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    
    # Top row: Standard CAM
    axes[0, 0].imshow(img_np)
    axes[0, 0].set_title(f'Original\nTrue: {CIFAR10_CLASSES[true_class]}')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(cam_colored)
    axes[0, 1].set_title('Standard CAM')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(cam_overlay)
    axes[0, 2].set_title(f'CAM Overlay\nPred: {CIFAR10_CLASSES[pred_class]}')
    axes[0, 2].axis('off')
    
    # Bottom row: Grad-CAM
    axes[1, 0].imshow(img_np)
    axes[1, 0].set_title(f'Original\nConfidence: {confidence:.2%}')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(grad_cam_colored)
    axes[1, 1].set_title('Grad-CAM')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(grad_cam_overlay)
    axes[1, 2].set_title('Grad-CAM Overlay')
    axes[1, 2].axis('off')
    
    plt.suptitle('CAM vs Grad-CAM Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    
    plt.show()
    plt.close()


def analyze_multiple_images(model, device, num_images=10, 
                            save_dir='outputs/cam'):
    """
    Generate CAM visualizations for multiple images.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Load test dataset
    transform = get_transform()
    test_dataset = datasets.CIFAR10(root='./data', train=False, 
                                     download=True, transform=transform)
    
    # Select one image from each class
    class_indices = {i: [] for i in range(10)}
    for idx, (_, label) in enumerate(test_dataset):
        class_indices[label].append(idx)
    
    for class_idx in range(min(num_images, 10)):
        if class_indices[class_idx]:
            img_idx = class_indices[class_idx][0]
            image, label = test_dataset[img_idx]
            image_tensor = image.unsqueeze(0).to(device)
            
            print(f"\n{'='*50}")
            print(f"Analyzing {CIFAR10_CLASSES[label]} (class {label})")
            print('='*50)
            
            # Forward pass
            model.eval()
            with torch.no_grad():
                output = model(image_tensor)
                probs = F.softmax(output, dim=1)
                pred_class = output.argmax(dim=1).item()
                confidence = probs[0, pred_class].item()
            
            # Generate CAM
            cam_gen = CAMGenerator(model)
            cam = cam_gen.generate_cam(pred_class)
            cam_gen.remove_hooks()
            
            # Visualize
            save_path = os.path.join(save_dir, f'cam_{CIFAR10_CLASSES[label]}.png')
            visualize_cam(
                image, cam,
                CIFAR10_CLASSES[label],
                CIFAR10_CLASSES[pred_class],
                confidence,
                save_path
            )


def visualize_multi_class_cam(model, image, true_class, device, 
                               save_path='outputs/cam/multi_class_cam.png'):
    """
    Show CAM for multiple classes on the same image.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    model.eval()
    image = image.to(device)
    if image.dim() == 3:
        image = image.unsqueeze(0)
    
    # Forward pass
    with torch.no_grad():
        output = model(image)
        probs = F.softmax(output, dim=1)[0]
    
    # Get top 5 classes
    top5_probs, top5_classes = probs.topk(5)
    
    # Generate CAM for each top class
    cam_gen = CAMGenerator(model)
    
    fig, axes = plt.subplots(1, 6, figsize=(18, 3))
    
    # Original image
    img_np = denormalize(image.squeeze(0)).permute(1, 2, 0).cpu().numpy()
    img_np = np.clip(img_np, 0, 1)
    axes[0].imshow(img_np)
    axes[0].set_title(f'Original\nTrue: {CIFAR10_CLASSES[true_class]}')
    axes[0].axis('off')
    
    # CAM for each top class
    for i, (class_idx, prob) in enumerate(zip(top5_classes, top5_probs)):
        cam = cam_gen.generate_cam(class_idx.item())
        cam_upsampled = upsample_cam(cam, (32, 32))
        cam_colored = apply_colormap(cam_upsampled)
        overlay = overlay_cam(img_np, cam_colored, alpha=0.4)
        
        axes[i + 1].imshow(overlay)
        axes[i + 1].set_title(f'{CIFAR10_CLASSES[class_idx.item()]}\n{prob.item():.2%}')
        axes[i + 1].axis('off')
    
    cam_gen.remove_hooks()
    
    plt.suptitle('CAM for Top-5 Predicted Classes', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Multi-class CAM saved to {save_path}")
    plt.show()
    plt.close()


def main():
    """Main function to run CAM visualization."""
    print("="*60)
    print("  CLASS ACTIVATION MAPS (CAM)")
    print("="*60)
    print("""
    This method highlights which regions of an image were
    most important for the model's classification decision
    by using the weights from the final fully connected layer.
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
    print("\n🔍 Generating CAM for sample images...")
    analyze_multiple_images(model, device, num_images=5)
    
    # Compare CAM vs Grad-CAM
    print("\n📊 Comparing CAM methods...")
    transform = get_transform()
    test_dataset = datasets.CIFAR10(root='./data', train=False, 
                                     download=True, transform=transform)
    image, label = test_dataset[0]
    compare_cam_methods(model, image, label, device, 
                       save_path='outputs/cam/cam_vs_gradcam.png')
    
    # Multi-class CAM
    print("\n🎯 Visualizing CAM for top predicted classes...")
    image, label = test_dataset[100]
    visualize_multi_class_cam(model, image, label, device)
    
    print("\n✅ Class Activation Maps visualization complete!")
    print("Check the 'outputs/cam/' directory for results.")


if __name__ == '__main__':
    main()
