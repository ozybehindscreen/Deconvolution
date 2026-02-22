"""
Method 4: Search Dataset Images Maximizing Activation

This technique finds images from the dataset that maximally activate 
specific neurons or feature maps in the network. This helps understand 
what features each neuron has learned to detect.

How it works:
1. Select a target layer and specific neuron/feature map
2. Pass all images through the network
3. Record activation values for the target neuron
4. Find and display images with the highest activations
5. Optionally, visualize receptive fields

This is a form of "network introspection" that reveals what 
patterns trigger strong responses in different parts of the network.
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from tqdm import tqdm
from collections import defaultdict

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
    mean = torch.tensor(mean).view(3, 1, 1).to(tensor.device)
    std = torch.tensor(std).view(3, 1, 1).to(tensor.device)
    return tensor * std + mean


class ActivationExtractor:
    """
    Extract activations from specific layers during forward pass.
    """
    
    def __init__(self, model):
        self.model = model
        self.activations = {}
        self.hooks = []
        
    def register_hooks(self, layer_names):
        """
        Register forward hooks on specified layers.
        
        Args:
            layer_names: List of layer names ('conv1', 'conv2', 'conv3', 'conv4')
        """
        layer_dict = {
            'conv1': self.model.conv1,
            'conv2': self.model.conv2,
            'conv3': self.model.conv3,
            'conv4': self.model.conv4,
        }
        
        for name in layer_names:
            if name in layer_dict:
                hook = layer_dict[name].register_forward_hook(
                    self._get_activation_hook(name)
                )
                self.hooks.append(hook)
    
    def _get_activation_hook(self, name):
        """Create a hook function that saves activations."""
        def hook(module, input, output):
            self.activations[name] = output.detach()
        return hook
    
    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def clear_activations(self):
        """Clear stored activations."""
        self.activations = {}


def find_max_activating_images(model, dataloader, layer_name, 
                                filter_idx, device, top_k=9):
    """
    Find images that maximally activate a specific filter.
    
    Args:
        model: The CNN model
        dataloader: DataLoader with images
        layer_name: Target layer name
        filter_idx: Index of the filter to analyze
        device: Computation device
        top_k: Number of top activating images to return
        
    Returns:
        top_images: List of (image, activation, label, spatial_max) tuples
    """
    model.eval()
    extractor = ActivationExtractor(model)
    extractor.register_hooks([layer_name])
    
    # Store results: (max_activation, image, label, spatial_location)
    results = []
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc=f'Searching {layer_name} filter {filter_idx}'):
            images = images.to(device)
            
            # Forward pass
            _ = model(images)
            
            # Get activations for target layer
            activations = extractor.activations[layer_name]  # [B, C, H, W]
            
            # Get activations for target filter
            filter_activations = activations[:, filter_idx]  # [B, H, W]
            
            # Find max activation for each image
            for i in range(images.size(0)):
                act_map = filter_activations[i]  # [H, W]
                max_val = act_map.max().item()
                max_loc = torch.where(act_map == act_map.max())
                spatial_loc = (max_loc[0][0].item(), max_loc[1][0].item())
                
                results.append({
                    'activation': max_val,
                    'image': images[i].cpu(),
                    'label': labels[i].item(),
                    'spatial_loc': spatial_loc,
                    'activation_map': act_map.cpu().numpy()
                })
            
            extractor.clear_activations()
    
    extractor.remove_hooks()
    
    # Sort by activation and get top_k
    results.sort(key=lambda x: x['activation'], reverse=True)
    return results[:top_k]


def visualize_max_activating_images(results, layer_name, filter_idx, 
                                    save_path=None):
    """
    Visualize the top activating images for a filter.
    """
    n = len(results)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols * 2, figsize=(cols * 5, rows * 2.5))
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    for i, result in enumerate(results):
        row = i // cols
        col = (i % cols) * 2
        
        # Original image
        img = denormalize(result['image']).permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1)
        axes[row, col].imshow(img)
        axes[row, col].set_title(
            f'{CIFAR10_CLASSES[result["label"]]}\nAct: {result["activation"]:.2f}',
            fontsize=9
        )
        axes[row, col].axis('off')
        
        # Activation map
        axes[row, col + 1].imshow(result['activation_map'], cmap='hot')
        axes[row, col + 1].set_title(f'Activation Map\nMax: {result["spatial_loc"]}', fontsize=9)
        axes[row, col + 1].axis('off')
    
    # Hide empty subplots
    for i in range(n, rows * cols):
        row = i // cols
        col = (i % cols) * 2
        axes[row, col].axis('off')
        axes[row, col + 1].axis('off')
    
    plt.suptitle(f'Top Activating Images for {layer_name} Filter {filter_idx}', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    
    plt.show()
    plt.close()


def analyze_layer_filters(model, dataloader, layer_name, device, 
                          num_filters=16, top_k=4,
                          save_dir='outputs/max_activation'):
    """
    Analyze multiple filters in a layer.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Get number of filters in the layer
    layer_dict = {
        'conv1': model.conv1,
        'conv2': model.conv2,
        'conv3': model.conv3,
        'conv4': model.conv4,
    }
    
    total_filters = layer_dict[layer_name].weight.size(0)
    filter_indices = np.random.choice(total_filters, min(num_filters, total_filters), replace=False)
    
    print(f"\n{'='*60}")
    print(f"  Analyzing {layer_name} ({total_filters} filters)")
    print('='*60)
    
    all_results = {}
    
    for filter_idx in filter_indices:
        print(f"\nFilter {filter_idx}:")
        results = find_max_activating_images(
            model, dataloader, layer_name, 
            filter_idx, device, top_k
        )
        all_results[filter_idx] = results
        
        # Visualize
        save_path = os.path.join(save_dir, f'{layer_name}_filter{filter_idx}.png')
        visualize_max_activating_images(results, layer_name, filter_idx, save_path)
    
    return all_results


def find_class_specific_filters(model, dataloader, layer_name, device,
                                save_path='outputs/max_activation/class_filters.png'):
    """
    Find filters that are most responsive to specific classes.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    model.eval()
    extractor = ActivationExtractor(model)
    extractor.register_hooks([layer_name])
    
    # Accumulate activations per class
    layer_dict = {
        'conv1': model.conv1,
        'conv2': model.conv2,
        'conv3': model.conv3,
        'conv4': model.conv4,
    }
    num_filters = layer_dict[layer_name].weight.size(0)
    
    # [num_classes, num_filters]
    class_filter_activations = torch.zeros(10, num_filters)
    class_counts = torch.zeros(10)
    
    print(f"\nAnalyzing class-specific filter activations in {layer_name}...")
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc='Processing'):
            images = images.to(device)
            _ = model(images)
            
            activations = extractor.activations[layer_name]  # [B, C, H, W]
            # Global average pooling to get single value per filter
            gap_activations = activations.mean(dim=(2, 3))  # [B, C]
            
            for i in range(images.size(0)):
                label = labels[i].item()
                class_filter_activations[label] += gap_activations[i].cpu()
                class_counts[label] += 1
            
            extractor.clear_activations()
    
    extractor.remove_hooks()
    
    # Average activations
    for c in range(10):
        if class_counts[c] > 0:
            class_filter_activations[c] /= class_counts[c]
    
    # Find top filters for each class
    top_filters_per_class = {}
    for class_idx in range(10):
        activations = class_filter_activations[class_idx]
        top_indices = activations.argsort(descending=True)[:5]
        top_filters_per_class[class_idx] = top_indices.tolist()
    
    # Visualize
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    
    for class_idx, ax in enumerate(axes.flatten()):
        top_filters = top_filters_per_class[class_idx]
        activations = class_filter_activations[class_idx, top_filters]
        
        bars = ax.bar(range(5), activations.numpy(), color='steelblue')
        ax.set_title(CIFAR10_CLASSES[class_idx], fontsize=10, fontweight='bold')
        ax.set_xticks(range(5))
        ax.set_xticklabels([f'F{f}' for f in top_filters], fontsize=8)
        ax.set_ylim(0, class_filter_activations.max().item() * 1.1)
        
        if class_idx % 5 == 0:
            ax.set_ylabel('Mean Activation')
    
    plt.suptitle(f'Top-5 Most Responsive Filters per Class ({layer_name})', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Class-specific filters saved to {save_path}")
    plt.show()
    plt.close()
    
    return top_filters_per_class, class_filter_activations


def visualize_filter_weights(model, layer_name, num_filters=32,
                             save_path='outputs/max_activation/filter_weights.png'):
    """
    Visualize the learned filter weights.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    layer_dict = {
        'conv1': model.conv1,
        'conv2': model.conv2,
        'conv3': model.conv3,
        'conv4': model.conv4,
    }
    
    weights = layer_dict[layer_name].weight.data.cpu()  # [out_channels, in_channels, H, W]
    out_channels, in_channels, H, W = weights.shape
    
    # For conv1, we can visualize the filters directly (3 input channels = RGB)
    if layer_name == 'conv1':
        # Normalize weights for visualization
        weights = weights - weights.min()
        weights = weights / weights.max()
        
        num_to_show = min(num_filters, out_channels)
        cols = 8
        rows = (num_to_show + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.2, rows * 1.2))
        axes = axes.flatten()
        
        for i in range(num_to_show):
            # Transpose to [H, W, C] for imshow
            filter_img = weights[i].permute(1, 2, 0).numpy()
            axes[i].imshow(filter_img)
            axes[i].axis('off')
            axes[i].set_title(f'{i}', fontsize=8)
        
        for i in range(num_to_show, len(axes)):
            axes[i].axis('off')
        
        plt.suptitle(f'{layer_name} Learned Filters (RGB)', fontsize=14, fontweight='bold')
    else:
        # For deeper layers, show the filter response patterns
        # Use PCA or just show a subset of channels
        num_to_show = min(num_filters, out_channels)
        cols = 8
        rows = (num_to_show + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.2, rows * 1.2))
        axes = axes.flatten()
        
        for i in range(num_to_show):
            # Average across input channels
            filter_avg = weights[i].mean(dim=0).numpy()
            filter_avg = (filter_avg - filter_avg.min()) / (filter_avg.max() - filter_avg.min() + 1e-8)
            axes[i].imshow(filter_avg, cmap='viridis')
            axes[i].axis('off')
            axes[i].set_title(f'{i}', fontsize=8)
        
        for i in range(num_to_show, len(axes)):
            axes[i].axis('off')
        
        plt.suptitle(f'{layer_name} Learned Filters (Avg. across channels)', 
                     fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Filter weights saved to {save_path}")
    plt.show()
    plt.close()


def analyze_feature_evolution(model, image, label, device,
                              save_path='outputs/max_activation/feature_evolution.png'):
    """
    Visualize how features evolve through the network layers.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    model.eval()
    extractor = ActivationExtractor(model)
    extractor.register_hooks(['conv1', 'conv2', 'conv3', 'conv4'])
    
    image = image.to(device)
    if image.dim() == 3:
        image = image.unsqueeze(0)
    
    # Forward pass
    with torch.no_grad():
        output = model(image)
        pred = output.argmax(dim=1).item()
    
    # Visualize feature maps at each layer
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    
    # Original image
    img_np = denormalize(image.squeeze(0)).permute(1, 2, 0).cpu().numpy()
    img_np = np.clip(img_np, 0, 1)
    axes[0, 0].imshow(img_np)
    axes[0, 0].set_title(f'Input\n{CIFAR10_CLASSES[label]}', fontsize=10)
    axes[0, 0].axis('off')
    
    # Feature maps from each layer
    layers = ['conv1', 'conv2', 'conv3', 'conv4']
    for i, layer in enumerate(layers):
        acts = extractor.activations[layer]  # [1, C, H, W]
        
        # Show average activation map
        avg_act = acts[0].mean(dim=0).cpu().numpy()
        axes[0, i + 1].imshow(avg_act, cmap='viridis')
        axes[0, i + 1].set_title(f'{layer}\n({acts.shape[1]} filters, {acts.shape[2]}x{acts.shape[3]})', 
                                  fontsize=9)
        axes[0, i + 1].axis('off')
        
        # Show top activated filter
        max_channel = acts[0].mean(dim=(1, 2)).argmax().item()
        axes[1, i + 1].imshow(acts[0, max_channel].cpu().numpy(), cmap='hot')
        axes[1, i + 1].set_title(f'Max filter #{max_channel}', fontsize=9)
        axes[1, i + 1].axis('off')
    
    axes[1, 0].axis('off')
    
    extractor.remove_hooks()
    
    plt.suptitle(f'Feature Evolution (Predicted: {CIFAR10_CLASSES[pred]})', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Feature evolution saved to {save_path}")
    plt.show()
    plt.close()


def main():
    """Main function to run maximum activation search."""
    print("="*60)
    print("  SEARCH DATASET IMAGES MAXIMIZING ACTIVATION")
    print("="*60)
    print("""
    This method finds images that maximally activate specific
    neurons or filters in the network, revealing what patterns
    each part of the network has learned to detect.
    """)
    
    # Load model
    print("\n📦 Loading trained model...")
    try:
        model, device = load_model()
        print(f"Model loaded successfully on {device}")
    except FileNotFoundError:
        print("❌ Model not found! Please run train_cnn.py first.")
        return
    
    # Load dataset
    print("\n📦 Loading CIFAR-10 dataset...")
    transform = get_transform()
    test_dataset = datasets.CIFAR10(root='./data', train=False, 
                                     download=True, transform=transform)
    dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=128, shuffle=False, num_workers=2
    )
    
    # Visualize filter weights
    print("\n🔍 Visualizing learned filter weights...")
    visualize_filter_weights(model, 'conv1')
    visualize_filter_weights(model, 'conv4')
    
    # Feature evolution through layers
    print("\n📊 Visualizing feature evolution...")
    image, label = test_dataset[42]
    analyze_feature_evolution(model, image, label, device)
    
    # Find max activating images for specific filters
    print("\n🔎 Finding maximally activating images...")
    analyze_layer_filters(model, dataloader, 'conv1', device, 
                         num_filters=4, top_k=6)
    analyze_layer_filters(model, dataloader, 'conv4', device, 
                         num_filters=4, top_k=6)
    
    # Class-specific filters
    print("\n🎯 Finding class-specific filters...")
    find_class_specific_filters(model, dataloader, 'conv4', device)
    
    print("\n✅ Maximum activation search complete!")
    print("Check the 'outputs/max_activation/' directory for results.")


if __name__ == '__main__':
    main()
