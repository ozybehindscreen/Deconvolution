"""
CNN Model Architecture for CIFAR-10 Classification

This model is designed to support all four visualization methods:
- Uses standard convolution layers for occlusion sensitivity
- Allows gradient computation for gradient ascent
- Has a GAP layer for Class Activation Maps (CAM)
- Provides access to intermediate feature maps
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CIFAR10CNN(nn.Module):
    """
    A CNN architecture designed for CIFAR-10 that supports CAM visualization.
    
    Architecture:
    - 4 Convolutional blocks with BatchNorm and ReLU
    - Global Average Pooling (required for CAM)
    - Single fully connected layer for classification
    
    The architecture follows the CAM paper requirements:
    - GAP after last conv layer
    - Single FC layer connecting GAP output to class scores
    """
    
    def __init__(self, num_classes=10):
        super(CIFAR10CNN, self).__init__()
        
        self.num_classes = num_classes
        
        # Convolutional blocks
        # Block 1: 3 -> 64 channels
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        
        # Block 2: 64 -> 128 channels
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        
        # Block 3: 128 -> 256 channels
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        
        # Block 4: 256 -> 512 channels
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        
        # Global Average Pooling - crucial for CAM
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # Single FC layer (required for standard CAM)
        self.fc = nn.Linear(512, num_classes)
        
        # For storing intermediate feature maps
        self.feature_maps = {}
        
    def forward(self, x):
        # Block 1
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)  # 32x32 -> 16x16
        self.feature_maps['conv1'] = x
        
        # Block 2
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)  # 16x16 -> 8x8
        self.feature_maps['conv2'] = x
        
        # Block 3
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, 2)  # 8x8 -> 4x4
        self.feature_maps['conv3'] = x
        
        # Block 4 (last conv layer - used for CAM)
        x = F.relu(self.bn4(self.conv4(x)))  # Still 4x4
        self.feature_maps['conv4'] = x  # This is used for CAM
        
        # Global Average Pooling
        x = self.gap(x)  # 4x4 -> 1x1
        x = x.view(x.size(0), -1)  # Flatten: [batch, 512]
        
        # Classification
        logits = self.fc(x)
        
        return logits
    
    def get_cam_weights(self):
        """
        Get the weights from the final FC layer.
        These weights are used to compute Class Activation Maps.
        
        Returns:
            weights: [num_classes, 512] - weights connecting GAP output to each class
        """
        return self.fc.weight.data
    
    def get_last_conv_features(self):
        """
        Get the feature maps from the last convolutional layer.
        Used for generating CAM.
        
        Returns:
            feature_maps: [batch, 512, 4, 4]
        """
        return self.feature_maps.get('conv4', None)
    
    def get_feature_maps(self, layer_name):
        """
        Get feature maps from a specific layer.
        
        Args:
            layer_name: One of 'conv1', 'conv2', 'conv3', 'conv4'
            
        Returns:
            feature_maps: Tensor of feature maps
        """
        return self.feature_maps.get(layer_name, None)


class CAMModel(nn.Module):
    """
    Wrapper model that makes CAM computation easier.
    Registers hooks to capture feature maps during forward pass.
    """
    
    def __init__(self, base_model):
        super(CAMModel, self).__init__()
        self.base_model = base_model
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward and backward hooks on the last conv layer."""
        
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()
        
        # Register on conv4 (last conv layer)
        self.base_model.conv4.register_forward_hook(forward_hook)
        self.base_model.conv4.register_full_backward_hook(backward_hook)
    
    def forward(self, x):
        return self.base_model(x)
    
    def get_cam(self, class_idx):
        """
        Generate Class Activation Map for a specific class.
        
        Args:
            class_idx: Index of the class to generate CAM for
            
        Returns:
            cam: [H, W] normalized CAM heatmap
        """
        # Get weights for the target class
        weights = self.base_model.fc.weight[class_idx]  # [512]
        
        # Get the last conv feature maps
        features = self.activations  # [1, 512, H, W]
        
        # Compute weighted combination
        cam = torch.zeros(features.shape[2:], device=features.device)
        for i, w in enumerate(weights):
            cam += w * features[0, i, :, :]
        
        # ReLU to keep only positive contributions
        cam = F.relu(cam)
        
        # Normalize
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        
        return cam


def get_model(pretrained_path=None, device='cpu'):
    """
    Factory function to get the model.
    
    Args:
        pretrained_path: Path to pretrained weights
        device: Device to load model on
        
    Returns:
        model: CIFAR10CNN model
    """
    model = CIFAR10CNN(num_classes=10)
    
    if pretrained_path:
        model.load_state_dict(torch.load(pretrained_path, map_location=device))
    
    return model.to(device)


# CIFAR-10 class names for reference
CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]


if __name__ == '__main__':
    # Test the model
    model = CIFAR10CNN()
    x = torch.randn(1, 3, 32, 32)
    
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Last conv features shape: {model.get_last_conv_features().shape}")
    print(f"CAM weights shape: {model.get_cam_weights().shape}")
