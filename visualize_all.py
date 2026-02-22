"""
Unified Visualization Script

This script runs all four deconvolution/visualization methods
and generates a comprehensive report with all results.
"""

import os
import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import numpy as np

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


def print_header(title):
    """Print a formatted header."""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)


def check_model_exists():
    """Check if trained model exists."""
    if not os.path.exists('checkpoints/best_model.pth'):
        print("❌ No trained model found!")
        print("Please run: python train_cnn.py")
        return False
    return True


def create_summary_report(output_dir='outputs'):
    """
    Create a summary report of all visualizations generated.
    """
    summary_path = os.path.join(output_dir, 'SUMMARY.md')
    
    with open(summary_path, 'w') as f:
        f.write("# CNN Visualization Methods - Summary Report\n\n")
        f.write("This report summarizes all the visualization methods applied to understand\n")
        f.write("the CNN model trained on CIFAR-10 dataset.\n\n")
        
        f.write("## 📊 Methods Overview\n\n")
        
        f.write("### 1. Occlusion Sensitivity\n")
        f.write("- **Purpose**: Identify which regions of an image are most important for classification\n")
        f.write("- **Method**: Systematically cover parts of the image and observe prediction changes\n")
        f.write("- **Key Insight**: Regions where occlusion causes the largest confidence drop are most important\n")
        f.write("- **Output**: `outputs/occlusion/`\n\n")
        
        f.write("### 2. Gradient Ascent (Class Model Visualization)\n")
        f.write("- **Purpose**: Visualize what patterns the network associates with each class\n")
        f.write("- **Method**: Generate synthetic images that maximize class activation scores\n")
        f.write("- **Key Insight**: Reveals learned class prototypes and texture patterns\n")
        f.write("- **Output**: `outputs/gradient_ascent/`\n\n")
        
        f.write("### 3. Class Activation Maps (CAM)\n")
        f.write("- **Purpose**: Highlight discriminative regions used for classification\n")
        f.write("- **Method**: Weight last conv layer features by classification layer weights\n")
        f.write("- **Key Insight**: Shows spatial attention of the network\n")
        f.write("- **Output**: `outputs/cam/`\n\n")
        
        f.write("### 4. Maximum Activation Search\n")
        f.write("- **Purpose**: Understand what features each neuron has learned to detect\n")
        f.write("- **Method**: Find dataset images that maximally activate specific filters\n")
        f.write("- **Key Insight**: Reveals feature detectors learned at each layer\n")
        f.write("- **Output**: `outputs/max_activation/`\n\n")
        
        f.write("## 📁 Generated Files\n\n")
        
        # List all generated files
        for subdir in ['occlusion', 'gradient_ascent', 'cam', 'max_activation']:
            subdir_path = os.path.join(output_dir, subdir)
            if os.path.exists(subdir_path):
                f.write(f"### {subdir}/\n")
                for file in os.listdir(subdir_path):
                    f.write(f"- `{file}`\n")
                f.write("\n")
        
        f.write("## 🔗 References\n\n")
        f.write("1. Zeiler & Fergus (2014) - \"Visualizing and Understanding Convolutional Networks\"\n")
        f.write("2. Simonyan et al. (2014) - \"Deep Inside Convolutional Networks\"\n")
        f.write("3. Zhou et al. (2016) - \"Learning Deep Features for Discriminative Localization\"\n")
    
    print(f"Summary report created: {summary_path}")


def run_all_visualizations():
    """
    Run all four visualization methods.
    """
    print_header("CNN DECONVOLUTION & VISUALIZATION METHODS")
    print("""
    This script will run all four visualization methods:
    
    1. 🔍 Occlusion Sensitivity
    2. 🎨 Gradient Ascent (Class Model Visualization)
    3. 🗺️  Class Activation Maps (CAM)
    4. 🔎 Maximum Activation Search
    
    Make sure you have trained the model first!
    """)
    
    # Check model
    if not check_model_exists():
        return
    
    print("\n" + "="*70)
    print("  RUNNING ALL VISUALIZATION METHODS")
    print("="*70)
    
    # Create output directories
    os.makedirs('outputs', exist_ok=True)
    
    # Method 1: Occlusion Sensitivity
    print_header("METHOD 1: OCCLUSION SENSITIVITY")
    try:
        from occlusion_sensitivity import main as occlusion_main
        occlusion_main()
    except Exception as e:
        print(f"Error in occlusion sensitivity: {e}")
    
    # Method 2: Gradient Ascent
    print_header("METHOD 2: GRADIENT ASCENT")
    try:
        from gradient_ascent import main as gradient_main
        gradient_main()
    except Exception as e:
        print(f"Error in gradient ascent: {e}")
    
    # Method 3: Class Activation Maps
    print_header("METHOD 3: CLASS ACTIVATION MAPS")
    try:
        from class_activation_maps import main as cam_main
        cam_main()
    except Exception as e:
        print(f"Error in class activation maps: {e}")
    
    # Method 4: Maximum Activation Search
    print_header("METHOD 4: MAXIMUM ACTIVATION SEARCH")
    try:
        from max_activation_search import main as max_act_main
        max_act_main()
    except Exception as e:
        print(f"Error in max activation search: {e}")
    
    # Create summary report
    print_header("CREATING SUMMARY REPORT")
    create_summary_report()
    
    # Final summary
    print_header("ALL VISUALIZATIONS COMPLETE")
    print("""
    ✅ All four visualization methods have been executed!
    
    📁 Check the 'outputs/' directory for all generated visualizations:
    
        outputs/
        ├── training_history.png    # Training curves
        ├── SUMMARY.md              # Summary report
        ├── occlusion/              # Occlusion sensitivity maps
        ├── gradient_ascent/        # Class model visualizations
        ├── cam/                    # Class activation maps
        └── max_activation/         # Maximum activation analysis
    
    🎓 Learning Points:
    
    1. OCCLUSION SENSITIVITY shows which image regions the model relies on
       by measuring prediction changes when regions are hidden.
    
    2. GRADIENT ASCENT reveals the "ideal" patterns for each class by
       generating images that maximize class scores.
    
    3. CLASS ACTIVATION MAPS provide a spatial heatmap of important regions
       using the network's internal feature weights.
    
    4. MAXIMUM ACTIVATION SEARCH finds real images that trigger specific
       neurons, revealing learned feature detectors.
    
    Together, these methods provide comprehensive insight into HOW and WHY
    the CNN makes its classification decisions!
    """)


def quick_demo(image_idx=42):
    """
    Run a quick demo on a single image with all methods.
    """
    print_header("QUICK DEMO - ALL METHODS ON ONE IMAGE")
    
    if not check_model_exists():
        return
    
    # Load model and image
    model, device = load_model()
    
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    test_dataset = datasets.CIFAR10(root='./data', train=False, 
                                     download=True, transform=transform)
    
    image, label = test_dataset[image_idx]
    image_tensor = image.unsqueeze(0).to(device)
    
    print(f"Demo image: {CIFAR10_CLASSES[label]} (class {label})")
    
    # Quick visualization of all methods
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    
    # Original image
    def denorm(t):
        mean_t = torch.tensor(mean).view(3, 1, 1)
        std_t = torch.tensor(std).view(3, 1, 1)
        return t * std_t + mean_t
    
    img_np = denorm(image).permute(1, 2, 0).numpy()
    img_np = np.clip(img_np, 0, 1)
    
    axes[0, 0].imshow(img_np)
    axes[0, 0].set_title(f'Original Image\nClass: {CIFAR10_CLASSES[label]}')
    axes[0, 0].axis('off')
    
    # Model prediction
    model.eval()
    with torch.no_grad():
        output = model(image_tensor)
        probs = torch.softmax(output, dim=1)
        pred = output.argmax(dim=1).item()
        conf = probs[0, pred].item()
    
    # 1. Simple occlusion sensitivity (quick version)
    from occlusion_sensitivity import occlusion_sensitivity
    sensitivity_map, _ = occlusion_sensitivity(
        model, image_tensor, label, device,
        patch_size=6, stride=2
    )
    axes[0, 1].imshow(sensitivity_map, cmap='hot')
    axes[0, 1].set_title('Occlusion Sensitivity')
    axes[0, 1].axis('off')
    
    # 2. CAM
    from class_activation_maps import CAMGenerator, upsample_cam, apply_colormap, overlay_cam
    cam_gen = CAMGenerator(model)
    _ = model(image_tensor)
    cam = cam_gen.generate_cam(pred)
    cam_up = upsample_cam(cam, (32, 32))
    cam_colored = apply_colormap(cam_up)
    overlay = overlay_cam(img_np, cam_colored, alpha=0.4)
    cam_gen.remove_hooks()
    
    axes[0, 2].imshow(overlay)
    axes[0, 2].set_title(f'CAM\nPred: {CIFAR10_CLASSES[pred]}')
    axes[0, 2].axis('off')
    
    # 3. Gradient Ascent result (load if exists)
    ga_path = f'outputs/gradient_ascent/class_{pred}_{CIFAR10_CLASSES[pred]}.png'
    if os.path.exists(ga_path):
        from PIL import Image
        ga_img = Image.open(ga_path)
        axes[1, 0].imshow(ga_img)
    else:
        axes[1, 0].text(0.5, 0.5, 'Run gradient_ascent.py\nfirst', 
                        ha='center', va='center', fontsize=12)
    axes[1, 0].set_title(f'Class Prototype\n({CIFAR10_CLASSES[pred]})')
    axes[1, 0].axis('off')
    
    # 4. Feature maps
    feature_maps = model.get_last_conv_features()
    if feature_maps is not None:
        avg_features = feature_maps[0].mean(dim=0).cpu().numpy()
        axes[1, 1].imshow(avg_features, cmap='viridis')
        axes[1, 1].set_title('Last Conv Features\n(Avg. across channels)')
    axes[1, 1].axis('off')
    
    # 5. Prediction probabilities
    top5_probs, top5_indices = probs[0].topk(5)
    colors = ['green' if i == label else 'blue' for i in top5_indices.cpu().numpy()]
    axes[1, 2].barh(range(5), top5_probs.cpu().numpy(), color=colors)
    axes[1, 2].set_yticks(range(5))
    axes[1, 2].set_yticklabels([CIFAR10_CLASSES[i] for i in top5_indices.cpu().numpy()])
    axes[1, 2].set_xlabel('Probability')
    axes[1, 2].set_title(f'Top-5 Predictions\nCorrect: {pred == label}')
    axes[1, 2].set_xlim(0, 1)
    
    plt.suptitle('CNN Visualization Methods - Quick Demo', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    os.makedirs('outputs', exist_ok=True)
    plt.savefig('outputs/quick_demo.png', dpi=150, bbox_inches='tight')
    print("Demo saved to outputs/quick_demo.png")
    plt.show()
    plt.close()


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--quick':
        # Quick demo
        quick_demo()
    else:
        # Full visualization
        run_all_visualizations()
