# CNN Visualization & Deconvolution Methods Tutorial

This project demonstrates four key techniques for understanding what Convolutional Neural Networks (CNNs) learn and how they make decisions.

## 🎯 Methods Covered

### 1. **Occlusion Sensitivity**
Systematically occlude parts of the input image and observe how the classification probability changes. Regions where occlusion causes the largest drop in confidence are the most important for classification.

### 2. **Gradient Ascent (Class Model Visualization)**
Start from random noise and iteratively modify the image to maximize a specific class score. This reveals what patterns the network has learned to recognize for each class.

### 3. **Class Activation Maps (CAM)**
Use the spatial information preserved in the last convolutional layer to generate a heatmap highlighting which regions of the image were important for classification.

### 4. **Search Dataset Images Maximizing Activation**
Find images from the dataset that maximally activate specific neurons or feature maps, revealing what features the network has learned to detect.

## 📦 Installation

```bash
pip install -r requirements.txt
```

## 🚀 Usage

### Train the Model
```bash
python train_cnn.py
```

### Run All Visualizations
```bash
python visualize_all.py
```

### Individual Methods
```bash
python occlusion_sensitivity.py
python gradient_ascent.py
python class_activation_maps.py
python max_activation_search.py
```

## 📁 Project Structure

```
Deconvolution/
├── README.md
├── requirements.txt
├── model/
│   └── cnn_model.py          # CNN architecture
├── train_cnn.py              # Training script
├── occlusion_sensitivity.py  # Method 1
├── gradient_ascent.py        # Method 2
├── class_activation_maps.py  # Method 3
├── max_activation_search.py  # Method 4
├── visualize_all.py          # Run all methods
└── outputs/                  # Generated visualizations
```

## 📊 Dataset

Uses CIFAR-10 dataset (automatically downloaded):
- 10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
- 60,000 32x32 color images
- Perfect for demonstrating visualization techniques

## 📚 References

1. Zeiler & Fergus (2014) - "Visualizing and Understanding Convolutional Networks"
2. Simonyan et al. (2014) - "Deep Inside Convolutional Networks"
3. Zhou et al. (2016) - "Learning Deep Features for Discriminative Localization"
