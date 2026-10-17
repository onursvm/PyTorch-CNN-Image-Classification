# 🍰 Dessert Image Classification Using PyTorch CNN

A deep learning project that classifies dessert images using a custom **Convolutional Neural Network (CNN)** developed with PyTorch.

The project covers the complete image classification pipeline, including dataset exploration, preprocessing, data augmentation, CNN architecture design, model training, performance evaluation, and predictions on unseen images.

## 🎯 Project Overview

The goal of this project is to build and train a CNN model capable of recognizing four different dessert categories:

- **Baklava**
- **Cannoli**
- **Cupcakes**
- **Donuts**

The CNN architecture is implemented from scratch using PyTorch, without relying on a pretrained model.

## 🛠️ Technologies Used

- Python
- PyTorch
- Torchvision
- NumPy
- Matplotlib
- Pillow (PIL)
- Torchinfo
- Jupyter Notebook

## 📂 Dataset

The dataset is organized into training and test directories using `torchvision.datasets.ImageFolder`.

| Property | Value |
|---|---|
| Training Images | 316 |
| Test Images | 77 |
| Number of Classes | 4 |
| Image Dimensions | 64 × 64 |
| Image Channels | 3 (RGB) |
| Batch Size | 32 |

### Dataset Structure

```text
data/
└── desert101/
    ├── train/
    │   ├── baklava/
    │   ├── cannoli/
    │   ├── cup_cakes/
    │   └── donuts/
    └── test/
        ├── baklava/
        ├── cannoli/
        ├── cup_cakes/
        └── donuts/
```

## 🔄 Data Preprocessing and Augmentation

To improve model generalization, several transformations are applied to the training images.

### Training Transformations

- Resize images to 64 × 64 pixels
- Random horizontal flipping (50% probability)
- Random rotation (±15 degrees)
- TrivialAugmentWide
- Convert images to tensors
- Add Gaussian noise (mean = 0, std = 0.05)
- Normalize RGB channels

### Test Transformations

- Resize images to 64 × 64 pixels
- Convert images to tensors
- Normalize RGB channels

The normalization statistics are calculated directly from the training dataset.

```python
mean = [0.5483, 0.4638, 0.3865]
std = [0.2542, 0.2576, 0.2630]
```

## 🧠 CNN Model Architecture

A custom CNN named `DesertClassifier` is implemented using `torch.nn.Module`.

The architecture consists of two convolutional blocks followed by a fully connected classification layer.

### Architecture Details

**Convolutional Block 1**
- Conv2D: 3 → 32 channels, 3×3 kernel
- ReLU activation
- Conv2D: 32 → 32 channels, 3×3 kernel
- ReLU activation
- MaxPooling: 2×2

**Convolutional Block 2**
- Conv2D: 32 → 32 channels, 3×3 kernel
- ReLU activation
- Conv2D: 32 → 32 channels, 3×3 kernel
- ReLU activation
- MaxPooling: 2×2

**Classification Layer**
- Flatten
- Linear: 8192 → 4 output classes

### Model Flow

```text
Input Image (3 × 64 × 64)
          ↓
    Conv Block 1
          ↓
      MaxPool
          ↓
    Conv Block 2
          ↓
      MaxPool
          ↓
        Flatten
          ↓
    Fully Connected
          ↓
    4 Class Logits
```

## ⚙️ Training Configuration

| Hyperparameter | Value |
|---|---|
| Optimizer | Adam |
| Learning Rate | 0.001 |
| Loss Function | CrossEntropyLoss |
| Epochs | 10 |
| Batch Size | 32 |
| Hidden Channels | 32 |

The model is trained using a custom training loop that includes forward propagation, loss calculation, backpropagation, and optimizer updates.

Evaluation is performed using `model.eval()` and `torch.inference_mode()`.

## 📊 Model Performance

The following results were recorded in the notebook after 10 training epochs.

| Metric | Result |
|---|---|
| Final Training Accuracy | 42.41% |
| Final Test Accuracy | 54.41% |
| Final Training Loss | 1.2729 |
| Final Test Loss | 1.1915 |

*Accuracy values are based on the notebook's batch-averaged accuracy calculations.*

Training and test loss and accuracy curves are visualized using Matplotlib to monitor learning progress.

The model is an experimental implementation, and classification performance can be further improved.

## 🔍 Model Predictions

The project includes functionality to:

- Generate predictions using a trained CNN.
- Convert model logits into probabilities using Softmax.
- Display random test images with predicted and actual labels.
- Highlight correct predictions in green and incorrect predictions in red.
- Load and classify an external dessert image.

## 🚀 Future Improvements

- Experiment with additional CNN architectures.
- Tune data augmentation parameters.
- Improve model accuracy through hyperparameter optimization.
- Evaluate transfer learning using pretrained CNN models.
- Increase dataset size to improve generalization.
- Introduce a validation dataset for model selection.

## 📌 Learning Outcomes

This project provides hands-on experience with:

- Convolutional Neural Networks (CNNs)
- Image classification with PyTorch
- Custom neural network architectures
- Image preprocessing and augmentation
- Dataset normalization
- Training and evaluation loops
- Model predictions and visualization

## 📜 Conclusion

This project demonstrates an end-to-end computer vision workflow for classifying dessert images using a custom PyTorch CNN. It serves as a practical foundation for further exploration of deep learning, image recognition, and model optimization.
