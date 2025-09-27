# Deep Learning Assignment 1: Facial Expression Analysis
## NUCES FAST, Islamabad

**Instructors:** Mahnoor Tariq & Dr. Qurat Ul Ain  
**Due Date:** 27th September 2025

---

## 1. Network Details and Architecture

### 1.1 Model Architectures

#### 1.1.1 ResNet50 (Transfer Learning Baseline)
- **Architecture**: ResNet50 with ImageNet pre-trained weights
- **Backbone**: torchvision.models.resnet50 with weights=IMAGENET1K_V2
- **Feature Extraction**: 2048-dimensional features from final layer
- **Classification Head**: Enhanced with attention mechanism
  - Feature Processor: 2048 → 512 → 256 (with BatchNorm, ReLU, Dropout)
  - Attention Module: 256 → 128 → 256 (Sigmoid activation)
  - Classification: 256 → 128 → 8 classes
  - Valence/Arousal: 256 → 128 → 64 → 1 (with Tanh activation)
- **Total Parameters**: ~25.6M (backbone) + ~0.5M (heads) = ~26.1M
- **Rationale**: ResNet50 chosen as baseline due to its proven performance in computer vision tasks, residual connections for better gradient flow, and strong feature extraction capabilities.

#### 1.1.2 VGG16 (Transfer Learning Baseline)
- **Architecture**: VGG16 with ImageNet pre-trained weights
- **Backbone**: torchvision.models.vgg16 with weights=IMAGENET1K_V1
- **Feature Extraction**: 4096-dimensional features from final layer
- **Classification Head**: Same enhanced architecture as ResNet50
- **Total Parameters**: ~138M (backbone) + ~0.5M (heads) = ~138.5M
- **Rationale**: VGG16 chosen as second baseline for comparison due to its simple, uniform architecture and proven effectiveness in transfer learning scenarios.

#### 1.1.3 Custom CNN (From Scratch)
- **Architecture**: Custom-designed CNN for facial expression recognition
- **Feature Extraction**:
  - Block 1: Conv2d(3→64, 7x7, stride=2) + BatchNorm + ReLU + MaxPool
  - Block 2: Conv2d(64→128, 3x3) + BatchNorm + ReLU + Conv2d(128→128, 3x3) + BatchNorm + ReLU + MaxPool
  - Block 3: Conv2d(128→256, 3x3) + BatchNorm + ReLU + Conv2d(256→256, 3x3) + BatchNorm + ReLU + MaxPool
  - Block 4: Conv2d(256→512, 3x3) + BatchNorm + ReLU + Conv2d(512→512, 3x3) + BatchNorm + ReLU + AdaptiveAvgPool
- **Classification Head**: 512 → 256 → 128 → 8 classes (with BatchNorm, ReLU, Dropout)
- **Valence/Arousal Head**: 512 → 256 → 128 → 64 → 1 (with Tanh activation)
- **Total Parameters**: ~15.2M
- **Rationale**: Custom CNN designed specifically for facial expression recognition without domain gap from ImageNet pre-training.

### 1.2 Training Settings

#### 1.2.1 Hyperparameters
- **Learning Rate**: 1e-4 (reduced from 3e-4 for better convergence)
- **Batch Size**: 32 (ResNet50/VGG16), 64 (Custom CNN)
- **Epochs**: 25 (ResNet50/VGG16), 30 (Custom CNN)
- **Weight Decay**: 1e-4
- **Lambda Regularization**: 1.0 (balancing classification and regression losses)
- **Optimizer**: AdamW with betas=(0.9, 0.999), eps=1e-8
- **Scheduler**: Cosine annealing with 3-epoch warmup

#### 1.2.2 Advanced Training Techniques
- **Early Stopping**: Patience of 5 epochs
- **Mixed Precision Training**: Automatic mixed precision (AMP) for faster training
- **Data Augmentation**: Enhanced augmentation pipeline
- **Loss Functions**: Focal Loss for classification, Smooth L1 Loss for regression
- **Class Balancing**: Weighted random sampling for imbalanced classes

---

## 2. Dataset Splits

### 2.1 Dataset Overview
- **Total Images**: 3,999 face images (224x224 RGB)
- **Annotations**: 15,996 .npy files containing:
  - Expression labels (0-7): 8 emotion categories
  - Valence values [-1, +1]: Continuous emotional valence
  - Arousal values [-1, +1]: Continuous emotional arousal
  - 68 facial landmarks (136 coordinates)

### 2.2 Train-Validation Split
- **Training Set**: 3,199 images (80%)
- **Validation Set**: 800 images (20%)
- **Split Method**: Random split with seed=42 for reproducibility
- **Stratification**: Maintained class distribution across splits

### 2.3 Data Augmentation Pipeline
```python
# Training Augmentations
transforms.Compose([
    RandomResizedCrop(224, scale=(0.85, 1.0), ratio=(0.8, 1.2)),
    RandomHorizontalFlip(p=0.5),
    RandomRotation(degrees=15),
    RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    RandomGrayscale(p=0.1),
    RandomPerspective(distortion_scale=0.1, p=0.3),
    ToTensor(),
    RandomErasing(p=0.2, scale=(0.02, 0.1), ratio=(0.3, 3.3)),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```

---

## 3. Training Graphs

### 3.1 Loss Curves
The training graphs show the following patterns:

#### ResNet50 Training Progress:
- **Training Loss**: Decreased from 1.985 to 0.052 over 10 epochs
- **Classification Loss**: Decreased from 1.820 to 0.027
- **Regression Loss**: Decreased from 0.165 to 0.025
- **Validation Accuracy**: Increased from 37.9% to 49.5%
- **Validation F1**: Increased from 0.349 to 0.496

#### VGG16 Training Progress:
- **Training Loss**: Decreased from 2.278 to 1.740 over 10 epochs
- **Classification Loss**: Decreased from 2.090 to 1.595
- **Regression Loss**: Decreased from 0.188 to 0.145
- **Validation Accuracy**: Increased from 10.5% to 35.6%
- **Validation F1**: Increased from 0.059 to 0.342

#### Custom CNN Training Progress:
- **Training Loss**: Decreased from 2.339 to 1.483 over 20 epochs
- **Classification Loss**: Decreased from 2.100 to 1.200
- **Regression Loss**: Decreased from 0.239 to 0.283
- **Validation Accuracy**: Increased from 14.4% to 38.5%
- **Validation F1**: Increased from 0.080 to 0.381

### 3.2 Convergence Analysis
- **ResNet50**: Fastest convergence, best final performance
- **VGG16**: Slower convergence, moderate performance
- **Custom CNN**: Steady improvement, competitive performance

---

## 4. Performance Measures

### 4.1 Classification Metrics

| Model | Accuracy | Macro F1 | AUC | Kappa | Alpha |
|-------|----------|----------|-----|-------|-------|
| ResNet50 | 49.5% | 0.496 | 0.852 | 0.423 | 0.423 |
| VGG16 | 35.6% | 0.342 | 0.774 | 0.265 | 0.265 |
| Custom CNN | 38.5% | 0.381 | 0.720 | 0.310 | 0.310 |

### 4.2 Regression Metrics

| Model | Valence RMSE | Arousal RMSE | Valence CCC | Arousal CCC | Valence CORR | Arousal CORR |
|-------|--------------|--------------|-------------|-------------|--------------|--------------|
| ResNet50 | 0.376 | 0.347 | 0.566 | 0.438 | 0.600 | 0.470 |
| VGG16 | 0.428 | 0.364 | 0.313 | 0.186 | 0.402 | 0.310 |
| Custom CNN | 0.420 | 0.380 | 0.428 | 0.245 | 0.450 | 0.320 |

### 4.3 Continuous Domain Evaluation Metrics Discussion

#### 4.3.1 RMSE (Root Mean Square Error)
- **Rationale**: Measures the average magnitude of prediction errors
- **Interpretation**: Lower values indicate better regression performance
- **Wild Scenario Suitability**: Good for overall error assessment but sensitive to outliers

#### 4.3.2 CORR (Pearson Correlation)
- **Rationale**: Measures linear relationship between predicted and actual values
- **Interpretation**: Values close to ±1 indicate strong linear correlation
- **Wild Scenario Suitability**: Excellent for understanding prediction trends and direction

#### 4.3.3 SAGR (Sign Agreement Rate)
- **Rationale**: Measures the proportion of predictions with correct sign (positive/negative)
- **Interpretation**: Values close to 1 indicate good directional prediction
- **Wild Scenario Suitability**: Most practical for real-world applications where direction matters more than exact values

#### 4.3.4 CCC (Concordance Correlation Coefficient)
- **Rationale**: Measures agreement between predicted and actual values, combining correlation and accuracy
- **Interpretation**: Values close to 1 indicate perfect agreement
- **Wild Scenario Suitability**: Best overall metric for continuous emotion prediction as it considers both correlation and accuracy

**Recommendation for Wild System**: **SAGR** should be prioritized as it measures the most practical aspect - whether the system correctly identifies the emotional direction (positive/negative valence, high/low arousal), which is crucial for real-world emotion recognition applications.

---

## 5. Performance Comparison of CNN Architectures

### 5.1 Accuracy Comparison

| Model | Classification Accuracy | Valence CCC | Arousal CCC | Composite Score |
|-------|------------------------|-------------|-------------|-----------------|
| **ResNet50** | **49.5%** | **0.566** | **0.438** | **1.500** |
| Custom CNN | 38.5% | 0.428 | 0.245 | 1.058 |
| VGG16 | 35.6% | 0.313 | 0.186 | 0.841 |

**Winner**: ResNet50 with highest composite score of 1.500

### 5.2 Training Time Analysis

| Model | Epochs to Converge | Training Time (approx.) | Parameters |
|-------|-------------------|-------------------------|------------|
| ResNet50 | 10 | ~45 minutes | 26.1M |
| VGG16 | 10 | ~60 minutes | 138.5M |
| Custom CNN | 20 | ~35 minutes | 15.2M |

### 5.3 Efficiency Analysis
- **ResNet50**: Best accuracy, moderate training time, efficient parameter usage
- **Custom CNN**: Fastest training, competitive accuracy, most parameter-efficient
- **VGG16**: Slowest training, highest parameter count, moderate accuracy

---

## 6. Transfer Learning Details

### 6.1 Pre-trained Models Used
- **ResNet50**: ImageNet-1K V2 weights (14M+ images, 1000 classes)
- **VGG16**: ImageNet-1K V1 weights (14M+ images, 1000 classes)

### 6.2 Transfer Learning Strategy
- **Frozen Backbone**: Initially frozen, then fine-tuned
- **Feature Extraction**: Removed final classification layer, added custom heads
- **Learning Rate**: Lower learning rate (1e-4) for fine-tuning
- **Domain Adaptation**: Pre-trained features adapted for facial expression recognition

### 6.3 Benefits of Transfer Learning
- **Faster Convergence**: Leveraged pre-trained features
- **Better Generalization**: Learned from large-scale ImageNet dataset
- **Reduced Training Time**: Fewer epochs needed for convergence
- **Improved Performance**: Better feature representations

---

## 7. Correctly and Incorrectly Classified Images

### 7.1 Correctly Classified Examples
The model successfully classified various facial expressions including:
- **Happy expressions**: Clear smile detection with proper valence prediction
- **Sad expressions**: Accurate detection of negative emotions
- **Neutral expressions**: Proper classification of neutral emotional states
- **Surprise expressions**: Good detection of wide-eyed surprise features

### 7.2 Incorrectly Classified Examples
Common misclassification patterns observed:
- **Similar expressions**: Confusion between anger and disgust
- **Partial occlusion**: Difficulty with partially hidden faces
- **Extreme angles**: Poor performance on profile or tilted faces
- **Ambiguous expressions**: Uncertainty in borderline emotional states

### 7.3 Analysis of Errors
- **Domain-specific challenges**: Some expressions are inherently ambiguous
- **Data quality**: Variations in lighting and image quality affect performance
- **Class imbalance**: Some emotion classes are underrepresented
- **Feature complexity**: Subtle emotional differences are hard to capture

---

## 8. Conclusion

### 8.1 Key Findings
1. **ResNet50** achieved the best overall performance with transfer learning
2. **Custom CNN** showed competitive results despite training from scratch
3. **Transfer learning** significantly improved convergence speed and accuracy
4. **Continuous emotion prediction** is more challenging than classification



