# 📑 Research Report: Waste Classification Using Neural Networks

## 1. Introduction and Project Goals

### 1.1 Motivation
The waste classification project aims to create an intelligent system for recognizing different types of waste based on images from a smartphone camera. In the era of growing environmental issues and the necessity of proper waste segregation, automatic recognition of waste categories can significantly support recycling processes and environmental protection.

### 1.2 Research Objectives
- Develop a deep learning model capable of classifying 6 categories of waste with high accuracy
- Compare the effectiveness of different neural network architectures (ResNet50, EfficientNet-B0, MobileNetV2)
- Implement a solution suitable for deployment in a mobile application

## 2. Literature Review and Approach Justification

### 2.1 Transfer Learning in Image Classification
Transfer learning is a technique that uses models pre-trained on large datasets (ImageNet) to solve similar classification problems. This approach is particularly effective with limited computational resources and smaller datasets.

### 2.2 Neural Network Architectures
- **ResNet50**
- **EfficientNet-B0**
- **MobileNetV2**

## 3. Research Methodology

### 3.1 Dataset
- **Source**: Garbage_classification dataset containing images of waste in 6 categories (https://www.kaggle.com/datasets/sumn2u/garbage-classification-v2), (https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification/data)
- **Categories**: cardboard, glass, metal, paper, plastic, trash (mixed waste)
- **Dataset size**: ~13,000 samples
- **Data split**: 60% training, 10% validation, 30% testing

### 3.2 Data Split (60/10/30)

### 3.3 Preprocessing and Data Augmentation

#### 3.3.1 Image Size Justification
- **ResNet50**: 256×256 pixels - standard input size for this model, balancing detail preservation and memory requirements
- **EfficientNet-B0**: 224×224 pixels - native input size optimized for efficiency
- **MobileNetV2**: 224×224 pixels - optimized for mobile devices

#### 3.3.2 Augmentation Decisions
In initial experiments, various augmentation techniques were tested:
- **Initial approach**: Aggressive augmentation (±30° rotations, flips, brightness adjustments)
- **Issue**: The model learned mostly from distorted images, reducing performance on real-world data
- **Final solution**: Minimal augmentation - only resizing

This allowed the models to focus on learning actual distinguishing features of waste rather than coping with artificial distortions.

### 3.4 Training Architecture

#### 3.4.1 Batch Size
Batch size of 32 was chosen as a compromise between:
- **GPU memory limitations**
- **Training speed**

Tests with batch size 16 caused unstable training, while 64 showed no significant improvement.

## 4. Experimental Results

### 4.1 Model Performance Comparison

| Model | Validation Accuracy | Validation Loss | Learning Rate | Parameters |
|-------|----------------------|-----------------|---------------|-------------|
| **ResNet50** | **95.21%** | 1.107 | 5.5e-05 | ~23.5M |
| **EfficientNet-B0** | **92.08%** | **0.264** | 1e-04 | ~5.3M |
| **MobileNetV2** | **88.54%** | 0.412 | 2e-04 | ~3.5M |

### 4.2 Learning Curve Analysis

#### 4.2.1 ResNet50 - Best Accuracy
- **Start**: 82.58% (epoch 1)
- **Peak**: 96.41% (epoch 4)
- **Final**: 95.21% (epoch 8)
- **Characteristics**: Stable training, high loss values but excellent accuracy

#### 4.2.2 EfficientNet-B0 - Best Loss
- **Start**: 74.58% (epoch 1)
- **Peak**: 92.45% (epoch 7)
- **Final**: 92.08% (epoch 8)
- **Characteristics**: Lowest validation loss, smooth convergence

#### 4.2.3 MobileNetV2 - Fastest Convergence
- **Start**: 85.70% (epoch 1)
- **Peak**: 91.98% (epoch 4)
- **Final**: 88.54% (epoch 8)
- **Characteristics**: Strong start, but plateaued in later epochs

## 5. Detailed Results Analysis

### 5.1 ResNet50 - Accuracy Leader
**Strengths:**
- Highest validation accuracy (95.21%)
- Stable training behavior
- Good generalization across diverse waste types

**Weaknesses:**
- High loss values (possible overconfidence)
- Largest memory requirements

### 5.2 EfficientNet-B0 - Best Balance
**Strengths:**
- Lowest validation loss (0.264)
- Very good accuracy (92.08%)
- Best accuracy-to-size tradeoff

**Weaknesses:**
- Slightly lower accuracy than ResNet50
- Sensitive to input image quality

### 5.3 MobileNetV2 - Mobile Optimization
**Strengths:**
- Smallest model size (3.5M parameters)
- Fastest inference time
- Suitable for mobile apps

**Weaknesses:**
- Lowest accuracy (88.54%)
- Limited model expressiveness

## 6. Problematic Cases Analysis

### 6.1 Category Confusion
Typical misclassifications observed during training:

1. **Paper vs Cardboard**: Similar textures in certain views
2. **Plastic vs Metal**: Surface reflections can confuse models
3. **Glass vs Metal**: Transparency of glass can be problematic

### 6.2 Impact of Image Quality
Models showed varying sensitivity to:
- **Lighting**: EfficientNet most sensitive
- **Angle**: ResNet50 most robust
- **Background clutter**: MobileNetV2 most affected

## 7. Final Model Selection Justification

### 7.1 Decision Criteria
After detailed analysis, **ResNet50 was chosen as the final model**.

### 7.2 Final Argumentation
**ResNet50 outperformed competitors** for the following reasons:

1. **Highest accuracy**: 95.21% is production-level
2. **Reliability**: Stable results under varying image conditions
3. **Proven technology**: ResNet has a long history of success
4. **Optimization potential**: Further fine-tuning possible for specific use cases

### 7.3 Trade-offs and Limitations
**Accepted trade-offs:**
- Higher memory requirements (23.5M vs 5.3M for EfficientNet)
- Slower inference (acceptable for web applications)
- Higher energy consumption (less critical in server-side apps)

## 8. Research Contributions and Innovations

### 8.1 Implementation Optimizations
- **Efficient Data Loading**: Optimized DataLoader for CUDA
- **Memory Management**: GPU memory optimization using @torch.no_grad()

### 8.2 Comparative Methodology
- **Uniform test conditions**: All models trained under identical settings
- **Standardized metrics**: Consistent measures for fair comparison
- **Reproducible results**: Seed fixing ensured repeatability

## 9. Conclusions and Future Directions

### 9.1 Key Achievements
1. **High classification accuracy**: 95.21% on validation set
2. **Comprehensive architecture comparison**: Thorough analysis of 3 approaches
3. **Deployment readiness**: Model prepared for integration with a web application

### 9.2 Practical Applications
ResNet50 with 95.21% accuracy is ready for deployment in:
- **Mobile applications** for waste segregation
- **IoT systems** in smart bins
- **Automated sorting machines**

## 10. Web Application
To practically demonstrate the trained model’s capabilities, a simple web application was built. It allows users to interactively classify waste.

### 10.1 Application Architecture
The app follows a client-server architecture:
- **Backend**: Python server using lightweight Flask framework
- **Frontend**: User interface built with HTML, CSS, and JavaScript

### 10.2 Backend (Flask)
The core of the app is a Flask server responsible for handling requests and performing predictions.
1. **Model Loading**
- On startup, the ResNet50 model is defined.
- Instead of retraining, pretrained=False is used and weights of the final trained model are loaded from a `.pth` file. The model is switched to evaluation mode (`model.eval()`), ensuring consistent predictions.
2. **Image Preprocessing**
- The server defines a `/predict` endpoint listening for POST requests.
- After receiving an image, it is resized to the appropriate resolution and converted.
3. **Prediction**
- The preprocessed tensor is fed into the ResNet50 model.
- The model outputs a probability vector for each of the 6 waste classes.
- The backend maps predictions to class names with probabilities and returns a JSON response to the frontend.

### 10.3 Frontend (JavaScript)
The user interface provides two ways to submit an image for classification:

1. **File Upload**: User selects a graphic file (JPG, PNG) from their device.
2. **Camera Capture**: The app requests camera access to take a real-time picture of waste.

In both cases, the JavaScript code:
- Captures the image (from file or video frame)
- Formats the data appropriately
- Sends the image via an asynchronous POST request to the Flask `/predict` endpoint
- Receives the JSON response and displays the predicted waste category to the user

## 11. Desktop Camera Application

The project also includes a local desktop application that does not require Flask.
It shows a live camera preview and performs one prediction only when the user clicks
**Capture & Classify**.

### 11.1 Train and Package the Model

Place images under `data/garbage_classification/<class_name>/`, then run:

```bash
python pytorch_resnet.py
```

The training script saves the best model and its class metadata to:

```text
models/pytorch_resnet/waste_classifier_resnet50.pth
```

### 11.2 Run Camera Classification

```bash
python main.py
```

The application requires a camera available as device `0`. On Windows, enable
camera access for desktop applications and close other programs that may be using
the camera before starting it.

