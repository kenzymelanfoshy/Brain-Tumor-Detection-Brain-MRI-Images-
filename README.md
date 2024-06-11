# Brain Tumor Detection using Convolutional Neural Networks (CNN)

## Overview

This repository contains a Jupyter notebook for detecting brain tumors using Convolutional Neural Networks (CNN) and Image Filters to enhance edges also to detect the tumor shape easier. The project leverages deep learning techniques to accurately classify brain MRI images, identifying the presence of tumors. This solution aims to assist in the early detection and diagnosis of brain tumors, potentially improving patient outcomes.


## Language and Libraries Used

### Programming Language
- **Python**: The primary programming language used for this project.

### Libraries and Frameworks
- **NumPy**: For numerical operations and handling arrays.
- **Pandas**: For data manipulation and analysis.
- **Matplotlib**: For plotting and visualizing data.
- **Seaborn**: For enhanced data visualization.
- **TensorFlow/Keras**: For building and training the Convolutional Neural Network.
- **Sklearn**: For additional machine learning utilities and metrics.
- **OpenCV**: For image processing tasks.
- **Glob**: For file handling and directory operations.
- **OS**: For operating system interactions.

## Notebook Description

The notebook is structured into various sections, each serving a specific purpose in the brain tumor detection pipeline:

1. **Introduction**: Provides an overview of the project and its objectives.
2. **Data Loading and Preprocessing**: 
   - Uses libraries like `Glob` and `OS` to load MRI images.
   - Applies image preprocessing techniques using `OpenCV` to enhance image quality.
3. **Exploratory Data Analysis (EDA)**:
   - Utilizes `Pandas`, `Matplotlib`, and `Seaborn` to analyze the dataset.
   - Visualizes the distribution of images and key features.
4. **Model Building**:
   - Constructs a Convolutional Neural Network (CNN) using `TensorFlow/Keras`.
   - Configures the model architecture, including layers, activation functions, and optimization algorithms.
5. **Model Training**:
   - Trains the CNN model on the preprocessed dataset.
   - Implements techniques to handle overfitting, such as data augmentation and dropout.
6. **Model Evaluation**:
   - Evaluates the model's performance using metrics from `Sklearn`.
   - Analyzes the accuracy, precision, recall, and F1-score.
7. **Results Visualization**:
   - Plots training and validation metrics to assess model performance.
   - Visualizes predictions on sample images to demonstrate the model's effectiveness.
8. **Conclusion**:
   - Summarizes the findings and discusses potential improvements and future work.

## Usage

1. **Clone the repository**:
   ```sh
   git clone https://github.com/kenzymelanfoshy/Brain-Tumor-Detection-Brain-MRI-Images-.git
   cd brain-tumor-detection
