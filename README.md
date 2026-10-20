# Facial Recognition & Classification Pipeline

An end-to-end computer vision project that processes raw images, extracts facial features using Wavelet Transforms, and classifies identities using classical machine learning models. Originally built as a university project (CSE422), this repository contains the refurbished and optimized pipeline.

## Features

* **Automated Face & Eye Detection:** Utilizes OpenCV's Haar Cascade classifiers to identify and crop faces only when eyes are clearly visible, ensuring high-quality training data.
* **Feature Extraction:** Combines raw scalled images with Haar Wavelet Transforms (`db1`) to capture both spatial and frequency-domain features for robust classification.
* **Model Benchmarking:** Compares multiple scikit-learn classifiers, including Support Vector Machines (Linear, RBF, Sigmoid), Random Forest, Naive Bayes, and Multi-Layer Perceptron (MLP) Neural Networks.

## Tech Stack

* **Language:** Python
* **Computer Vision:** OpenCV (`cv2`)
* **Machine Learning:** Scikit-Learn
* **Signal Processing:** PyWavelets (`pywt`)
* **Data Manipulation:** NumPy
* **Visualization:** Matplotlib

## Dataset Structure

The pipeline expects a `Dataset` folder containing subfolders for each individual (e.g., celebrities or students). The preprocessing script automatically creates a `Cropped` directory with the isolated face data.

```text
📁 Dataset/
│
├── 📁 Original_Images/
│   ├── 📁 Person_A/ (contains .jpg, .png)
│   └── 📁 Person_B/
│
└── 📁 Cropped/ (Generated automatically)
    ├── 📁 Person_A/
    └── 📁 Person_B/
```

## Setup & Installation

1. Clone the repository:

```bash git clone [https://github.com/yourusername/face-recognition-pipeline.git](https://github.com/yourusername/face-recognition-pipeline.git)
cd face-recognition-pipeline

2. Install the required dependencies:

```bash pip install numpy opencv-python matplotlib PyWavelets scikit-learn

3. Ensure the dataset paths in main.py match your local or cloud environment.

## Usage
- Data Preprocessing: Run the cropping loop. The script iterates through your dataset, detects faces using haarcascade_frontalface_default.xml and haarcascade_eye.xml, and saves the valid cropped faces.

- Feature Engineering: The pipeline stacks the raw scaled image arrays vertically with the wavelet-transformed arrays.

- Training & Evaluation: The script splits the data and trains multiple models.

## Model Performance

Based on initial testing with the dataset:
- SVM (Linear): ~89-92% Testing Accuracy (Top Performer)
- Random Forest: ~83% Testing Accuracy
- SVM (RBF): ~73-76% Testing Accuracy
- Naive Bayes / MLP: Requires further tuning for this specific feature space.

