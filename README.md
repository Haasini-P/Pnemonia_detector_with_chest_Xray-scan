# Pneumonia Detection Using Chest X-ray Scans

## Overview
This project utilizes deep learning techniques to detect pneumonia from chest X-ray images. The model is trained on a dataset of labeled X-ray scans and employs convolutional neural networks (CNNs) for classification. The goal is to assist in early and accurate diagnosis of pneumonia.

## Features
- Uses Convolutional Neural Networks (CNNs) for image classification.
- Trained on a publicly available dataset of chest X-ray images.
- Achieves high accuracy in distinguishing between normal and pneumonia-affected lungs.
- Provides a web-based interface for uploading and predicting chest X-rays (optional).

## Dataset
The dataset used for training and validation is sourced from [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia). It consists of:
- **Training Set**: X-ray images of normal and pneumonia cases.
- **Validation Set**: Used to fine-tune the model.
- **Test Set**: Evaluates model performance.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/pneumonia-detection.git
   cd pneumonia-detection
   ```
2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download the dataset and place it in the `data/` directory.

## Model Training
To train the model, run:
```bash
python train.py
```
This script preprocesses the dataset, trains the CNN model, and saves the trained model as `pneumonia_model.h5`.

## Inference
To test the model on a new chest X-ray image:
```bash
python predict.py --image path/to/xray.jpg
```

## Web Interface (Optional)
A Flask-based web application is included for an easy-to-use interface. To run it:
```bash
python app.py
```
Access the web interface at `http://localhost:5000/`.

## Performance
- Achieves an accuracy of **XX%** on the test set.
- Utilizes **transfer learning** techniques for improved results.

## Future Improvements
- Enhance model accuracy with additional data augmentation.
- Deploy as a cloud-based API for real-world usability.
- Improve explainability with heatmaps for visualization.

## Train the Model
If you don’t have the model file, train it by running:
```bash
python train.py

## Contributors
- **P.Haasini** (haasinip1111@gmail.com)

