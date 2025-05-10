# Pneumonia Detection Using Chest X-Ray Images

This project implements a deep learning-based web application for detecting pneumonia from chest X-ray images. It uses an ensemble of four pretrained convolutional neural networks (DenseNet121, MobileNetV3 Small, EfficientNet-B0, and VGG16) to classify images as Normal or Pneumonia, with a confidence score. The application is built with Flask, HTML, JavaScript, and CSS, and deployed on an AWS EC2 instance.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Deployment](#deployment)
- [Contributing](#contributing)

## Project Overview
The project leverages a dataset of 5,856 pediatric chest X-ray images to train and evaluate deep learning models for pneumonia detection. The ensemble model combines predictions from DenseNet121, MobileNetV3 Small, EfficientNet-B0, and VGG16 using hard majority voting and provides a confidence score by averaging model probabilities. The web application allows users to upload X-ray images (PNG, JPG, JPEG) and receive predictions with confidence scores.

## Dataset
The dataset, sourced from the Guangzhou Women and Children’s Medical Center, contains:
- **Train**: 3,883 Pneumonia, 1,349 Normal
- **Validation**: 780 Pneumonia, 267 Normal
- **Test**: 390 Pneumonia, 234 Normal

Available at: [Kaggle Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/tolgadincer/labeled-chest-xray-images).

## Installation
To set up the project locally, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/YA195/Pneumonia-Detector
   cd pneumonia-detection
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Download pretrained model weights**:
   The model weights (`best_densenet_model.pth`, `Best_mobilenet.pth`, `best_efficientnet_model.pth`, `best_vgg16_model.pth`) are managed using Git LFS. To download them, ensure Git LFS is installed (`git lfs install`) and run:
   ```bash
   git lfs pull
## Usage
![Screenshot 2025-05-10 193031](https://github.com/user-attachments/assets/4c28848a-f85c-42c1-8834-390119257e15)
1. **Run the Flask application**:
   ```bash
   python app.py
   ```
   The application will start on `http://localhost:5126`.

2. **Access the web interface**:
   Open a browser and navigate to `http://localhost:5126`. Upload a chest X-ray image (PNG, JPG, or JPEG, max 16 MB) to receive a prediction (Normal or Pneumonia) and a confidence score.

3. **API Endpoint**:
   Use the `/analyze` endpoint for programmatic access:
   ```bash
   curl -X POST -F "file=@path/to/image.jpg" http://localhost:5126/analyze
   ```
   Response example:
   ```json
   {
       "prediction": "PNEUMONIA",
       "confidence": 92.34
   }
   ```

## Deployment
The application is deployed on an AWS EC2 instance with the following specifications:
- **vCPUs**: 2
- **Platform**: Linux/UNIX
- **Launch Time**: May 1, 2025, 23:25:53 GMT+0300
### Access the application at (http://16.171.254.216:8000/).



## Contributing
Contributions are welcome! To contribute:
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/your-feature`).
3. Commit changes (`git commit -m "Add your feature"`).
4. Push to the branch (`git push origin feature/your-feature`).
5. Open a pull request.

