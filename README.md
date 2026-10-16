# Siamese Neural Network-Based Face Recognition System

## Overview

This project implements a face recognition system using a **Siamese Neural Network (SNN)** with a shared Convolutional Neural Network (CNN) feature extractor. The system generates 128-dimensional facial embeddings and compares them using Euclidean distance to identify registered users or classify an input face as unknown.

The model is trained using the AT&T Database of Faces and integrated with real-time face detection using OpenCV's Haar Cascade classifier.

## Key Features

- **Face Detection:** Detects faces using the Viola–Jones method with a Haar Cascade classifier.
- **Image Preprocessing:** Converts facial images to grayscale, resizes them to 112 × 92 pixels, and normalizes pixel values.
- **Siamese Neural Network:** Uses a shared CNN feature extractor to generate 128-dimensional facial embeddings.
- **Similarity Measurement:** Computes Euclidean distance between facial embeddings.
- **Threshold-Based Authentication:** Uses a validation-selected threshold to distinguish registered users from unknown users.
- **User Enrollment:** Captures multiple facial samples and stores a reference embedding for each registered user.
- **Real-Time Recognition:** Supports webcam-based face detection and recognition.
- **Graphical User Interface:** Provides options to register users, start recognition, view registered users, delete users, and exit the application.

## System Architecture

The system consists of two main stages:

1. **Model Development:** Face images are preprocessed, positive and negative image pairs are generated, and the Siamese Network is trained using contrastive loss.
2. **Real-Time Authentication:** A webcam frame is processed, a facial embedding is generated, and its distance from stored user embeddings is compared against the recognition threshold.

## Technology Stack

| Component | Technologies |
|---|---|
| Programming Language | Python |
| Deep Learning | TensorFlow, Keras |
| Computer Vision | OpenCV |
| Numerical Processing | NumPy |
| Data Processing | Pandas |
| Visualization | Matplotlib |
| Machine Learning Utilities | Scikit-learn |
| Model Format | HDF5 (`.h5`) |
| User Data Storage | JSON |
| Development Environment | Google Colab, Visual Studio Code |

## Dataset

The project uses the **AT&T Database of Faces**, also known as the ORL Face Database.

- 40 subjects
- 10 grayscale images per subject
- 400 images in total
- Original images are resized to 112 × 92 pixels
- Pixel values are normalized to the range [0, 1]

The dataset is divided by identity for training and testing to avoid overlap between the training and test identities.

Dataset source: [AT&T Database of Faces — Kaggle](https://www.kaggle.com/datasets/kasikrit/att-database-of-faces)

## Model Architecture

The shared CNN feature extractor consists of:

1. Input layer with dimensions 112 × 92 × 1.
2. Convolutional layer with 64 filters and a 3 × 3 kernel, using ReLU activation.
3. Max-pooling layer.
4. Second convolutional layer with 64 filters and a 3 × 3 kernel, using ReLU activation.
5. Flatten layer.
6. Dense layer with 128 units and sigmoid activation.

The Siamese Network processes two input images through the same shared CNN. Their resulting embeddings are compared using Euclidean distance.

### Contrastive Loss

Contrastive loss is used to learn discriminative facial representations. Positive pairs contain images of the same person, while negative pairs contain images of different people.

The model is trained using the Adam optimizer with a contrastive-loss margin of 1.

## Training Configuration

| Parameter | Value |
|---|---|
| Input Image Size | 112 × 92 × 1 |
| Embedding Dimension | 128 |
| Optimizer | Adam |
| Contrastive Loss Margin | 1.0 |
| Batch Size | 128 |
| Initial Training Epochs | 10 |
| Training Accuracy | 100% |
| Validation Accuracy | 96.09% |
| Test Accuracy | 78.52% |
| Selected Recognition Threshold | 0.5319 |

**Note:** Training, validation, and test accuracy refer to the recorded experimental results. The test accuracy is lower than the validation accuracy, indicating that further evaluation and generalization improvements may be useful.

## Project Structure

```text
Siamese_Face_Recognition/
├── app.py
├── face_detection.py
├── onboarding.py
├── face_recognition_system.py
├── models/
│   ├── shared_cnn.h5
│   ├── siamese_nn.h5
│   └── threshold.json
├── registered_users/
├── Siamese_Face_Recognition.ipynb
├── requirements.txt
├── README.md
└── .gitignore
```

The structure above describes the intended organization. Keep any additional modules required by your actual implementation.

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/siamese-face-recognition.git
cd siamese-face-recognition
```

Replace `YOUR_USERNAME` with your GitHub username.

### 2. Create a virtual environment

For Windows:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Use Python 3.11 for the TensorFlow 2.15 environment used during development.

### 4. Verify the model files

Ensure that the following files are present inside the `models/` directory:

- `shared_cnn.h5`
- `siamese_nn.h5`
- `threshold.json`

The trained models and threshold configuration must be available before running the application.

### 5. Run the application

```bash
python app.py
```

Allow webcam access when prompted. Ensure that your webcam is connected and available.

## User Enrollment

1. Open the application.
2. Select the user registration option.
3. Enter a unique user ID and name.
4. Position your face in front of the webcam.
5. Capture the required facial samples.
6. The system generates embeddings using the shared CNN.
7. The reference embedding and user information are stored in the `registered_users/` directory.

## Face Recognition

1. Start the face recognition option.
2. The webcam captures a frame.
3. The Haar Cascade classifier detects the face.
4. The face is cropped and preprocessed.
5. The shared CNN generates a 128-dimensional embedding.
6. The embedding is compared with the stored reference embeddings.
7. The minimum Euclidean distance is compared with the selected threshold.
8. The application displays the recognized user's identity or `Unknown`.

## Experimental Results

The following results were observed during testing:

| Test Case | Observed Distance | Result |
|---|---:|---|
| Registered user: Pratiyusha (U001) | 0.2801 | Recognized |
| Registered user: Barna (U002) | 0.3454 | Recognized |
| Unknown user | 0.8272 | Unknown |
| No face detected | N/A | Authentication skipped |

The selected recognition threshold was 0.5319. Distances below or equal to the threshold are considered matches, subject to the system's minimum-distance comparison.

## Model Files and Privacy

The trained model files can be included in the repository if their sizes comply with GitHub's file-size limits. If a model file is too large, distribute it separately or use Git Large File Storage (Git LFS).

Do not commit real users' facial embeddings, registration JSON files, or other personal biometric data to a public repository. The application can create the `registered_users/` directory locally when needed.

## Limitations

- Recognition performance may vary with lighting, pose, image quality, and camera conditions.
- The recorded test accuracy is lower than the validation accuracy.
- The selected threshold may not generalize to all cameras or environments.
- Haar Cascade detection may fail for unusual poses or difficult lighting.
- The system is an academic prototype and should not be treated as a production-grade security or identity verification solution.

## Future Enhancements

- Improve generalization through additional training data and augmentation.
- Evaluate alternative face detectors and embedding architectures.
- Calibrate the recognition threshold on a larger, representative validation dataset.
- Add liveness detection to reduce presentation attacks.
- Improve handling of multiple simultaneous faces.
- Introduce secure storage and access controls for registered user data.
- Develop a deployment-ready interface with stronger authentication and monitoring.

## Author

**Pratiyusha Kanungo**

B.Tech — Computer Science and Engineering (Artificial Intelligence and Machine Learning)

Institute of Technical Education and Research (ITER), SOA University

## License

This project is intended for academic and educational purposes. Add an appropriate open-source license if you choose to permit reuse and redistribution.
