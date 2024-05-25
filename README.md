# ASL Sign Language Recognition

This repository contains code for training and evaluating a Convolutional Neural Network (CNN) model to recognize American Sign Language (ASL) signs in real-time using a webcam.

## Table of Contents

- [Introduction](#introduction)
- [Setup](#setup)
  - [Windows](#windows)
  - [macOS](#macos)
  - [Linux](#linux)
- [Training the Model](#training-the-model)
- [Evaluating the Model](#evaluating-the-model)
- [Real-time Detection](#real-time-detection)
- [Google Colab Notebook](#google-colab-notebook)

## Introduction

This project uses a CNN to recognize ASL signs from images. The model is trained on a dataset of ASL images and can be used for real-time recognition through a webcam.

## Setup

### Prerequisites

- Python 3.x
- Virtual environment tool (optional but recommended)

### Windows

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/madhu-sv/asl-sign-language-recognition.git
   cd asl-sign-language-recognition```
2. **Set Up a Virtual Environment:**
    ```bash
    python -m venv venv
    venv\Scripts\activate
    ```
3. **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### macOS

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/madhu-sv/asl-sign-language-recognition.git
   cd asl-sign-language-recognition\
   ```
2. **Set Up a Virtual Environment:**
    ```bash
    python -m venv venv
    venv\Scripts\activate
    ```
3. **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Training the Model

1. **Prepare the Dataset:**
   - Download the ASL dataset and place it in the `data` directory.
   - Ensure the directory structure is as follows:
     ```
     data/
       ├── asl_alphabet_train/
       └── asl_alphabet_test/
     ```

2. **Run the Training Script:**
   ```bash
   python src/train.py
   ```

3. **Output:**
    - The trained model will be saved in the `models` directory `as asl_model.h5`.

## Evaluating the Model

1. **Run the Evaluation Script:**
   ```bash
   python src/evaluate.py
   ```

2. **Output:**
    - The script will print the test accuracy of the model.


## Real-time Detection

1. **Run the Real-time Detection Script:**
   ```bash
   python src/real_time_detection.py
   ```
2. **Note:**
    - This script uses OpenCV to access the webcam and perform real-time ASL recognition.
    - Ensure your webcam is connected and accessible.

