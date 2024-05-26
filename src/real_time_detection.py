# src/real_time_detection.py

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_model(model_path='../models/asl_model.keras'):
    """
    Load the trained CNN model.
    
    Args:
        model_path (str): Path to the trained model.
    
    Returns:
        model: The loaded Keras model.
    """
    try:
        model = tf.keras.models.load_model(model_path)
        print("Model loaded successfully.")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def preprocess_frame(frame, img_size=(64, 64)):
    """
    Preprocess the webcam frame for prediction.
    
    Args:
        frame (numpy array): The captured frame.
        img_size (tuple): Target size of the image.
    
    Returns:
        img: Preprocessed image.
    """
    img = cv2.resize(frame, img_size)
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def real_time_detection(model, sequence_length=30, class_labels=None):
    cap = cv2.VideoCapture(0)
    frame_buffer = []

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break

        frame = preprocess_frame(frame)
        frame_buffer.append(frame)

        if len(frame_buffer) == sequence_length:
            frames = np.expand_dims(np.array(frame_buffer), axis=0)
            predictions = model.predict(frames)
            predicted_class = np.argmax(predictions[0])
            predicted_label = class_labels[predicted_class] if class_labels else predicted_class

            cv2.putText(frame, f'Class: {predicted_label}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            frame_buffer.pop(0)

        cv2.imshow('Real-time ASL Recognition', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    model_path = '../models/asl_model.keras'
    model = load_model(model_path)
    
    if model:
        # Extract class labels from the training generator (assuming it's available)
        train_dir = '../data/asl_alphabet_train'
        train_datagen = ImageDataGenerator(rescale=1./255)
        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=(64, 64),
            batch_size=32,
            class_mode='categorical'
        )
        class_labels = list(train_generator.class_indices.keys())
        print("Class labels:", class_labels)
        
        real_time_detection(model, class_labels=class_labels)
