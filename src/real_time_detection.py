# src/real_time_detection.py

import cv2
import numpy as np
import tensorflow as tf

def load_model(model_path='../models/asl_model.h5'):
    """
    Load the trained CNN model.
    
    Args:
        model_path (str): Path to the trained model.
    
    Returns:
        model: The loaded Keras model.
    """
    model = tf.keras.models.load_model(model_path)
    return model

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

def main():
    model = load_model()
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        img = preprocess_frame(frame)
        predictions = model.predict(img)
        predicted_class = np.argmax(predictions[0])

        cv2.putText(frame, f'Class: {predicted_class}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.imshow('Real-time ASL Recognition', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
