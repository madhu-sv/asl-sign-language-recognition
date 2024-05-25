# src/data_preprocessing.py

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def create_data_generators(train_dir, test_dir, img_size=(64, 64), batch_size=32):
    """
    Create training and validation data generators.
    
    Args:
        train_dir (str): Path to the training data directory.
        test_dir (str): Path to the test data directory.
        img_size (tuple): Target size of the images.
        batch_size (int): Batch size for the data generators.
    
    Returns:
        train_generator, test_generator: Training and validation data generators.
    """
    train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical'
    )
    
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical'
    )
    
    return train_generator, test_generator

if __name__ == "__main__":
    train_dir = '../data/asl_alphabet_train'
    test_dir = '../data/asl_alphabet_test'
    
    train_generator, test_generator = create_data_generators(train_dir, test_dir)
    
    print(f"Training samples: {train_generator.samples}")
    print(f"Validation samples: {test_generator.samples}")
