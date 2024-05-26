# src/train.py

import os
import tensorflow as tf
from data_preprocessing import create_data_generators
from model import build_model

def train_model(train_dir, test_dir, epochs=10, batch_size=32):
    """
    Train the CNN model on the ASL dataset.
    
    Args:
        train_dir (str): Path to the training data directory.
        test_dir (str): Path to the test data directory.
        epochs (int): Number of epochs to train the model.
        batch_size (int): Batch size for training.
    """
    train_generator, test_generator = create_data_generators(train_dir, test_dir, batch_size=batch_size)
    model = build_model(num_classes=len(train_generator.class_indices))

    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=test_generator
    )

    model.save('../models/asl_model.keras')

    test_loss, test_acc = model.evaluate(test_generator)
    print(f'Test accuracy: {test_acc}')
    return history

if __name__ == "__main__":
    train_dir = '../data/asl_alphabet_train'
    test_dir = '../data/asl_alphabet_test'
    
    if not os.path.exists(train_dir):
        raise FileNotFoundError(f"Training directory not found: {train_dir}")
    if not os.path.exists(test_dir):
        raise FileNotFoundError(f"Test directory not found: {test_dir}")

    history = train_model(train_dir, test_dir)
