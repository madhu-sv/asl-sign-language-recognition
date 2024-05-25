# src/evaluate.py

import tensorflow as tf
from data_preprocessing import create_data_generators

def evaluate_model(test_dir):
    """
    Evaluate the trained CNN model on the ASL test dataset.
    
    Args:
        test_dir (str): Path to the test data directory.
    """
    test_generator = create_data_generators(test_dir, test_dir)[1]  # Only need test generator
    model = tf.keras.models.load_model('../models/asl_model.h5')

    test_loss, test_acc = model.evaluate(test_generator)
    print(f'Test accuracy: {test_acc}')
    return test_loss, test_acc

if __name__ == "__main__":
    test_dir = '../data/asl_alphabet_test'
    evaluate_model(test_dir)
