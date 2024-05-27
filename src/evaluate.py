# src/evaluate.py

import tensorflow as tf
from data_preprocessing_tfdata import create_data_generators

def evaluate_model(test_dir):
    """
    Evaluate the trained CNN model on the ASL test dataset.
    
    Args:
        test_dir (str): Path to the test data directory.
    """
    _, test_dataset, num_classes = create_data_generators(test_dir=test_dir, batch_size=32)
    model = tf.keras.models.load_model('../models/asl_model.keras')

    test_loss, test_acc = model.evaluate(test_dataset)
    print(f'Test accuracy: {test_acc}')
    return test_loss, test_acc

if __name__ == "__main__":
    test_dir = '../data/asl_alphabet_test'
    evaluate_model(test_dir)
