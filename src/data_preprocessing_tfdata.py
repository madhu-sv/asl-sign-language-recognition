# src/data_preprocessing_tfdata.py

import tensorflow as tf
import os

def parse_image(filename, label, num_classes):
    img = tf.io.read_file(filename)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [64, 64])
    img = img / 255.0
    label = tf.one_hot(label, num_classes)
    return img, label

def load_dataset(data_dir, batch_size=32, augment=False):
    classes = sorted(os.listdir(data_dir))
    class_indices = {cls: idx for idx, cls in enumerate(classes)}

    filepaths = []
    labels = []

    for cls in classes:
        class_dir = os.path.join(data_dir, cls)
        for img in os.listdir(class_dir):
            filepaths.append(os.path.join(class_dir, img))
            labels.append(class_indices[cls])

    filepaths = tf.constant(filepaths)
    labels = tf.constant(labels)

    num_classes = len(class_indices)

    dataset = tf.data.Dataset.from_tensor_slices((filepaths, labels))
    dataset = dataset.map(lambda x, y: parse_image(x, y, num_classes), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    
    if augment:
        dataset = dataset.map(lambda x, y: (tf.image.random_flip_left_right(x), y), num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.map(lambda x, y: (tf.image.random_brightness(x, max_delta=0.2), y), num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.map(lambda x, y: (tf.image.random_contrast(x, lower=0.8, upper=1.2), y), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    
    dataset = dataset.shuffle(buffer_size=len(filepaths))
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return dataset, num_classes

def create_data_generators(train_dir=None, test_dir=None, batch_size=32):
    train_dataset, test_dataset = None, None
    num_classes = 0
    
    if train_dir:
        train_dataset, num_classes = load_dataset(train_dir, batch_size=batch_size, augment=True)
    
    if test_dir:
        test_dataset, num_classes = load_dataset(test_dir, batch_size=batch_size, augment=False)
    
    return train_dataset, test_dataset, num_classes

if __name__ == "__main__":
    train_dir = '../data/asl_alphabet_train'
    test_dir = '../data/asl_alphabet_test'
    
    train_dataset, test_dataset, num_classes = create_data_generators(train_dir, test_dir)
    
    print(f"Number of classes: {num_classes}")
