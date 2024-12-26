import tensorflow as tf
import numpy as np
import pickle

# Load the CIFAR-10 dataset
from tensorflow.keras.datasets import cifar10
(x_train_full, y_train_full), (x_test_full, y_test_full) = cifar10.load_data()

# Subset the dataset (e.g., 10%)
subset_size_train = x_train_full.shape[0] // 10
subset_size_test = x_test_full.shape[0] // 10
x_train = x_train_full[:subset_size_train]
y_train = y_train_full[:subset_size_train]
x_test = x_test_full[:subset_size_test]
y_test = y_test_full[:subset_size_test]

# Normalize pixel values to be between 0 and 1
x_train, x_test = x_train / 255.0, x_test / 255.0

# Save the preprocessed subset data for later use
with open('preprocessed_subset_data.pkl', 'wb') as f:
    pickle.dump((x_train, y_train, x_test, y_test), f)

# Optional: Load the preprocessed subset data from disk
with open('preprocessed_subset_data.pkl', 'rb') as f:
    x_train_loaded, y_train_loaded, x_test_loaded, y_test_loaded = pickle.load(f)

