# FILE CONTAINS TWO APPLICATIONS FOR SVD FROM TF.LINALG:
# 1. USING SVD FOR FEATURE EXTRACTION USING ITS REDUCED DIMENSIONALITY IMPLEMENTATION AS INPUT TO SIMPLE NEURAL NETWORK:

import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0  # Normalize

# Function to perform SVD and extract features
def svd_feature_extraction(images, k):
    features = []
    for img in images:
        # Compute SVD
        u, s, v = tf.linalg.svd(tf.cast(img, tf.float32))
        # Reconstruct the image using the top-k singular values and vectors
        img_reduced = tf.matmul(u[:, :k], tf.linalg.diag(s[:k]))
        img_reduced = tf.matmul(img_reduced, v[:k, :])
        features.append(img_reduced.numpy().flatten())  # Flatten for model input
    return np.array(features)

# Extract features using top-k singular values
k = 10  # Number of singular values to retain
x_train_svd = svd_feature_extraction(x_train, k)
x_test_svd = svd_feature_extraction(x_test, k)

# Build a simple neural network
model = Sequential([
    Dense(128, activation='relu', input_shape=(x_train_svd.shape[1],)),
    Dense(10, activation='softmax')  # 10 classes for MNIST digits
])

# Compile and train the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train_svd, y_train, epochs=5, batch_size=32, validation_data=(x_test_svd, y_test))

# Evaluate the model
loss, accuracy = model.evaluate(x_test_svd, y_test)
print(f"Test Accuracy: {accuracy:.2f}")





# COMPRESS IMAGE DATA WHILE PRESERVING CRITICAL FEATURES AND REDUCING NOISE:
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.datasets import mnist

# Load MNIST dataset
(x_train, _), (x_test, _) = mnist.load_data()
x_test = x_test / 255.0  # Normalize

# Add random noise to images
noise_factor = 0.5
x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)  # Keep values in range [0, 1]

# Function to perform SVD-based compression and noise reduction
def svd_compress_denoise(images, k):
    denoised_images = []
    for img in images:
        # Compute SVD
        u, s, v = tf.linalg.svd(tf.cast(img, tf.float32))
        # Reconstruct the image using the top-k singular values and vectors
        img_compressed = tf.matmul(u[:, :k], tf.linalg.diag(s[:k]))
        img_compressed = tf.matmul(img_compressed, v[:k, :])
        denoised_images.append(img_compressed.numpy())
    return np.array(denoised_images)

# Compress and denoise images with top-k singular values
k = 10  # Number of singular values to retain
x_test_denoised = svd_compress_denoise(x_test_noisy, k)

# Visualize original, noisy, and denoised images
def plot_images(original, noisy, denoised, n=5):
    plt.figure(figsize=(10, 5))
    for i in range(n):
        # Original
        plt.subplot(3, n, i + 1)
        plt.imshow(original[i], cmap='gray')
        plt.axis('off')
        if i == 0:
            plt.title("Original")
        # Noisy
        plt.subplot(3, n, i + 1 + n)
        plt.imshow(noisy[i], cmap='gray')
        plt.axis('off')
        if i == 0:
            plt.title("Noisy")
        # Denoised
        plt.subplot(3, n, i + 1 + 2 * n)
        plt.imshow(denoised[i], cmap='gray')
        plt.axis('off')
        if i == 0:
            plt.title("Denoised")
    plt.show()

plot_images(x_test, x_test_noisy, x_test_denoised)