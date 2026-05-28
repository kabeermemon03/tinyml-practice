# What Is FashionMNIST?

# FashionMNIST is:

# 28×28 grayscale clothing images,
# 10 clothing categories,
# 60,000 training images,
# 10,000 testing images.

# Examples:

# shoes
# shirts
# bags
# dresses
# sneakers

# ============================================================

# CNNs expect:

# (height, width, channels)

# ====================================================================

# ReLU function:

# f(x)=max(0,x)

# Negative values become zero.

# Adds nonlinearity

# =======================================================================

# MaxPooling2D
# MaxPooling2D(2,2)

# Shrinks feature maps.

# Example:

# 28×28
# ↓
# 14×14

# Reduces computation.

# ==================================================================


# Flatten Layer
# Flatten()

# Converts 2D feature maps into 1D vector.

# Needed before Dense layers.

# ======================================================================

# Load dataset
# ↓
# Preprocess images
# ↓
# Build CNN
# ↓
# Train CNN
# ↓
# Evaluate CNN
# ↓
# Predict clothing classes
# ↓
# Visualize predictions
# ↓
# Save trained model

# ===================================================================

# IMPORTANT UNDERSTANDING
# First CNN Layer

# Detects:

# simple edges
# lines
# Second CNN Layer

# Detects:

# more advanced patterns
# shoe shapes
# soles

# ====================================================================


# The model sees an image and answers:

# Horse 🐴
# or
# Human 🧍

# =====================================================================

# Regular neural networks are bad for images because images are huge matrices.

# CNNs (Convolutional Neural Networks) are designed specifically for images because they:

# detect edges
# detect shapes
# detect textures
# combine features into objects

# For example:

# Early CNN layers detect:
# edges
# curves
# colors
# Middle layers detect:
# eyes
# legs
# face shapes
# Deep layers detect:
# “this looks like a horse”

# That’s the magic of CNNs

# =====================================================================

# from tensorflow.keras.preprocessing.image import ImageDataGenerator

# This is one of the most important tools in computer vision.

# What it does

# It:

# loads images
# resizes them
# labels them
# feeds them to model in batches

# Think of it like:

# “Image manager for TensorFlow.”

# =====================================================================


# TensorFlow Datasets (TFDS) Summary

# TensorFlow Datasets (TFDS) is a library that makes it easier to load and use datasets in TensorFlow. Instead of manually downloading ZIP files, organizing folders, and labeling images, TFDS automatically downloads, prepares, and formats datasets for training neural networks.

# Previously, datasets like Horses vs Humans required images to be sorted into folders such as `horses/` and `humans/`. TensorFlow used these folder names as labels through `ImageDataGenerator`. While effective, this method required manual setup and specific folder structures.

# With TFDS, datasets can be loaded directly using:

# ```python
# import tensorflow_datasets as tfds

# data = tfds.load("fashion_mnist")
# ```

# TFDS supports many types of datasets including images, text, audio, and video.

# Important concepts in TFDS include:

# * **Splits API**: divides data into training, validation, and testing sets.
# * **Mapping Functions**: apply preprocessing or augmentation to every image.

# Example augmentation workflow:

# ```python
# def augmentimages(image, label):
#     image = tf.cast(image, tf.float32)
#     image = image / 255
#     image = tf.image.random_flip_left_right(image)
#     return image, label

# data = tfds.load('horses_or_humans', split='train', as_supervised=True)

# train = data.map(augmentimages)
# ```

# Key ideas:

# * `split='train'` loads the training data.
# * `as_supervised=True` returns data as `(image, label)` pairs.
# * `map()` applies a function to every image.
# * Dividing by 255 normalizes pixel values from 0–255 to 0–1.
# * Image augmentation improves model performance by creating modified versions of images.

# TFDS simplifies data handling and preprocessing, making it easier to train machine learning and TinyML models efficiently.


# ========================================================================

# train = data.map(augmentimages)

# means:

# Take every image
# Send it into augmentimages()
# Return modified image



# ==============================================================================


# PART 7 — What is augmentation?

# Augmentation =

# artificially changing images to improve learning.

# Examples:

# flip image
# rotate image
# zoom
# brightness change

# Why?

# Because it helps AI generalize better.

# PART 8 — This function explained line-by-line

# They wrote:

# def augmentimages(image, label):

# This creates a function.

# Inputs:

# image
# label

# =====================================================================

# PART 9 — Random flip
# image = tf.image.random_flip_left_right(image)

# This randomly flips images horizontally.

# Example:

# Original:
# 🐴 →

# Flipped:
# ← 🐴

# This creates more training examples automatically.

# ================================================================

# ```python
# data = tfds.load(
#     'horses_or_humans',
#     split='train',
#     as_supervised=True
# )
# ```

### Explanation

# * `tfds.load()` → loads and prepares a dataset using TensorFlow Datasets (TFDS).
# * `'horses_or_humans'` → specifies the dataset name.
# * `split='train'` → loads only the training portion of the dataset.
# * `as_supervised=True` → returns data as `(image, label)` pairs instead of dictionary format MUCH EASIER FOR THE MACHINE TO UNDERSTAND.

# ### Output

# The variable `data` contains a TensorFlow dataset of horse and human images with their labels, ready for preprocessing and training.

# =====================================================================

