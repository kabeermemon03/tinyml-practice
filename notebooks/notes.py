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