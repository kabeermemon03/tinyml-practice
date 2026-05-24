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