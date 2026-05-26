import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # Image manager for neural networks (Loads,Resizes,Labels,Creates batches of images)

# Training Data Section 

train_datagen = ImageDataGenerator(
    rescale=1/255,# Neural networks work better with small numbers (0 - 1)
     
    #  ===================================
    # OPTIMIZATION TECHNIQUE
    # rotation_range=30,
    # width_shift_range=0.2,
    # height_shift_range=0.2,
    # shear_range=0.2,
    # zoom_range=0.2,
    # horizontal_flip=True,
    # =======================================

    ) 
train_generator = train_datagen.flow_from_directory(
    'datasets/horses-or-humans/train',
    target_size=(300, 300),
    batch_size=128,
    class_mode='binary'  # Binary classification (0 or 1)
)

# Instead of loading ALL images at once: TensorFlow loads: 128 images per step

#  Validation Data Section
validation_datagen = ImageDataGenerator(rescale=1/255)
validation_generator = validation_datagen.flow_from_directory(
    'datasets/horses-or-humans/validation',
    target_size=(300, 300),
    batch_size=32,
    class_mode='binary'  # Binary classification (0 or 1)
)

model = tf.keras.models.Sequential([

    tf.keras.layers.Conv2D(
        16,
        (3,3),
        activation='relu',
        input_shape=(300,300,3)
    ),

    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    # Flatten Layer => Convert 2D feature maps to 1D feature vectors because Dense Layer needs 1D
    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(128, activation='relu'), # 512 neurons increased model complexity and caused stronger overfitting on this small dataset 

    tf.keras.layers.Dense(1, activation='sigmoid')

])

# ============================================================

# Layer	What It Learns
# Conv2D 16	simple edges
# Conv2D 32	curves + textures
# Conv2D 64	object parts
# Dense layer	complete object understanding

# =============================================================

model.compile(
    optimizer='RMSprop',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# We Use this For CallBack to Save the Best Model
checkpoint = ModelCheckpoint(   
    "models/best_horse_human_model.keras",
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

history = model.fit(
    train_generator,
    epochs=15,
    validation_data=validation_generator,
    callbacks=[checkpoint]
)

# We Got Lower Accuracy Now But For Now We Will Use This Model
# We Can Always Improve This Model Later
