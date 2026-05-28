import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping

callback = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

train_dir = "datasets/horses-or-humans/train"
validation_dir = "datasets/horses-or-humans/validation"

train_datagen = ImageDataGenerator(

    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'

)

# ========================
# Without augmentation:
# AI memorizes images
# With augmentation:
# AI learns general patterns
# ========================

validation_datagen = ImageDataGenerator(
    rescale=1./255
)

train_generator = train_datagen.flow_from_directory(

    train_dir,
    target_size=(300,300),
    batch_size=32,
    class_mode='binary'
)

validation_generator = validation_datagen.flow_from_directory(

    validation_dir,
    target_size=(300,300),
    batch_size=32,
    class_mode='binary'
)


model = tf.keras.Sequential([

    tf.keras.Input(shape=(300,300,3)),

    tf.keras.layers.Conv2D(
        16,
        (3,3),
        activation='relu',
    ),

    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(
        32,
        (3,3),
        activation='relu'
    ),

    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(
        64,
        (3,3),
        activation='relu'
    ),

    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(
        64,
        (3,3),
        activation='relu'
    ),

    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')

])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=2,
    callbacks=[callback]
)

model.save('models/augmented-horses-or-humans/horses_or_humans_model.keras')









