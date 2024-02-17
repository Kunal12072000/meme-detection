import os

import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
import sys
import matplotlib.pyplot as plt
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout


# Define model
def define_model():
    model = Sequential()

    # Convolutional layer with 32 filters, a 3x3 kernel, and ReLU activation
    model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(224, 224, 3)))
    # Max pooling layer with 2x2 pool size
    model.add(MaxPooling2D((2, 2)))

    # Convolutional layer with 64 filters, a 3x3 kernel, and ReLU activation
    model.add(Conv2D(128, (3, 3), activation='relu'))
    # Max pooling layer with 2x2 pool size
    model.add(MaxPooling2D((2, 2)))

    # Convolutional layer with 128 filters, a 3x3 kernel, and ReLU activation
    model.add(Conv2D(256, (3, 3), activation='relu'))
    # Max pooling layer with 2x2 pool size
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(1024, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    # Flatten the output before feeding it into the fully connected layers
    model.add(Flatten())

    # Fully connected layer with 128 units and ReLU activation
    model.add(Dense(128, activation='relu'))
    # Dropout layer for regularization
    model.add(Dropout(0.5))

    # Output layer with 1 unit and sigmoid activation for binary classification
    model.add(Dense(1, activation='sigmoid'))
    return model


# plot diagnostic learning curves
def summarize_diagnostics(history):
    plt.figure(figsize=(12, 6))

    # Plot training & validation loss values
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Validation'], loc='upper right')

    # Plot training & validation accuracy values
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Validation'], loc='lower right')

    plt.tight_layout()
    plt.savefig('learning_curves.jpg')
    plt.show()


# Define data directories
train_dir = os.getcwd() + r"\src\data\train"
val_dir = os.getcwd() + r"\src\data\validation"
test_dir = os.getcwd() + r"\src\data\test"

# Data preprocessing and augmentation
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1. / 255)

# Load and augment training data
train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)

print(train_data)
# Load validation data
val_data = test_datagen.flow_from_directory(
    val_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)

# Load test data
test_data = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)

model = define_model()
# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
print(f'Model Summary:', model.summary())
# Train the model
history = model.fit(train_data, epochs=30, validation_data=val_data)

# Evaluate the model
test_loss, test_acc = model.evaluate(test_data)
print(f'Test Accuracy: {test_acc}')
summarize_diagnostics(history)
