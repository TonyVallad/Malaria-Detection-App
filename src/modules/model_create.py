import os
import sys
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers, models

# Adjust the path to access the config file from the root directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from config import Config

IMG_SIZE = 128  # Resize all images to 128x128
BATCH_SIZE = 32

def preprocess(image, label):
        # Resize the image and normalize pixel values
        image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
        image = image / 255.0  # Normalize to [0, 1]
        return image, label

def create_model():
    # Load the Malaria dataset
    dataset, info = tfds.load('malaria', split=['train'], as_supervised=True, with_info=True)
    
    # Unpack the train split
    train_dataset = dataset[0]

    # Print dataset information
    print(info)
    
    # Example: Visualize some samples from the dataset
    import matplotlib.pyplot as plt

    for image, label in train_dataset.take(5):  # Display 5 samples
        plt.imshow(image.numpy(), cmap='gray')
        plt.title(f"Label: {'Parasitized' if label.numpy() == 1 else 'Uninfected'}")
        plt.axis('off')
        plt.show()

    # Apply preprocessing and batching
    train_dataset = train_dataset.map(preprocess).batch(BATCH_SIZE).shuffle(1000)

    # Verify preprocessing
    for images, labels in train_dataset.take(1):
        print(f"Batch shape: {images.shape}, Labels: {labels.numpy()}")
    
    # Define a CNN model
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(1, activation='sigmoid')  # Binary classification
    ])

    # Compile the model
    model.compile(optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy'])

    # Train the model
    model.fit(train_dataset, epochs=10)
    
    # Save the model
    model.save(Config.MODEL_PATH)
    # model.save('malaria_model.h5')


create_model()