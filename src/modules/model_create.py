import os
import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers, models
from tensorflow.keras.layers import Input, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix, classification_report

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
    print(f"\n{Config.YELLOW}Loading dataset...{Config.RESET}")
    dataset, info = tfds.load('malaria', as_supervised=True, with_info=True)
    
    # Get the total number of samples
    total_samples = info.splits['train'].num_examples
    print(f"\n{Config.BLUE}Total samples: {total_samples}{Config.RESET}")

    # Calculate split sizes
    train_size = int(0.9 * total_samples)
    test_size = total_samples - train_size
    print(f"\n{Config.BLUE}Train size:{Config.RESET} {train_size}")
    print(f"{Config.BLUE}Test size:{Config.RESET} {test_size}")

    # Split the dataset into training and testing sets
    train_dataset = dataset['train'].take(train_size)
    test_dataset = dataset['train'].skip(train_size)

    # Apply preprocessing and batching
    print(f"\n{Config.YELLOW}Preprocessing and batching...{Config.RESET}")
    train_dataset = train_dataset.map(preprocess).batch(BATCH_SIZE).shuffle(1000)
    test_dataset = test_dataset.map(preprocess).batch(BATCH_SIZE)

    # Define a CNN model
    model = models.Sequential([
        Input(shape=(IMG_SIZE, IMG_SIZE, 3)),  # Define the input shape explicitly
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        Dropout(0.5),  # Add dropout after the first MaxPooling layer
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        Dropout(0.5),  # Add dropout after the second MaxPooling layer
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        Dropout(0.5),  # Add dropout before the final dense layer
        layers.Dense(1, activation='sigmoid')  # Binary classification
    ])

    # Compile the model
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    
    # Early stopping to prevent overfitting
    early_stopping = EarlyStopping(
        monitor='val_loss',    # Surveille la perte de validation
        patience=3,            # Arrête l’entraînement après 3 epochs sans amélioration
        restore_best_weights=True  # Restaure les poids du meilleur modèle
    )

    # Train the model
    print(f"\n{Config.YELLOW}Training model...{Config.RESET}")
    history = model.fit(
        train_dataset,
        batch_size=BATCH_SIZE,
        epochs=20,
        validation_data=test_dataset,
        callbacks=[early_stopping]
    )
    
    # Save the model
    print(f"\n{Config.YELLOW}Saving model...{Config.RESET}")
    model.save(Config.MODEL_PATH)
    
    # Evaluate the model on the test dataset
    print(f"\n{Config.YELLOW}Evaluating model...{Config.RESET}")
    y_true = []
    y_pred = []
    wrong_predictions = []

    for images, labels in test_dataset:
        # Get true labels
        true_labels = labels.numpy()
        # Get predicted probabilities
        pred_probs = model.predict(images)
        # Convert probabilities to binary predictions
        pred_labels = (pred_probs > 0.5).astype(int).flatten()

        # Append results
        y_true.extend(true_labels)
        y_pred.extend(pred_labels)

        # Log wrong predictions
        for i in range(len(true_labels)):
            if true_labels[i] != pred_labels[i]:
                wrong_predictions.append({
                    "True Label": true_labels[i],
                    "Predicted Label": pred_labels[i],
                    "Predicted Probability": pred_probs[i][0]
                })

    # Convert to NumPy arrays for metrics
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Generate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print(f"\n{Config.BLUE}Confusion Matrix:{Config.RESET}")
    print(cm)

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Uninfected', 'Parasitized'], 
                yticklabels=['Uninfected', 'Parasitized'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()

    # Generate classification report
    print(f"\n{Config.BLUE}Classification Report:{Config.RESET}")
    print(classification_report(y_true, y_pred, target_names=['Uninfected', 'Parasitized']))

    # Display wrong predictions
    if wrong_predictions:
        print(f"\n{Config.RED}Wrong Predictions:{Config.RESET}")
        wrong_df = pd.DataFrame(wrong_predictions)
        print(wrong_df)
    else:
        print(f"\n{Config.GREEN}No wrong predictions!{Config.RESET}")

create_model()