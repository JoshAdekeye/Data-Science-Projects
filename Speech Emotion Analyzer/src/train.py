import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import pandas as pd

from utils import load_audio, extract_features
from model import create_model, save_model

def prepare_dataset(data_dir):
    """Prepare dataset from audio files."""
    features = []
    labels = []
    
    # Process RAVDESS dataset
    ravdess_dir = os.path.join(data_dir, 'RAVDESS', 'Audio_Song_Actors_01-24')
    if os.path.exists(ravdess_dir):
        for actor_dir in os.listdir(ravdess_dir):
            if actor_dir.startswith('Actor_'):
                actor_path = os.path.join(ravdess_dir, actor_dir)
                for filename in os.listdir(actor_path):
                    if filename.endswith(".wav"):
                        try:
                            # Extract emotion from filename (7th character)
                            # RAVDESS emotion mapping: 1=neutral, 2=calm, 3=happy, 4=sad, 5=angry, 6=fearful, 7=disgust, 8=surprised
                            emotion_num = int(filename[6])
                            # Map RAVDESS emotions to our 0-4 range
                            emotion_map = {
                                1: 0,  # neutral -> neutral
                                2: 0,  # calm -> neutral
                                3: 1,  # happy -> happy
                                4: 2,  # sad -> sad
                                5: 3,  # angry -> angry
                                6: 4,  # fearful -> fearful
                                7: 3,  # disgust -> angry (combining similar emotions)
                                8: 4   # surprised -> fearful (combining similar emotions)
                            }
                            emotion = emotion_map.get(emotion_num, 0)  # default to neutral if unknown
                            
                            # Determine gender (first 12 actors are female)
                            actor_num = int(actor_dir.split('_')[1])
                            gender = 0 if actor_num <= 12 else 1  # 0 for female, 1 for male
                            
                            # Calculate final label (0-4 for female, 5-9 for male)
                            final_label = emotion + (gender * 5)
                            
                            # Load and extract features
                            file_path = os.path.join(actor_path, filename)
                            data, sample_rate = load_audio(file_path)
                            if data is not None:
                                file_features = extract_features(data, sample_rate)
                                if file_features is not None:
                                    features.append(file_features)
                                    labels.append(final_label)
                        except (ValueError, IndexError) as e:
                            print(f"Error processing file {filename}: {str(e)}")
                            continue
    
    # Process SAVEE dataset
    savee_dir = os.path.join(data_dir, 'SAVEE')
    if os.path.exists(savee_dir):
        # SAVEE emotions are already male and mapped correctly (5-9)
        emotion_map = {
            'n': 5,  # neutral
            'h': 6,  # happy
            'sa': 7, # sad
            'a': 8,  # angry
            'f': 9   # fearful
        }
        for filename in os.listdir(savee_dir):
            if filename.endswith(".wav"):
                try:
                    # Get emotion from filename prefix
                    emotion_code = filename.split('_')[0].lower()
                    emotion = emotion_map.get(emotion_code)
                    if emotion is not None:
                        file_path = os.path.join(savee_dir, filename)
                        data, sample_rate = load_audio(file_path)
                        if data is not None:
                            file_features = extract_features(data, sample_rate)
                            if file_features is not None:
                                features.append(file_features)
                                labels.append(emotion)
                except Exception as e:
                    print(f"Error processing SAVEE file {filename}: {str(e)}")
                    continue
    
    if len(features) == 0:
        print("No audio files found. Please check the following directories:")
        print(f"RAVDESS: {ravdess_dir}")
        print(f"SAVEE: {savee_dir}")
        return None, None
    
    # Convert to numpy arrays
    features = np.array(features)
    labels = np.array(labels)
    
    print(f"Dataset statistics:")
    print(f"Total samples: {len(features)}")
    print("Label distribution:")
    unique, counts = np.unique(labels, return_counts=True)
    for label, count in zip(unique, counts):
        print(f"Label {label}: {count} samples")
    
    return features, labels

def train_model(data_dir, model_save_path, epochs=50, batch_size=32):
    """Train the emotion recognition model."""
    # Prepare dataset
    print("Preparing dataset...")
    features, labels = prepare_dataset(data_dir)
    
    if features is None or labels is None:
        print("No data found in the specified directories. Please ensure the dataset is properly organized:")
        print("- RAVDESS dataset should be in: data/RAVDESS/Audio_Song_Actors_01-24/")
        print("- SAVEE dataset should be in: data/SAVEE/")
        return None, None
    
    print(f"Dataset prepared successfully! Found {len(features)} audio samples.")
    
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42
    )
    
    # Create and compile model
    print("Creating model...")
    model = create_model(input_shape=features.shape[1])
    
    # Define callbacks
    callbacks = [
        ModelCheckpoint(
            os.path.join('saved_models', 'best_model.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        )
    ]
    
    # Train model
    print("Training model...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save final model
    if model_save_path:
        save_model(model, model_save_path)
    
    # Evaluate model
    print("\nEvaluating model...")
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test accuracy: {accuracy*100:.2f}%")
    
    return model, history

if __name__ == "__main__":
    # Set paths
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    model_save_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                                  'saved_models', 'emotion_model.h5')
    
    # Create saved_models directory if it doesn't exist
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    
    # Train model
    print("Starting training process...")
    print(f"Looking for datasets in: {data_dir}")
    model, history = train_model(data_dir, model_save_path)
    
    if model is not None:
        print("\nTraining completed successfully!")
        print(f"Model saved to: {model_save_path}")
    else:
        print("\nTraining failed. Please check the dataset organization and try again.") 