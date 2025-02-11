import tensorflow as tf
from tensorflow.keras import layers, models

def create_model(input_shape, num_classes=10):
    """Create CNN model for speech emotion recognition."""
    model = models.Sequential([
        # Reshape layer to convert 1D input to 2D
        layers.Reshape((input_shape, 1), input_shape=(input_shape,)),
        
        # First Convolutional Block
        layers.Conv1D(64, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(0.2),
        
        # Second Convolutional Block
        layers.Conv1D(128, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(0.2),
        
        # Third Convolutional Block
        layers.Conv1D(256, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(0.3),
        
        # Flatten and Dense Layers
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        
        # Output Layer
        layers.Dense(num_classes, activation='softmax')
    ])
    
    # Compile model
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def load_model(model_path):
    """Load trained model from file."""
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None

def save_model(model, model_path):
    """Save model to file."""
    try:
        model.save(model_path)
        return True
    except Exception as e:
        print(f"Error saving model: {str(e)}")
        return False 