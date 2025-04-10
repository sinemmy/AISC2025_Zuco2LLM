from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

def train_encoding_models(layer_to_word_embeddings, eeg_features):
    """
    Train encoding models to predict EEG features from embeddings at each layer
    
    Args:
        layer_to_word_embeddings: Dictionary mapping layer indices to lists of (word, embedding) tuples
        eeg_features: Dictionary mapping words to EEG features
        
    Returns:
        Dictionary mapping layer indices to trained encoding models
    """
    layer_to_model = {}
    layer_to_performance = {}
    
    for layer_idx, word_embeddings in layer_to_word_embeddings.items():
        # Extract words and embeddings
        words = [w for w, _ in word_embeddings]
        embeddings = np.array([e for _, e in word_embeddings])
        
        # Get corresponding EEG features
        features = np.array([eeg_features[w] for w in words])
        
        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            embeddings, features, test_size=0.2, random_state=42)
        
        # Train a ridge regression model
        model = Ridge(alpha=1.0)
        model.fit(X_train, y_train)
        
        # Evaluate the model
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        
        # Store the model and its performance
        layer_to_model[layer_idx] = model
        layer_to_performance[layer_idx] = mse
        
        print(f"Layer {layer_idx}: MSE = {mse:.4f}")
    
    return layer_to_model, layer_to_performance