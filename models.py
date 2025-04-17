import tensorflow as tf
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import (
    Input, Dense, Conv1D, LSTM, Bidirectional, Dropout, 
    Reshape, Flatten, RepeatVector, TimeDistributed, GRU,
    MaxPooling1D, BatchNormalization, Concatenate
)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import numpy as np
import os
import time

class CNNBiLSTM:
    """
    CNN-BiLSTM model for Non-Intrusive Load Monitoring (NILM)
    Used for disaggregating total power consumption into individual appliances
    """
    def __init__(self, sequence_length=24, n_features=1, n_appliances=8, model_path=None):
        self.sequence_length = sequence_length
        self.n_features = n_features  # Input features (usually just total power)
        self.n_appliances = n_appliances  # Number of appliances to disaggregate
        self.model = None
        self.history = None
        self.model_path = model_path
        self.training_time = 0
        self.inference_time = 0
        
    def build_model(self):
        """
        Build the CNN-BiLSTM model architecture
        """
        # Input layer
        input_layer = Input(shape=(self.sequence_length, self.n_features))
        
        # CNN layers for feature extraction
        x = Conv1D(filters=32, kernel_size=3, activation='relu', padding='same')(input_layer)
        x = BatchNormalization()(x)
        x = MaxPooling1D(pool_size=2)(x)
        
        x = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = MaxPooling1D(pool_size=2)(x)
        
        # BiLSTM layers for sequence modeling
        x = Bidirectional(LSTM(64, return_sequences=True))(x)
        x = Dropout(0.2)(x)
        x = Bidirectional(LSTM(32, return_sequences=False))(x)
        x = Dropout(0.2)(x)
        
        # Output layers - one for each appliance
        outputs = []
        for i in range(self.n_appliances):
            output = Dense(24, activation='relu')(x)
            output = Dense(self.sequence_length, activation='relu', name=f'appliance_{i}')(output)
            outputs.append(output)
            
        # Create model
        self.model = Model(inputs=input_layer, outputs=outputs)
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        return self.model
    
    def train(self, X_train, y_train, X_val=None, y_val=None, epochs=20, batch_size=32):
        """
        Train the CNN-BiLSTM model
        
        Parameters:
        -----------
        X_train : np.array
            Shape (n_samples, sequence_length, n_features)
        y_train : list of np.array
            Each element should have shape (n_samples, sequence_length) for each appliance
        """
        if self.model is None:
            self.build_model()
        
        # Setup callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        ]
        
        # If model_path is provided, save the best model
        if self.model_path:
            callbacks.append(
                ModelCheckpoint(
                    filepath=self.model_path,
                    monitor='val_loss',
                    save_best_only=True,
                    verbose=1
                )
            )
        
        # Start timing
        start_time = time.time()
        
        # Train model
        validation_data = (X_val, y_val) if X_val is not None and y_val is not None else None
        self.history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=1
        )
        
        # End timing
        self.training_time = time.time() - start_time
        
        return self.history
    
    def predict(self, X):
        """
        Predict appliance-specific power consumption
        
        Parameters:
        -----------
        X : np.array
            Shape (n_samples, sequence_length, n_features)
            
        Returns:
        --------
        predictions : list of np.array
            Each element has shape (n_samples, sequence_length) for each appliance
        """
        if self.model is None:
            if self.model_path and os.path.exists(self.model_path):
                self.model = load_model(self.model_path)
            else:
                raise ValueError("Model not built or trained and no saved model found")
                
        # Start timing
        start_time = time.time()
        
        # Make predictions
        predictions = self.model.predict(X)
        
        # End timing
        self.inference_time = time.time() - start_time
        
        return predictions
    
    def load(self, model_path=None):
        """Load a pre-trained model"""
        path = model_path if model_path else self.model_path
        if path and os.path.exists(path):
            self.model = load_model(path)
            return True
        return False
    
    def save(self, model_path=None):
        """Save the trained model"""
        if self.model is None:
            return False
        
        path = model_path if model_path else self.model_path
        if path:
            self.model.save(path)
            return True
        return False
    
    def get_efficiency_metrics(self):
        """Return model efficiency metrics"""
        return {
            "model_type": "CNN-BiLSTM",
            "training_time_seconds": self.training_time,
            "inference_time_seconds": self.inference_time,
            "params_count": self.model.count_params() if self.model else 0,
            "model_size_mb": self._get_model_size_mb(),
        }
    
    def _get_model_size_mb(self):
        """Calculate approximate model size in MB"""
        if self.model is None:
            return 0
        # A rough estimate - each parameter is a 32-bit float (4 bytes)
        return self.model.count_params() * 4 / (1024 * 1024)


class AutoencoderGRU:
    """
    Autoencoder + GRU model for anomaly detection
    Used for detecting energy theft and grid faults
    """
    def __init__(self, sequence_length=24, n_features=15, latent_dim=8, model_path=None):
        self.sequence_length = sequence_length
        self.n_features = n_features  # Input features (power, voltage, current, etc.)
        self.latent_dim = latent_dim  # Size of the latent space
        self.model = None
        self.encoder = None
        self.decoder = None
        self.history = None
        self.model_path = model_path
        self.training_time = 0
        self.inference_time = 0
        self.threshold = None  # Anomaly threshold
        
    def build_model(self):
        """
        Build the Autoencoder + GRU model architecture
        """
        # Encoder
        inputs = Input(shape=(self.sequence_length, self.n_features))
        
        # GRU encoder
        encoded = GRU(64, return_sequences=True)(inputs)
        encoded = Dropout(0.2)(encoded)
        encoded = GRU(32, return_sequences=False)(encoded)
        encoded = Dense(self.latent_dim, activation='relu')(encoded)
        
        # Create encoder model
        self.encoder = Model(inputs=inputs, outputs=encoded, name='encoder')
        
        # Decoder
        decoder_input = Input(shape=(self.latent_dim,))
        
        # Reshape for GRU
        decoded = RepeatVector(self.sequence_length)(decoder_input)
        
        # GRU decoder
        decoded = GRU(32, return_sequences=True)(decoded)
        decoded = Dropout(0.2)(decoded)
        decoded = GRU(64, return_sequences=True)(decoded)
        decoded = TimeDistributed(Dense(self.n_features))(decoded)
        
        # Create decoder model
        self.decoder = Model(inputs=decoder_input, outputs=decoded, name='decoder')
        
        # Autoencoder (encoder + decoder)
        encoded_input = self.encoder(inputs)
        decoded_output = self.decoder(encoded_input)
        
        # Create autoencoder model
        self.model = Model(inputs=inputs, outputs=decoded_output, name='autoencoder')
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return self.model
    
    def train(self, X_train, X_val=None, epochs=20, batch_size=32):
        """
        Train the Autoencoder + GRU model
        
        Parameters:
        -----------
        X_train : np.array
            Shape (n_samples, sequence_length, n_features)
        X_val : np.array, optional
            Validation data with same shape as X_train
        """
        if self.model is None:
            self.build_model()
        
        # Setup callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss' if X_val is not None else 'loss', 
                          patience=5, restore_best_weights=True)
        ]
        
        # If model_path is provided, save the best model
        if self.model_path:
            callbacks.append(
                ModelCheckpoint(
                    filepath=self.model_path,
                    monitor='val_loss' if X_val is not None else 'loss',
                    save_best_only=True,
                    verbose=1
                )
            )
        
        # Start timing
        start_time = time.time()
        
        # In autoencoder training, we want to reconstruct the input
        validation_data = (X_val, X_val) if X_val is not None else None
        
        # Train model
        self.history = self.model.fit(
            X_train, X_train,  # Input and target are the same for autoencoders
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=1
        )
        
        # End timing
        self.training_time = time.time() - start_time
        
        # Calculate reconstruction error threshold for anomaly detection
        self._calculate_threshold(X_train)
        
        return self.history
    
    def _calculate_threshold(self, X_train, percentile=95):
        """
        Calculate reconstruction error threshold for anomaly detection
        
        Parameters:
        -----------
        X_train : np.array
            Shape (n_samples, sequence_length, n_features)
        percentile : float
            Percentile for threshold calculation (default: 95)
        """
        # Get reconstruction errors for training data
        predictions = self.model.predict(X_train)
        mae = np.mean(np.abs(X_train - predictions), axis=(1, 2))
        
        # Set threshold as the nth percentile of reconstruction errors
        self.threshold = np.percentile(mae, percentile)
        return self.threshold
    
    def detect_anomalies(self, X):
        """
        Detect anomalies in the data based on reconstruction error
        
        Parameters:
        -----------
        X : np.array
            Shape (n_samples, sequence_length, n_features)
            
        Returns:
        --------
        anomalies : np.array
            Binary array where 1 indicates an anomaly, shape (n_samples,)
        scores : np.array
            Anomaly scores (reconstruction errors), shape (n_samples,)
        """
        if self.model is None:
            if self.model_path and os.path.exists(self.model_path):
                self.model = load_model(self.model_path)
            else:
                raise ValueError("Model not built or trained and no saved model found")
        
        if self.threshold is None:
            # Use a default threshold if not calculated
            self.threshold = 0.1
        
        # Start timing
        start_time = time.time()
        
        # Get reconstructions
        reconstructions = self.model.predict(X)
        
        # Calculate reconstruction error for each sample
        mae = np.mean(np.abs(X - reconstructions), axis=(1, 2))
        
        # End timing
        self.inference_time = time.time() - start_time
        
        # Detect anomalies based on threshold
        anomalies = (mae > self.threshold).astype(int)
        
        return anomalies, mae
    
    def load(self, model_path=None):
        """Load a pre-trained model"""
        path = model_path if model_path else self.model_path
        if path and os.path.exists(path):
            self.model = load_model(path)
            
            # Recreate encoder and decoder from full model
            inputs = self.model.inputs
            encoded_layer = self.model.layers[-2].output  # Assuming encoder is the second-to-last layer
            decoded_layer = self.model.outputs[0]
            
            self.encoder = Model(inputs=inputs, outputs=encoded_layer)
            
            # For decoder, we need to find the right input shape
            decoder_input = Input(shape=(self.latent_dim,))
            # The rest would need to be reconstructed based on the full model architecture
            
            return True
        return False
    
    def save(self, model_path=None):
        """Save the trained model"""
        if self.model is None:
            return False
        
        path = model_path if model_path else self.model_path
        if path:
            self.model.save(path)
            return True
        return False
    
    def get_efficiency_metrics(self):
        """Return model efficiency metrics"""
        return {
            "model_type": "Autoencoder-GRU",
            "training_time_seconds": self.training_time,
            "inference_time_seconds": self.inference_time,
            "params_count": self.model.count_params() if self.model else 0,
            "model_size_mb": self._get_model_size_mb(),
        }
    
    def _get_model_size_mb(self):
        """Calculate approximate model size in MB"""
        if self.model is None:
            return 0
        # A rough estimate - each parameter is a 32-bit float (4 bytes)
        return self.model.count_params() * 4 / (1024 * 1024)


def preprocess_data_for_deep_learning(data, sequence_length=24, train_split=0.7, feature_cols=None):
    """
    Preprocess data for deep learning models
    
    Parameters:
    -----------
    data : pd.DataFrame
        The input data
    sequence_length : int
        The length of sequences for time series data
    train_split : float
        Proportion of data to use for training
    feature_cols : list
        List of feature column names to use
        
    Returns:
    --------
    dict containing preprocessed data for different models
    """
    result = {}
    
    # Default feature columns if none specified
    if feature_cols is None:
        feature_cols = ['active_power']
    
    # Create sequences for time series
    sequences = []
    appliance_sequences = []
    
    # Get appliance power columns
    appliance_cols = [col for col in data.columns if any(app.lower() in col.lower() for app in [
        'refrigerator', 'washing_machine', 'dishwasher', 'microwave', 
        'television', 'lighting', 'air_conditioner', 'computer'
    ])]
    
    # Convert data to numpy arrays
    X_data = data[feature_cols].values
    y_data = data[appliance_cols].values if appliance_cols else None
    
    # Create sequences
    for i in range(len(data) - sequence_length + 1):
        # Input sequence
        seq = X_data[i:i + sequence_length]
        sequences.append(seq)
        
        # Target appliance powers
        if y_data is not None:
            app_seq = []
            for app_idx in range(y_data.shape[1]):
                app_seq.append(y_data[i:i + sequence_length, app_idx])
            appliance_sequences.append(app_seq)
    
    # Convert to numpy arrays
    X = np.array(sequences)
    
    # For NILM model, we need appliance-specific targets
    if appliance_sequences:
        # Transpose to get list of arrays, one for each appliance
        y_nilm = []
        appliance_sequences = np.array(appliance_sequences)
        for app_idx in range(len(appliance_cols)):
            y_nilm.append(appliance_sequences[:, app_idx, :])
    else:
        y_nilm = None
    
    # Split into train and test
    split_idx = int(len(X) * train_split)
    
    # Data for CNN-BiLSTM (NILM)
    result['nilm'] = {
        'X_train': X[:split_idx], 
        'X_test': X[split_idx:],
        'y_train': [y[:split_idx] for y in y_nilm] if y_nilm else None,
        'y_test': [y[split_idx:] for y in y_nilm] if y_nilm else None,
        'appliance_names': [col.replace('_power', '') for col in appliance_cols]
    }
    
    # Data for Autoencoder-GRU (Anomaly Detection)
    # For anomaly detection, we need more features
    anomaly_cols = feature_cols + [
        'voltage', 'current', 'power_factor', 'frequency', 
        'thd', 'temperature'
    ]
    anomaly_cols = [col for col in anomaly_cols if col in data.columns]
    
    # Create sequences for anomaly detection
    anomaly_sequences = []
    for i in range(len(data) - sequence_length + 1):
        seq = data[anomaly_cols].values[i:i + sequence_length]
        anomaly_sequences.append(seq)
    
    X_anomaly = np.array(anomaly_sequences)
    
    # For training anomaly detection model, we only want normal data
    # Filter out known anomalies (theft_flag or fault_flag = 1)
    if 'theft_flag' in data.columns and 'fault_flag' in data.columns:
        normal_indices = []
        anomaly_indices = []
        
        for i in range(len(data) - sequence_length + 1):
            # If any point in sequence has an anomaly, consider whole sequence anomalous
            if (data['theft_flag'].values[i:i + sequence_length].sum() > 0 or 
                data['fault_flag'].values[i:i + sequence_length].sum() > 0):
                anomaly_indices.append(i)
            else:
                normal_indices.append(i)
        
        # Split normal data into train/test
        normal_split_idx = int(len(normal_indices) * train_split)
        train_indices = normal_indices[:normal_split_idx]
        test_normal_indices = normal_indices[normal_split_idx:]
        
        # Create labeled test set with both normal and anomaly data
        test_indices = test_normal_indices + anomaly_indices
        test_labels = np.zeros(len(test_indices))
        test_labels[len(test_normal_indices):] = 1  # Anomalies are labeled 1
        
        # Create the final datasets
        X_train_anomaly = X_anomaly[train_indices]
        X_test_anomaly = X_anomaly[test_indices]
        y_test_anomaly = test_labels
        
    else:
        # If no anomaly flags, just split normally
        split_idx = int(len(X_anomaly) * train_split)
        X_train_anomaly = X_anomaly[:split_idx]
        X_test_anomaly = X_anomaly[split_idx:]
        y_test_anomaly = None
    
    result['anomaly'] = {
        'X_train': X_train_anomaly,
        'X_test': X_test_anomaly,
        'y_test': y_test_anomaly,
        'feature_names': anomaly_cols
    }
    
    return result
