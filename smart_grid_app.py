import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.colors as mcolors
import altair as alt
from datetime import datetime, timedelta
import os
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, IsolationForest
from sklearn.neural_network import MLPRegressor, MLPClassifier
import xgboost as xgb
import time
import threading
from io import BytesIO
import base64
import seaborn as sns
from matplotlib.cm import get_cmap
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, r2_score

# ------------------------------------
# Streamlit Setup
# ------------------------------------
st.set_page_config(
    page_title="Smart Grid Energy Monitoring",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
APPLIANCES = [
    "Refrigerator", "Washing Machine", "Dishwasher", "Microwave", 
    "Television", "Lighting", "Air Conditioner", "Computer"
]

# Define appliance colors for consistent visualization
APPLIANCE_COLORS = {
    "Refrigerator": "#1f77b4",
    "Washing Machine": "#ff7f0e",
    "Dishwasher": "#2ca02c",
    "Microwave": "#d62728",
    "Television": "#9467bd",
    "Lighting": "#8c564b",
    "Air Conditioner": "#e377c2",
    "Computer": "#7f7f7f"
}

# ------------------------------------
# MLP-based Model for NILM (replacing CNN+BiLSTM)
# ------------------------------------
class MLPNILMModel:
    """
    MLP model for Non-Intrusive Load Monitoring (NILM)
    Used for disaggregating power consumption appliance-wise
    """
    def __init__(self, sequence_length=24, n_features=1, n_appliances=8):
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.n_appliances = n_appliances
        self.models = []
        self.training_time = 0
        self.inference_time = 0
        self.model_size = 0
        self.history = {'loss': [], 'val_loss': [], 'mae': [], 'val_mae': []}
    
    def build_model(self):
        """Build MLP models - one for each appliance"""
        for i in range(self.n_appliances):
            model = MLPRegressor(
                hidden_layer_sizes=(64, 32),
                activation='relu',
                solver='adam',
                max_iter=200,
                alpha=0.0001,
                learning_rate='adaptive',
                random_state=42
            )
            self.models.append(model)
        return self.models
    
    def train(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=32, verbose=1):
        """Train the models and record training time"""
        if not self.models:
            self.build_model()
            
        # Record training time
        start_time = time.time()
        
        # Flatten the input for MLPRegressor
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        X_val_flat = X_val.reshape(X_val.shape[0], -1)
        
        # Train one model for each appliance
        for i in range(self.n_appliances):
            # Get target for this appliance
            if isinstance(y_train, list):
                y_train_app = y_train[i]
            else:
                y_train_app = y_train[:, i]
                
            if isinstance(y_val, list):
                y_val_app = y_val[i]
            else:
                y_val_app = y_val[:, i]
            
            # Train model
            self.models[i].fit(X_train_flat, y_train_app)
            
            # Simulate history for plotting
            train_pred = self.models[i].predict(X_train_flat)
            val_pred = self.models[i].predict(X_val_flat)
            
            train_loss = mean_squared_error(y_train_app, train_pred)
            val_loss = mean_squared_error(y_val_app, val_pred)
            
            train_mae = np.mean(np.abs(y_train_app - train_pred))
            val_mae = np.mean(np.abs(y_val_app - val_pred))
            
            self.history['loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['mae'].append(train_mae)
            self.history['val_mae'].append(val_mae)
        
        self.training_time = time.time() - start_time
        self.model_size = sum(model.coefs_[0].size for model in self.models)
        
        return self.history
    
    def predict(self, X):
        """Make predictions and record inference time"""
        if not self.models:
            raise ValueError("Models not trained. Call train() first.")
            
        start_time = time.time()
        
        # Flatten the input for MLPRegressor
        X_flat = X.reshape(X.shape[0], -1)
        
        # Make predictions for each appliance
        predictions = []
        for i in range(self.n_appliances):
            pred = self.models[i].predict(X_flat)
            predictions.append(pred)
        
        self.inference_time = time.time() - start_time
        
        # Stack predictions to match expected output format
        predictions = np.column_stack(predictions)
        
        return predictions
    
    def get_efficiency_metrics(self):
        """Return efficiency metrics for the model"""
        return {
            "training_time_seconds": self.training_time,
            "inference_time_seconds": self.inference_time,
            "model_size_parameters": self.model_size,
            "model_memory_mb": self.model_size * 4 / (1024 * 1024)  # Approx memory in MB
        }
    
    def save(self, path="models/mlp_nilm.joblib"):
        """Save the model to disk"""
        if not self.models:
            raise ValueError("No models to save. Train the models first.")
            
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.models, path)
        
    def load(self, path="models/mlp_nilm.joblib"):
        """Load the model from disk"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"No model found at {path}")
            
        self.models = joblib.load(path)

# ------------------------------------
# Isolation Forest for Anomaly Detection (replacing Autoencoder+GRU)
# ------------------------------------
class AnomalyDetectionModel:
    """
    Isolation Forest model for Energy Theft and Smart Grid Fault detection
    """
    def __init__(self, sequence_length=24, n_features=1):
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.model = None
        self.feature_extractor = None
        self.training_time = 0
        self.inference_time = 0
        self.model_size = 0
        self.threshold = None
        self.history = None
    
    def build_model(self):
        """Build Isolation Forest model"""
        self.model = IsolationForest(
            n_estimators=100,
            max_samples='auto',
            contamination=0.05,  # Expected proportion of anomalies
            random_state=42
        )
        
        # Simple feature extractor (PCA-like dimensionality reduction)
        self.feature_extractor = RandomForestRegressor(
            n_estimators=10,
            max_depth=5,
            random_state=42
        )
        
        return self.model
    
    def train(self, X_train, epochs=50, batch_size=32, validation_split=0.2, verbose=1):
        """Train the model and record training time"""
        if self.model is None:
            self.build_model()
            
        # Record training time
        start_time = time.time()
        
        # Flatten the input data
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        
        # Create synthetic targets for feature extractor
        # Just use the sum of each sample as a simple target
        synthetic_targets = X_train_flat.sum(axis=1)
        
        # First train feature extractor
        self.feature_extractor.fit(X_train_flat, synthetic_targets)
        
        # Extract features - just use the feature importances
        feature_importances = self.feature_extractor.feature_importances_
        
        # Apply feature importance weighting
        X_train_weighted = X_train_flat * feature_importances
        
        # Train anomaly detector
        self.model.fit(X_train_weighted)
        
        self.training_time = time.time() - start_time
        
        # Approximate model size (number of trees * average tree nodes)
        self.model_size = self.model.n_estimators * 100  # rough estimate
        
        # Set threshold based on training data
        scores = self.model.decision_function(X_train_weighted)
        self.threshold = np.percentile(scores, 5)  # 5th percentile
        
        return None  # No history for this model
    
    def detect_anomalies(self, X):
        """Detect anomalies based on isolation scores"""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
            
        start_time = time.time()
        
        # Flatten the input data
        X_flat = X.reshape(X.shape[0], -1)
        
        # Apply feature importance weighting
        feature_importances = self.feature_extractor.feature_importances_
        X_weighted = X_flat * feature_importances
        
        # Get anomaly scores (-1 for anomalies, 1 for normal)
        raw_scores = self.model.decision_function(X_weighted)
        
        # Convert to reconstruction error-like format (lower is normal, higher is anomaly)
        reconstruction_errors = -raw_scores
        
        # Detect anomalies based on threshold
        anomalies = reconstruction_errors > -self.threshold
        
        self.inference_time = time.time() - start_time
        
        return {
            'predictions': None,  # No reconstructions for Isolation Forest
            'reconstruction_errors': reconstruction_errors,
            'anomalies': anomalies,
            'anomaly_scores': reconstruction_errors / -self.threshold  # Normalized scores
        }
    
    def extract_features(self, X):
        """Extract features using the feature extractor"""
        if self.feature_extractor is None:
            raise ValueError("Feature extractor not created. Build or load the model first.")
            
        # Flatten the input data
        X_flat = X.reshape(X.shape[0], -1)
        
        # Use feature importances as a simple feature extraction
        feature_importances = self.feature_extractor.feature_importances_
        features = X_flat * feature_importances
        
        # Reduce to 8 dimensions by aggregating
        chunk_size = features.shape[1] // 8
        extracted_features = np.array([
            features[:, i:i+chunk_size].mean(axis=1) 
            for i in range(0, features.shape[1], chunk_size)
        ]).T
        
        # Ensure we have exactly 8 dimensions
        if extracted_features.shape[1] > 8:
            extracted_features = extracted_features[:, :8]
        elif extracted_features.shape[1] < 8:
            # Pad with zeros if needed
            padding = np.zeros((extracted_features.shape[0], 8 - extracted_features.shape[1]))
            extracted_features = np.hstack([extracted_features, padding])
        
        return extracted_features
    
    def get_efficiency_metrics(self):
        """Return efficiency metrics for the model"""
        return {
            "training_time_seconds": self.training_time,
            "inference_time_seconds": self.inference_time,
            "model_size_parameters": self.model_size,
            "model_memory_mb": self.model_size * 4 / (1024 * 1024)  # Approx memory in MB
        }
    
    def save(self, path="models/anomaly_detector.joblib"):
        """Save the model to disk"""
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")
            
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        models_dict = {
            'model': self.model,
            'feature_extractor': self.feature_extractor,
            'threshold': self.threshold
        }
        
        joblib.dump(models_dict, path)
        
    def load(self, path="models/anomaly_detector.joblib"):
        """Load the model from disk"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"No model found at {path}")
            
        models_dict = joblib.load(path)
        self.model = models_dict['model']
        self.feature_extractor = models_dict['feature_extractor']
        self.threshold = models_dict['threshold']

# ------------------------------------
# Decision Models (Random Forest and XGBoost)
# ------------------------------------
class DecisionModel:
    """
    Decision making model using Random Forest and XGBoost for classification, regression, and decision making
    """
    def __init__(self, model_type="classification", algorithm="rf"):
        """
        Initialize decision model
        
        Args:
            model_type: 'classification' or 'regression'
            algorithm: 'rf' (Random Forest) or 'xgb' (XGBoost)
        """
        self.model_type = model_type
        self.algorithm = algorithm
        self.model = None
        self.feature_importance = None
        self.training_time = 0
        self.inference_time = 0
    
    def build_model(self):
        """
        Build the selected model type
        """
        if self.model_type == "classification":
            if self.algorithm == "rf":
                self.model = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    min_samples_split=5,
                    random_state=42
                )
            elif self.algorithm == "xgb":
                self.model = xgb.XGBClassifier(
                    n_estimators=100,
                    max_depth=5,
                    learning_rate=0.1,
                    random_state=42
                )
        elif self.model_type == "regression":
            if self.algorithm == "rf":
                self.model = RandomForestRegressor(
                    n_estimators=100,
                    max_depth=10,
                    min_samples_split=5,
                    random_state=42
                )
            elif self.algorithm == "xgb":
                self.model = xgb.XGBRegressor(
                    n_estimators=100,
                    max_depth=5,
                    learning_rate=0.1,
                    random_state=42
                )
        
        return self.model
    
    def train(self, X_train, y_train):
        """
        Train the model
        
        Args:
            X_train: Training features
            y_train: Training targets
        """
        if self.model is None:
            self.build_model()
        
        # Record training time
        start_time = time.time()
        
        # Train model
        self.model.fit(X_train, y_train)
        
        self.training_time = time.time() - start_time
        
        # Get feature importance
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = self.model.feature_importances_
        
        return self.model
    
    def predict(self, X):
        """
        Make predictions with the model
        
        Args:
            X: Features for prediction
            
        Returns:
            Predictions
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Record inference time
        start_time = time.time()
        
        if self.model_type == "classification":
            predictions = self.model.predict(X)
            proba = None
            if hasattr(self.model, 'predict_proba'):
                proba = self.model.predict_proba(X)
        else:  # regression
            predictions = self.model.predict(X)
            proba = None
        
        self.inference_time = time.time() - start_time
        
        return {
            'predictions': predictions,
            'probabilities': proba
        }
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate the model performance
        
        Args:
            X_test: Test features
            y_test: Test targets
            
        Returns:
            Dictionary of performance metrics
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Make predictions
        result = self.predict(X_test)
        predictions = result['predictions']
        
        # Calculate metrics based on model type
        metrics = {}
        
        if self.model_type == "classification":
            metrics['accuracy'] = accuracy_score(y_test, predictions)
            metrics['precision'] = precision_score(y_test, predictions, average='weighted')
            metrics['recall'] = recall_score(y_test, predictions, average='weighted')
            metrics['f1'] = f1_score(y_test, predictions, average='weighted')
        else:  # regression
            metrics['mse'] = mean_squared_error(y_test, predictions)
            metrics['rmse'] = np.sqrt(metrics['mse'])
            metrics['r2'] = r2_score(y_test, predictions)
        
        # Add training and inference times
        metrics['training_time'] = self.training_time
        metrics['inference_time'] = self.inference_time
        
        return metrics
    
    def plot_feature_importance(self, feature_names=None, figsize=(10, 6)):
        """
        Plot feature importance
        
        Args:
            feature_names: Names of features
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        if self.feature_importance is None:
            raise ValueError("No feature importance available. Train a model that provides feature importance.")
        
        if feature_names is None:
            feature_names = [f"Feature {i}" for i in range(len(self.feature_importance))]
        
        # Sort features by importance
        indices = np.argsort(self.feature_importance)[::-1]
        sorted_feature_names = [feature_names[i] for i in indices]
        sorted_importance = self.feature_importance[indices]
        
        # Plot
        fig, ax = plt.subplots(figsize=figsize)
        ax.bar(range(len(sorted_importance)), sorted_importance, align="center")
        ax.set_xticks(range(len(sorted_importance)))
        ax.set_xticklabels(sorted_feature_names, rotation=90)
        ax.set_title(f"Feature Importance ({self.algorithm.upper()} {self.model_type})")
        ax.set_ylabel("Importance")
        plt.tight_layout()
        
        return fig
    
    def save(self, path=None):
        """
        Save the model to disk
        
        Args:
            path: Path to save the model
        """
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")
        
        if path is None:
            path = f"models/{self.algorithm}_{self.model_type}.joblib"
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save model
        joblib.dump(self.model, path)
        
        # Save metadata
        metadata = {
            'model_type': self.model_type,
            'algorithm': self.algorithm,
            'feature_importance': self.feature_importance,
            'training_time': self.training_time
        }
        joblib.dump(metadata, path.replace('.joblib', '_metadata.joblib'))
    
    def load(self, path=None):
        """
        Load the model from disk
        
        Args:
            path: Path to load the model from
        """
        if path is None:
            path = f"models/{self.algorithm}_{self.model_type}.joblib"
        
        if not os.path.exists(path):
            raise FileNotFoundError(f"No model found at {path}")
        
        # Load model
        self.model = joblib.load(path)
        
        # Load metadata
        metadata_path = path.replace('.joblib', '_metadata.joblib')
        if os.path.exists(metadata_path):
            metadata = joblib.load(metadata_path)
            self.model_type = metadata['model_type']
            self.algorithm = metadata['algorithm']
            self.feature_importance = metadata['feature_importance']
            self.training_time = metadata['training_time']

# ------------------------------------
# Data Processor
# ------------------------------------
class DataProcessor:
    """
    Handle data preprocessing for energy monitoring and disaggregation
    """
    def __init__(self, appliance_columns=None):
        self.scalers = {}
        self.sequence_length = 24  # Default sequence length (24 hours)
        self.appliance_columns = appliance_columns or []
        self.feature_columns = []
    
    def preprocess_data(self, df, house_id=None, appliance_columns=None, feature_columns=None, 
                         sequence_length=24, test_size=0.2, scaling=True):
        """
        Preprocess data for machine learning models
        
        Args:
            df: DataFrame containing energy data
            house_id: ID of the house to filter data for (optional)
            appliance_columns: Column names for appliance power consumption
            feature_columns: Column names for input features
            sequence_length: Length of sequence for time series data
            test_size: Proportion of data to use for testing
            scaling: Whether to apply Min-Max scaling
            
        Returns:
            Processed data ready for model training/inference
        """
        # Filter by house if specified
        if house_id is not None:
            df = df[df['house_id'] == house_id].copy()
        
        # Set default columns if not provided
        if appliance_columns is None:
            appliance_columns = self.appliance_columns
        else:
            self.appliance_columns = appliance_columns
            
        if feature_columns is None:
            # Default to using total power as feature if not specified
            feature_columns = ['total_power']
        
        self.feature_columns = feature_columns
        self.sequence_length = sequence_length
        
        # Apply scaling if requested
        if scaling:
            # Scale features
            for col in feature_columns:
                if col not in self.scalers:
                    self.scalers[col] = MinMaxScaler()
                df[f'{col}_scaled'] = self.scalers[col].fit_transform(df[col].values.reshape(-1, 1))
            
            # Scale appliance power
            for col in appliance_columns:
                if col not in self.scalers:
                    self.scalers[col] = MinMaxScaler()
                df[f'{col}_scaled'] = self.scalers[col].fit_transform(df[col].values.reshape(-1, 1))
            
            # Use scaled columns
            scaled_feature_cols = [f'{col}_scaled' for col in feature_columns]
            scaled_appliance_cols = [f'{col}_scaled' for col in appliance_columns]
        else:
            # Use original columns without scaling
            scaled_feature_cols = feature_columns
            scaled_appliance_cols = appliance_columns
        
        # Create sequences for time series
        X, y = self._create_sequences(df, scaled_feature_cols, scaled_appliance_cols, sequence_length)
        
        # Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)
        
        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'df': df,
            'feature_columns': feature_columns,
            'appliance_columns': appliance_columns
        }
    
    def _create_sequences(self, df, feature_cols, appliance_cols, sequence_length):
        """
        Create sequences for time series prediction
        
        Args:
            df: DataFrame with preprocessed data
            feature_cols: Columns to use as features
            appliance_cols: Columns to predict (appliance power)
            sequence_length: Length of input sequences
            
        Returns:
            X: Input sequences
            y: Target values
        """
        X, y = [], []
        
        for i in range(len(df) - sequence_length):
            # Get sequence of features
            X_sequence = df[feature_cols].iloc[i:i+sequence_length].values
            
            # Get target appliance values (at the end of sequence)
            y_values = df[appliance_cols].iloc[i+sequence_length].values
            
            X.append(X_sequence)
            y.append(y_values)
        
        return np.array(X), np.array(y)
    
    def create_anomaly_detection_data(self, df, house_id=None, feature_columns=None, sequence_length=24, scaling=True):
        """
        Preprocess data specifically for anomaly detection
        
        Args:
            df: DataFrame containing energy data
            house_id: ID of the house to filter data for (optional)
            feature_columns: Column names for input features
            sequence_length: Length of sequence for time series data
            scaling: Whether to apply Min-Max scaling
            
        Returns:
            Processed data for anomaly detection
        """
        # Filter by house if specified
        if house_id is not None:
            df = df[df['house_id'] == house_id].copy()
        
        # Set default columns if not provided
        if feature_columns is None:
            # For anomaly detection, use more features
            feature_columns = ['total_power', 'voltage', 'current', 'power_factor']
        
        self.feature_columns = feature_columns
        self.sequence_length = sequence_length
        
        # Apply scaling if requested
        if scaling:
            # Scale features
            for col in feature_columns:
                if col not in self.scalers:
                    self.scalers[col] = MinMaxScaler()
                df[f'{col}_scaled'] = self.scalers[col].fit_transform(df[col].values.reshape(-1, 1))
            
            # Use scaled columns
            scaled_feature_cols = [f'{col}_scaled' for col in feature_columns]
        else:
            # Use original columns without scaling
            scaled_feature_cols = feature_columns
        
        # Create sequences for autoencoder
        X = self._create_sequences_for_autoencoder(df, scaled_feature_cols, sequence_length)
        
        # Split into train and test sets - use first 70% for training, rest for testing
        split_idx = int(len(X) * 0.7)
        X_train, X_test = X[:split_idx], X[split_idx:]
        
        # Extract metadata for analysis
        metadata = {}
        if 'theft_flag' in df.columns:
            test_metadata = df.iloc[split_idx + sequence_length:].reset_index(drop=True)
            metadata['theft_flags'] = test_metadata['theft_flag'].values
        if 'fault_flag' in df.columns:
            test_metadata = df.iloc[split_idx + sequence_length:].reset_index(drop=True)
            metadata['fault_flags'] = test_metadata['fault_flag'].values
        
        return {
            'X_train': X_train,
            'X_test': X_test,
            'metadata': metadata,
            'df': df,
            'feature_columns': feature_columns
        }
    
    def _create_sequences_for_autoencoder(self, df, feature_cols, sequence_length):
        """
        Create sequences for anomaly detection
        
        Args:
            df: DataFrame with preprocessed data
            feature_cols: Columns to use as features
            sequence_length: Length of input sequences
            
        Returns:
            X: Input sequences
        """
        X = []
        
        for i in range(len(df) - sequence_length):
            # Get sequence of features
            X_sequence = df[feature_cols].iloc[i:i+sequence_length].values
            X.append(X_sequence)
        
        return np.array(X)
    
    def inverse_transform(self, data, column_name):
        """
        Inverse transform scaled data back to original scale
        
        Args:
            data: Scaled data to transform back
            column_name: Name of the column (to find the right scaler)
            
        Returns:
            Data in original scale
        """
        if column_name not in self.scalers:
            raise ValueError(f"No scaler found for column: {column_name}")
        
        # Reshape data if needed
        if len(data.shape) == 1:
            data = data.reshape(-1, 1)
        
        return self.scalers[column_name].inverse_transform(data)
    
    def plot_appliance_disaggregation(self, actual, predicted, appliance_names, figsize=(15, 10)):
        """
        Plot actual vs predicted power consumption for each appliance
        
        Args:
            actual: Actual power values
            predicted: Predicted power values
            appliance_names: List of appliance names
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        n_appliances = len(appliance_names)
        fig, axes = plt.subplots(n_appliances, 1, figsize=figsize)
        
        if n_appliances == 1:
            axes = [axes]
        
        for i, appliance in enumerate(appliance_names):
            axes[i].plot(actual[:, i], label='Actual', color='blue', alpha=0.7)
            axes[i].plot(predicted[:, i], label='Predicted', color='red', alpha=0.7)
            axes[i].set_title(f'{appliance} Power Consumption')
            axes[i].set_ylabel('Power (W)')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_anomaly_detection(self, reconstruction_errors, anomalies, metadata=None, figsize=(15, 5)):
        """
        Plot reconstruction errors with detected anomalies
        
        Args:
            reconstruction_errors: Array of reconstruction errors
            anomalies: Boolean array indicating anomalies
            metadata: Dictionary with additional metadata (theft_flags, fault_flags)
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot reconstruction errors
        ax.plot(reconstruction_errors, label='Reconstruction Error', alpha=0.7)
        
        # Mark anomalies
        anomaly_indices = np.where(anomalies)[0]
        ax.scatter(anomaly_indices, reconstruction_errors[anomaly_indices], color='red', label='Detected Anomalies')
        
        # Mark actual theft and fault if available
        if metadata and 'theft_flags' in metadata:
            theft_indices = np.where(metadata['theft_flags'])[0]
            if len(theft_indices) > 0:
                ax.scatter(theft_indices, reconstruction_errors[theft_indices], color='orange', marker='x', s=100, label='Actual Theft')
        
        if metadata and 'fault_flags' in metadata:
            fault_indices = np.where(metadata['fault_flags'])[0]
            if len(fault_indices) > 0:
                ax.scatter(fault_indices, reconstruction_errors[fault_indices], color='purple', marker='^', s=100, label='Actual Fault')
        
        # Add threshold line
        if hasattr(self, 'threshold'):
            ax.axhline(y=self.threshold, color='green', linestyle='--', label='Threshold')
        
        ax.set_title('Anomaly Detection: Reconstruction Errors')
        ax.set_xlabel('Time')
        ax.set_ylabel('Reconstruction Error')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig

# ------------------------------------
# Model Trainer
# ------------------------------------
class ModelTrainer:
    """
    Coordinates training of multiple models and tracks efficiency metrics
    """
    def __init__(self):
        self.models = {}
        self.metrics = {}
    
    def train_all_models(self, data_processor, dataset):
        """
        Train all required models for the smart grid application
        
        Args:
            data_processor: DataProcessor instance with preprocessed data
            dataset: Dictionary containing the dataset
            
        Returns:
            Dictionary of trained models and their metrics
        """
        # Extract preprocessed data
        X_train = dataset['X_train']
        X_test = dataset['X_test']
        y_train = dataset['y_train']
        y_test = dataset['y_test']
        
        # 1. Train MLP model for NILM
        print("Training MLP model for NILM...")
        nilm_model = self._train_nilm_model(X_train, y_train, X_test, y_test)
        self.models['nilm'] = nilm_model
        
        # 2. Train Isolation Forest for anomaly detection
        print("Training Isolation Forest model for anomaly detection...")
        # For anomaly detection, we need to preprocess data differently
        anomaly_data = data_processor.create_anomaly_detection_data(
            dataset['df'], 
            feature_columns=['total_power', 'voltage', 'current', 'power_factor']
        )
        anomaly_model = self._train_anomaly_model(anomaly_data['X_train'], anomaly_data['X_test'])
        self.models['anomaly'] = anomaly_model
        
        # 3. Train RF/XGBoost for additional tasks
        # Extract features for traditional ML models
        print("Extracting features for traditional ML models...")
        # Use encoder part of anomaly model to extract features
        train_features = anomaly_model.extract_features(anomaly_data['X_train'])
        test_features = anomaly_model.extract_features(anomaly_data['X_test'])
        
        # Create target variables - for this example, we'll use theft detection
        # In a real app, we would use actual labels
        theft_labels = None
        if 'metadata' in anomaly_data and 'theft_flags' in anomaly_data['metadata']:
            theft_labels = anomaly_data['metadata']['theft_flags']
        else:
            # Generate synthetic labels for demonstration
            test_anomalies = anomaly_model.detect_anomalies(anomaly_data['X_test'])['anomalies']
            theft_labels = test_anomalies
        
        # For regression, we'll predict total power consumption
        regression_target = dataset['df']['total_power'].tail(len(test_features)).values
        
        # Train classification model
        print("Training Random Forest classification model...")
        rf_model = self._train_rf_classification(
            train_features, 
            np.zeros(len(train_features)),  # Placeholder for training
            test_features, 
            theft_labels
        )
        self.models['rf_classification'] = rf_model
        
        # Train regression model
        print("Training XGBoost regression model...")
        xgb_model = self._train_xgb_regression(
            train_features,
            np.zeros(len(train_features)),  # Placeholder for training
            test_features,
            regression_target
        )
        self.models['xgb_regression'] = xgb_model
        
        # Collect and combine all metrics
        self._collect_efficiency_metrics()
        
        return {
            'models': self.models,
            'metrics': self.metrics
        }
    
    def _train_nilm_model(self, X_train, y_train, X_test, y_test):
        """Train and evaluate MLP model for NILM"""
        # Get input shape from data
        sequence_length, n_features = X_train.shape[1], X_train.shape[2]
        n_appliances = y_train.shape[1]
        
        # Create and train model
        model = MLPNILMModel(sequence_length=sequence_length, n_features=n_features, n_appliances=n_appliances)
        model.build_model()
        model.train(X_train, y_train, X_test, y_test, epochs=10, batch_size=32)
        
        # Get metrics
        self.metrics['nilm'] = {
            'efficiency': model.get_efficiency_metrics(),
            'history': model.history if model.history else None
        }
        
        return model
    
    def _train_anomaly_model(self, X_train, X_test):
        """Train and evaluate Isolation Forest model for anomaly detection"""
        # Get input shape from data
        sequence_length, n_features = X_train.shape[1], X_train.shape[2]
        
        # Create and train model
        model = AnomalyDetectionModel(sequence_length=sequence_length, n_features=n_features)
        model.build_model()
        model.train(X_train, epochs=10, batch_size=32)
        
        # Test model
        test_results = model.detect_anomalies(X_test)
        
        # Get metrics
        self.metrics['anomaly'] = {
            'efficiency': model.get_efficiency_metrics(),
            'anomalies_detected': sum(test_results['anomalies']),
            'history': None  # No history for Isolation Forest
        }
        
        return model
    
    def _train_rf_classification(self, X_train, y_train, X_test, y_test):
        """Train and evaluate Random Forest classification model"""
        model = DecisionModel(model_type="classification", algorithm="rf")
        model.build_model()
        model.train(X_train, y_train)
        
        # For evaluation, we'll use the test data if available
        if y_test is not None and len(y_test) > 0:
            metrics = model.evaluate(X_test, y_test)
            self.metrics['rf_classification'] = metrics
        
        return model
    
    def _train_xgb_regression(self, X_train, y_train, X_test, y_test):
        """Train and evaluate XGBoost regression model"""
        model = DecisionModel(model_type="regression", algorithm="xgb")
        model.build_model()
        model.train(X_train, y_train)
        
        # For evaluation, we'll use the test data if available
        if y_test is not None and len(y_test) > 0:
            metrics = model.evaluate(X_test, y_test)
            self.metrics['xgb_regression'] = metrics
        
        return model
    
    def _collect_efficiency_metrics(self):
        """Collect and aggregate efficiency metrics from all models"""
        overall_metrics = {
            'total_training_time': 0,
            'total_inference_time': 0,
            'model_sizes': {},
            'accuracies': {}
        }
        
        # Collect metrics from ML models
        for model_name in ['nilm', 'anomaly']:
            if model_name in self.metrics and 'efficiency' in self.metrics[model_name]:
                eff = self.metrics[model_name]['efficiency']
                overall_metrics['total_training_time'] += eff.get('training_time_seconds', 0)
                overall_metrics['total_inference_time'] += eff.get('inference_time_seconds', 0)
                overall_metrics['model_sizes'][model_name] = eff.get('model_size_parameters', 0)
        
        # Collect metrics from traditional ML models
        for model_name in ['rf_classification', 'xgb_regression']:
            if model_name in self.metrics:
                overall_metrics['total_training_time'] += self.metrics[model_name].get('training_time', 0)
                overall_metrics['total_inference_time'] += self.metrics[model_name].get('inference_time', 0)
                
                # Collect accuracy metrics
                if model_name == 'rf_classification':
                    overall_metrics['accuracies']['classification'] = self.metrics[model_name].get('accuracy', 0)
                elif model_name == 'xgb_regression':
                    overall_metrics['accuracies']['regression'] = self.metrics[model_name].get('r2', 0)
        
        self.metrics['overall'] = overall_metrics
    
    def save_all_models(self, base_path="models"):
        """
        Save all trained models to disk
        
        Args:
            base_path: Base directory to save models
        """
        os.makedirs(base_path, exist_ok=True)
        
        # Save ML models
        if 'nilm' in self.models:
            self.models['nilm'].save(f"{base_path}/mlp_nilm.joblib")
        if 'anomaly' in self.models:
            self.models['anomaly'].save(f"{base_path}/anomaly_detector.joblib")
        
        # Save traditional ML models
        if 'rf_classification' in self.models:
            self.models['rf_classification'].save(f"{base_path}/rf_classification.joblib")
        if 'xgb_regression' in self.models:
            self.models['xgb_regression'].save(f"{base_path}/xgb_regression.joblib")
    
    def load_all_models(self, base_path="models"):
        """
        Load all models from disk
        
        Args:
            base_path: Base directory to load models from
            
        Returns:
            Dictionary of loaded models
        """
        models = {}
        
        # Load ML models
        nilm_path = f"{base_path}/mlp_nilm.joblib"
        if os.path.exists(nilm_path):
            models['nilm'] = MLPNILMModel()
            models['nilm'].load(nilm_path)
        
        anomaly_path = f"{base_path}/anomaly_detector.joblib"
        if os.path.exists(anomaly_path):
            models['anomaly'] = AnomalyDetectionModel()
            models['anomaly'].load(anomaly_path)
        
        # Load traditional ML models
        rf_path = f"{base_path}/rf_classification.joblib"
        if os.path.exists(rf_path):
            models['rf_classification'] = DecisionModel(model_type="classification", algorithm="rf")
            models['rf_classification'].load(rf_path)
        
        xgb_path = f"{base_path}/xgb_regression.joblib"
        if os.path.exists(xgb_path):
            models['xgb_regression'] = DecisionModel(model_type="regression", algorithm="xgb")
            models['xgb_regression'].load(xgb_path)
        
        self.models = models
        return models

# ------------------------------------
# Data Generator
# ------------------------------------
class UKDaleDataGenerator:
    """
    Simulates smart meter data based on UK-DALE dataset patterns
    """
    def __init__(self, num_houses=5, seed=42, num_days=28):
        """
        Initialize data generator with a set number of houses
        """
        self.num_houses = num_houses
        self.seed = seed
        self.num_days = num_days
        np.random.seed(seed)
        
        # Time parameters
        self.start_date = datetime(2023, 1, 1)
        self.sampling_interval = timedelta(hours=1)  # Hourly intervals for 28 days
        
        # Appliance parameters (mean/std for power in watts)
        self.appliance_params = {
            'refrigerator': {'mean': 60, 'std': 20, 'pattern': 'cyclic', 'cycle_time': 30, 'duty_cycle': 0.4},
            'washing_machine': {'mean': 400, 'std': 100, 'pattern': 'occasional', 'prob': 0.05, 'duration': 6},
            'dishwasher': {'mean': 700, 'std': 150, 'pattern': 'occasional', 'prob': 0.03, 'duration': 4},
            'microwave': {'mean': 800, 'std': 100, 'pattern': 'occasional', 'prob': 0.02, 'duration': 1},
            'television': {'mean': 120, 'std': 30, 'pattern': 'daily', 'start': 18, 'end': 23},
            'lighting': {'mean': 150, 'std': 50, 'pattern': 'daily', 'start': 17, 'end': 23},
            'air_conditioner': {'mean': 1200, 'std': 300, 'pattern': 'seasonal', 'season_factor': 0.7},
            'computer': {'mean': 100, 'std': 20, 'pattern': 'daily', 'start': 9, 'end': 22}
        }
        
        # House parameters (baseline values)
        self.house_params = []
        for i in range(num_houses):
            # Generate house-specific parameters
            house = {
                'id': i + 1,
                'voltage_mean': 230 + np.random.normal(0, 5),  # Mean voltage (V)
                'voltage_std': 2 + np.random.exponential(1),   # Voltage stability
                'power_factor': 0.90 + np.random.uniform(0, 0.08),  # Power factor
                'frequency': 50 + np.random.normal(0, 0.1),    # Grid frequency (Hz)
                'thd': 2 + np.random.exponential(1),           # Total harmonic distortion (%)
                'temperature': 21 + np.random.normal(0, 2),    # Indoor temperature (°C)
                'appliance_factors': {}                        # House-specific appliance factors
            }
            
            # Generate appliance factors for this house
            for app in self.appliance_params:
                house['appliance_factors'][app] = max(0.5, min(1.5, np.random.normal(1.0, 0.2)))
            
            self.house_params.append(house)
        
        # Simulation anomalies - initialized as all False
        self.theft_simulation = [False] * num_houses
        self.fault_simulation = [False] * num_houses
        
        # Theft occurrence days (realistically rare) - only for specific houses on specific days
        self.theft_days = {}
        # Only 1-2 houses might have theft
        theft_house_ids = np.random.choice(range(num_houses), size=min(2, num_houses), replace=False)
        
        for house_idx in theft_house_ids:
            # Each house with theft might have 1-3 theft days in the 28-day period
            # More likely to be in the last week for testing
            num_theft_days = np.random.randint(1, 3)
            # Bias toward later days (for testing data)
            theft_days = np.random.choice(
                range(22, 28),  # Last 6 days
                size=num_theft_days, 
                replace=False
            )
            self.theft_days[house_idx] = theft_days
            
        # Fault occurrence days (also realistically rare)
        self.fault_days = {}
        # Only 1-2 houses might have grid faults
        fault_house_ids = np.random.choice(range(num_houses), size=min(2, num_houses), replace=False)
        
        for house_idx in fault_house_ids:
            # Each house with faults might have 1-2 fault days
            num_fault_days = np.random.randint(1, 3)
            # Distribute throughout the month but with bias toward testing period
            fault_days = np.random.choice(
                range(21, 28),  # Last 7 days (testing period)
                size=num_fault_days, 
                replace=False
            )
            self.fault_days[house_idx] = fault_days
    
    def generate_dataset(self):
        """
        Generate a complete dataset for all houses over 28 days
        Returns a pandas DataFrame with the dataset
        """
        # Calculate number of data points (hourly for 28 days)
        num_points = int(self.num_days * 24 / self.sampling_interval.total_seconds() * 3600)
        
        data = []
        current_time = self.start_date
        
        # Store day number for each timestamp for easier filtering
        day_counter = 1
        
        for i in range(num_points):
            current_hour = current_time.hour
            current_day = (current_time - self.start_date).days + 1  # Day 1, Day 2, etc.
            day_of_week = current_time.weekday()  # 0-6 (Monday to Sunday)
            is_weekend = day_of_week >= 5
            day_progress = current_hour / 24.0  # 0-1 progress through the day
            
            # Seasonal factor (simplified)
            seasonal_factor = 0.5 + 0.5 * np.sin((current_day - 1) / 28 * 2 * np.pi)
            
            for house_idx, house in enumerate(self.house_params):
                # Check if this is a theft day for this house
                is_theft_day = False
                if house_idx in self.theft_days and current_day in self.theft_days[house_idx]:
                    is_theft_day = True
                    
                # Check if this is a fault day for this house
                is_fault_day = False
                if house_idx in self.fault_days and current_day in self.fault_days[house_idx]:
                    is_fault_day = True
                
                # Get current voltage with some natural variation
                voltage = house['voltage_mean'] + np.random.normal(0, house['voltage_std'])
                
                # Get frequency with some variation
                frequency = house['frequency'] + np.random.normal(0, 0.05)
                
                # Calculate appliance power
                appliance_power = {}
                total_power = 0
                
                for app_name, params in self.appliance_params.items():
                    power = 0
                    
                    if params['pattern'] == 'cyclic':
                        # Cyclic pattern (like refrigerator)
                        cycle_position = (i % params['cycle_time']) / params['cycle_time']
                        if cycle_position < params['duty_cycle']:
                            power = params['mean'] + np.random.normal(0, params['std'])
                        else:
                            power = np.random.normal(0, params['std'] * 0.1)  # Standby power
                    
                    elif params['pattern'] == 'occasional':
                        # Occasional use (like washing machine)
                        # Increase probability based on time of day and weekends
                        time_factor = 1.0
                        if app_name == 'washing_machine' and is_weekend:
                            time_factor = 2.0  # More likely on weekends
                        elif app_name == 'dishwasher' and current_hour in [7, 8, 19, 20, 21]:
                            time_factor = 3.0  # More likely after meals
                            
                        if np.random.random() < params['prob'] * time_factor:
                            # Start a usage cycle
                            power = params['mean'] + np.random.normal(0, params['std'])
                    
                    elif params['pattern'] == 'daily':
                        # Daily pattern (like television, lighting)
                        if current_hour >= params['start'] and current_hour < params['end']:
                            usage_factor = min(1.0, (current_hour - params['start']) / 2)  # Ramp up over 2 hours
                            if is_weekend:
                                usage_factor *= 1.2  # Higher usage on weekends
                            power = params['mean'] * usage_factor + np.random.normal(0, params['std'])
                    
                    elif params['pattern'] == 'seasonal':
                        # Seasonal pattern (like air conditioner)
                        if seasonal_factor > 0.7:  # Only active in "summer"
                            power = params['mean'] * seasonal_factor * params['season_factor'] + np.random.normal(0, params['std'])
                    
                    # Apply house-specific factor
                    power *= house['appliance_factors'][app_name]
                    
                    # Ensure power is not negative
                    power = max(0, power)
                    
                    # Store appliance power
                    appliance_power[app_name] = power
                    
                    # Add to total
                    total_power += power
                
                # Add some baseline power
                baseline_power = 50 + 30 * np.random.random()
                total_power += baseline_power
                
                # Simulate power factor
                power_factor = house['power_factor'] - 0.05 * np.random.random()
                
                # Calculate apparent power and current
                apparent_power = total_power / power_factor
                current = apparent_power / voltage
                
                # Calculate reactive power
                reactive_power = np.sqrt(apparent_power**2 - total_power**2)
                
                # Calculate THD (total harmonic distortion)
                thd = house['thd'] + np.random.normal(0, 0.5)
                
                # Initialize theft variables
                theft_flag = 0
                theft_factor = 1.0
                theft_reason_str = ""
                
                # Simulate theft on theft days
                if is_theft_day:
                    theft_flag = 1
                    
                    # Different theft patterns
                    theft_type = np.random.choice([
                        "meter_bypass",
                        "meter_tampering",
                        "direct_connection"
                    ])
                    
                    if theft_type == "meter_bypass":
                        # Meter bypass: reported consumption is lower than actual
                        theft_factor = 0.7  # Report only 70% of consumption
                        theft_reason_str = "Meter Bypass"
                    elif theft_type == "meter_tampering":
                        # Tampering: consumption pattern looks irregular
                        theft_factor = 0.6 + 0.3 * np.random.random()  # Variable reporting
                        theft_reason_str = "Meter Tampering"
                    elif theft_type == "direct_connection":
                        # Direct connection: very low reported consumption
                        theft_factor = 0.4  # Report only 40% of consumption
                        theft_reason_str = "Direct Connection"
                
                # Apply theft factor to reported power
                reported_power = total_power * theft_factor
                
                # Initialize fault variables
                fault_flag = 0
                fault_reason_str = ""
                
                # Simulate grid faults on fault days
                if is_fault_day:
                    fault_flag = 1
                    
                    # Different fault types
                    fault_type = np.random.choice([
                        "voltage_sag",
                        "voltage_swell",
                        "harmonic_distortion"
                    ])
                    
                    if fault_type == "voltage_sag":
                        # Voltage sag
                        voltage *= 0.8  # 20% drop
                        fault_reason_str = "Voltage Sag"
                    elif fault_type == "voltage_swell":
                        # Voltage swell
                        voltage *= 1.15  # 15% increase
                        fault_reason_str = "Voltage Swell"
                    elif fault_type == "harmonic_distortion":
                        # Harmonic distortion
                        thd *= 3  # Triple the THD
                        fault_reason_str = "Harmonic Distortion"
                
                # Add data point to our list
                data_point = {
                    'timestamp': current_time,
                    'house_id': house['id'],
                    'day': current_day,
                    'hour': current_hour,
                    'is_weekend': int(is_weekend),
                    'day_of_week': day_of_week,
                    'day_progress': day_progress,
                    'seasonal_factor': seasonal_factor,
                    'temperature': house['temperature'] + np.random.normal(0, 1),
                    'voltage': voltage,
                    'current': current,
                    'frequency': frequency,
                    'power_factor': power_factor,
                    'thd': thd,
                    'total_power': total_power,
                    'reported_power': reported_power,
                    'reactive_power': reactive_power,
                    'apparent_power': apparent_power,
                    'baseline_power': baseline_power,
                    'theft_flag': theft_flag,
                    'theft_factor': theft_factor,
                    'theft_reason': theft_reason_str,
                    'fault_flag': fault_flag,
                    'fault_reason': fault_reason_str
                }
                
                # Add appliance-specific power
                for app_name, power in appliance_power.items():
                    data_point[app_name] = power
                
                data.append(data_point)
            
            # Increment time
            current_time += self.sampling_interval
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        return df
    
    def split_train_test(self, df):
        """
        Split the dataset into training (first 3 weeks) and testing (last week)
        """
        # Use day number for splitting
        train_df = df[df['day'] <= 21].copy()  # First 3 weeks for training
        test_df = df[df['day'] > 21].copy()   # Last week for testing
        
        return train_df, test_df

# ------------------------------------
# Utility Functions
# ------------------------------------

def get_csv_download_link(df, filename="energy_dataset.csv", text="Download CSV"):
    """Generate a link to download the dataframe as a CSV file"""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{text}</a>'
    return href

def get_image_download_link(fig, filename="plot.png", text="Download Plot"):
    """Generate a link to download the plot as an image"""
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    href = f'<a href="data:image/png;base64,{b64}" download="{filename}">{text}</a>'
    return href

def create_color_map(color_list, name='custom_cmap'):
    """Create a custom colormap from a list of colors"""
    return mcolors.LinearSegmentedColormap.from_list(name, color_list)

# ------------------------------------
# Session State Management
# ------------------------------------

# Initialize session state variables for tracking progress and storing models
if 'model_training_complete' not in st.session_state:
    st.session_state.model_training_complete = False
if 'model_training_progress' not in st.session_state:
    st.session_state.model_training_progress = 0
if 'training_thread_running' not in st.session_state:
    st.session_state.training_thread_running = False
if 'models' not in st.session_state:
    st.session_state.models = {}
if 'data_generator' not in st.session_state:
    st.session_state.data_generator = None
if 'dataset' not in st.session_state:
    st.session_state.dataset = None
if 'training_error' not in st.session_state:
    st.session_state.training_error = None
if 'current_house' not in st.session_state:
    st.session_state.current_house = 1
if 'analysis_data' not in st.session_state:
    st.session_state.analysis_data = {}
if 'training_data' not in st.session_state:
    st.session_state.training_data = None
if 'testing_data' not in st.session_state:
    st.session_state.testing_data = None
if 'data_processor' not in st.session_state:
    st.session_state.data_processor = DataProcessor(appliance_columns=[app.lower() for app in APPLIANCES])
if 'deep_learning_models_ready' not in st.session_state:
    st.session_state.deep_learning_models_ready = False
if 'nilm_results' not in st.session_state:
    st.session_state.nilm_results = None
if 'anomaly_results' not in st.session_state:
    st.session_state.anomaly_results = None
if 'metrics' not in st.session_state:
    st.session_state.metrics = {}

def run_model_training():
    """Function to run model training in a separate thread to avoid blocking the UI"""
    try:
        # Check if data is available
        if st.session_state.dataset is None:
            st.session_state.training_error = "Dataset not available. Generate data first."
            st.session_state.model_training_progress = 0
            st.session_state.training_thread_running = False
            return
        
        # Initialize data processor if not already done
        if 'data_processor' not in st.session_state:
            st.session_state.data_processor = DataProcessor(
                appliance_columns=[app.lower() for app in APPLIANCES]
            )
        
        # Get house ID for training
        house_id = st.session_state.current_house
        
        # Preprocess data
        st.session_state.model_training_progress = 0.1
        dataset_df = st.session_state.dataset
        
        # Filter for the selected house
        house_df = dataset_df[dataset_df['house_id'] == house_id].copy()
        
        # Split into training and testing sets based on day
        train_df = house_df[house_df['day'] <= 21].copy()  # First 3 weeks for training
        test_df = house_df[house_df['day'] > 21].copy()   # Last week for testing
        
        # Store in session state
        st.session_state.training_data = train_df
        st.session_state.testing_data = test_df
        
        # Preprocess for NILM
        st.session_state.model_training_progress = 0.2
        appliance_columns = [app.lower() for app in APPLIANCES]
        feature_columns = ['total_power', 'hour', 'is_weekend', 'day_of_week', 'temperature']
        
        preprocessed_data = st.session_state.data_processor.preprocess_data(
            house_df, 
            appliance_columns=appliance_columns, 
            feature_columns=feature_columns,
            sequence_length=24,
            test_size=0.3,
            scaling=True
        )
        
        # Initialize model trainer
        st.session_state.model_training_progress = 0.3
        trainer = ModelTrainer()
        
        # Train models
        st.session_state.model_training_progress = 0.4
        results = trainer.train_all_models(st.session_state.data_processor, preprocessed_data)
        
        # Store models and metrics
        st.session_state.models = results['models']
        st.session_state.metrics = results['metrics']
        
        # Generate predictions and analysis for display
        st.session_state.model_training_progress = 0.8
        
        # NILM predictions
        if 'nilm' in st.session_state.models:
            nilm_model = st.session_state.models['nilm']
            X_test = preprocessed_data['X_test']
            y_test = preprocessed_data['y_test']
            
            # Get predictions for test data
            nilm_predictions = nilm_model.predict(X_test)
            
            # Convert predictions to numpy array if they aren't already
            if isinstance(nilm_predictions, list):
                nilm_predictions = np.array(nilm_predictions).transpose(1, 0, 2).squeeze(axis=2)
            
            # Store results for display
            st.session_state.nilm_results = {
                'actual': y_test,
                'predicted': nilm_predictions,
                'appliance_names': appliance_columns
            }
        
        # Anomaly detection
        if 'anomaly' in st.session_state.models:
            # Create anomaly detection data
            anomaly_data = st.session_state.data_processor.create_anomaly_detection_data(
                house_df,
                feature_columns=['total_power', 'voltage', 'current', 'power_factor', 'thd'],
                sequence_length=24
            )
            
            anomaly_model = st.session_state.models['anomaly']
            X_test = anomaly_data['X_test']
            
            # Detect anomalies
            anomaly_results = anomaly_model.detect_anomalies(X_test)
            
            # Store results for display
            st.session_state.anomaly_results = {
                'reconstruction_errors': anomaly_results['reconstruction_errors'],
                'anomalies': anomaly_results['anomalies'],
                'metadata': anomaly_data['metadata'] if 'metadata' in anomaly_data else None
            }
        
        # Mark as complete
        st.session_state.model_training_progress = 1.0
        st.session_state.model_training_complete = True
        st.session_state.training_thread_running = False
        st.session_state.deep_learning_models_ready = True
        st.session_state.training_error = None
        
    except Exception as e:
        st.session_state.training_error = str(e)
        st.session_state.training_thread_running = False
        print(f"Error in model training: {e}")

def start_model_training_thread():
    """Start a background thread for model training"""
    if not st.session_state.training_thread_running:
        st.session_state.training_thread_running = True
        st.session_state.model_training_complete = False
        st.session_state.model_training_progress = 0.0
        
        # Run in a separate thread to avoid blocking the UI
        training_thread = threading.Thread(target=run_model_training)
        training_thread.start()

def load_or_create_dataset():
    """Load dataset from session state or create a new one if not present"""
    if st.session_state.dataset is None:
        st.session_state.data_generator = UKDaleDataGenerator(num_houses=5, seed=42, num_days=28)
        st.session_state.dataset = st.session_state.data_generator.generate_dataset()
    return st.session_state.dataset

def check_for_existing_models():
    """Check if there are already trained models available"""
    if 'models' in st.session_state and st.session_state.models:
        # Check if we have both models
        ml_ready = ('nilm' in st.session_state.models and 
                   'anomaly' in st.session_state.models)
        
        # Check if we have both traditional ML models
        traditional_ml_ready = ('rf_classification' in st.session_state.models and 
                               'xgb_regression' in st.session_state.models)
        
        return ml_ready and traditional_ml_ready
    
    return False

# ------------------------------------
# Streamlit UI Functions
# ------------------------------------

def render_sidebar():
    """Render the sidebar UI"""
    st.sidebar.title("Smart Grid Energy Monitoring")
    
    menu = st.sidebar.radio("Navigation", [
        "Dashboard Overview", 
        "Load Disaggregation", 
        "Theft Detection", 
        "Fault Detection",
        "Efficiency Analysis",
        "Data Explorer"
    ])
    
    # Load or generate data
    dataset = load_or_create_dataset()
    
    # House selection
    st.sidebar.subheader("House Selection")
    house_id = st.sidebar.selectbox(
        "Select House", 
        options=[house["id"] for house in st.session_state.data_generator.house_params],
        index=0,
        key="house_selector"
    )
    st.session_state.current_house = house_id
    
    # Train models button
    models_ready = check_for_existing_models()
    if not models_ready:
        if st.sidebar.button("Train Models") and not st.session_state.training_thread_running:
            start_model_training_thread()
    
    # Show training progress
    if st.session_state.training_thread_running:
        st.sidebar.subheader("Training Progress")
        st.sidebar.progress(st.session_state.model_training_progress)
        st.sidebar.text(f"Progress: {st.session_state.model_training_progress:.1%}")
    
    # Show training error if any
    if st.session_state.training_error:
        st.sidebar.error(f"Training Error: {st.session_state.training_error}")
    
    # Show model status
    st.sidebar.subheader("Model Status")
    if models_ready:
        st.sidebar.success("✅ All models are trained and ready")
    else:
        st.sidebar.info("ℹ️ Models need to be trained")
    
    return menu

def render_dashboard(house_id):
    """Render the main dashboard UI"""
    st.title("Smart Grid Energy Monitoring Dashboard")
    
    # Show overall metrics
    st.subheader("Energy Overview for House " + str(house_id))
    
    # Filter data for selected house
    house_data = st.session_state.dataset[st.session_state.dataset['house_id'] == house_id]
    
    # Create metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_consumption = house_data['total_power'].sum() / 1000  # kWh
        st.metric("Total Energy", f"{total_consumption:.1f} kWh")
    
    with col2:
        avg_power = house_data['total_power'].mean()  # W
        st.metric("Average Power", f"{avg_power:.1f} W")
    
    with col3:
        avg_voltage = house_data['voltage'].mean()  # V
        st.metric("Average Voltage", f"{avg_voltage:.1f} V")
    
    with col4:
        power_factor = house_data['power_factor'].mean()
        st.metric("Average Power Factor", f"{power_factor:.2f}")
    
    # Create charts
    st.subheader("Power Consumption Over Time")
    
    # Filter by time period
    time_period = st.selectbox(
        "Select Time Period",
        options=["Last 7 days", "Last 14 days", "Full month"],
        index=0
    )
    
    if time_period == "Last 7 days":
        filter_day = house_data['day'] > (st.session_state.data_generator.num_days - 7)
    elif time_period == "Last 14 days":
        filter_day = house_data['day'] > (st.session_state.data_generator.num_days - 14)
    else:
        filter_day = house_data['day'] > 0
    
    filtered_data = house_data[filter_day].copy()
    
    # Power consumption time series
    power_fig = plt.figure(figsize=(10, 5))
    plt.plot(filtered_data['timestamp'], filtered_data['total_power'], label='Total Power')
    plt.xlabel('Time')
    plt.ylabel('Power (W)')
    plt.title('Total Power Consumption')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(power_fig)
    
    # Appliance contribution
    st.subheader("Appliance Contribution")
    
    appliance_columns = [app.lower().replace(" ", "_") for app in APPLIANCES]
    appliance_data = filtered_data[appliance_columns].sum()
    appliance_data.index = APPLIANCES  # Restore original labels for display

    appliance_data.index = APPLIANCES
    
    appliance_colors = list(APPLIANCE_COLORS.values())
    
    # Create pie chart
    fig, ax = plt.subplots(figsize=(10, 6))
    wedges, texts, autotexts = ax.pie(
        appliance_data, 
        labels=appliance_data.index, 
        autopct='%1.1f%%',
        textprops={'fontsize': 9}, 
        colors=appliance_colors
    )
    ax.set_title('Appliance Power Consumption Distribution')
    plt.tight_layout()
    st.pyplot(fig)
    
    # If models are trained, show predictions
    if st.session_state.model_training_complete:
        st.subheader("Model Insights")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("Appliance Predictions (Last 24 hours)")
            if st.session_state.nilm_results:
                # Show the last day's worth of predictions
                last_idx = min(24, len(st.session_state.nilm_results['actual']))
                actual = st.session_state.nilm_results['actual'][-last_idx:]
                predicted = st.session_state.nilm_results['predicted'][-last_idx:]
                
                # Calculate accuracy
                mse = np.mean(np.square(actual - predicted))
                st.info(f"Mean Squared Error: {mse:.2f} W²")
                
                # Plot
                fig = plt.figure(figsize=(10, 6))
                plt.bar(
                    np.arange(len(APPLIANCES)) - 0.2, 
                    np.mean(actual, axis=0), 
                    width=0.4, 
                    label="Actual", 
                    color="blue", 
                    alpha=0.7
                )
                plt.bar(
                    np.arange(len(APPLIANCES)) + 0.2, 
                    np.mean(predicted, axis=0), 
                    width=0.4, 
                    label="Predicted", 
                    color="red", 
                    alpha=0.7
                )
                plt.xticks(np.arange(len(APPLIANCES)), [app[:3] for app in APPLIANCES], rotation=45)
                plt.legend()
                plt.ylabel("Average Power (W)")
                plt.title("Actual vs Predicted Appliance Power")
                plt.tight_layout()
                st.pyplot(fig)
        
        with col2:
            st.write("Anomaly Detection")
            if st.session_state.anomaly_results:
                # Show anomaly counts
                anomalies = st.session_state.anomaly_results['anomalies']
                num_anomalies = sum(anomalies)
                total_points = len(anomalies)
                
                st.info(f"Detected {num_anomalies} anomalies out of {total_points} data points ({num_anomalies/total_points:.1%})")
                
                # Plot reconstruction errors
                fig = plt.figure(figsize=(10, 6))
                plt.plot(st.session_state.anomaly_results['reconstruction_errors'], label="Reconstruction Error")
                plt.axhline(
                    y=st.session_state.models['anomaly'].threshold, 
                    color='red', 
                    linestyle='--', 
                    label="Threshold"
                )
                
                # Mark anomalies
                anomaly_indices = np.where(anomalies)[0]
                plt.scatter(
                    anomaly_indices, 
                    st.session_state.anomaly_results['reconstruction_errors'][anomaly_indices], 
                    color='red', 
                    marker='x', 
                    label="Anomalies"
                )
                
                plt.legend()
                plt.xlabel("Time")
                plt.ylabel("Reconstruction Error")
                plt.title("Anomaly Detection Results")
                plt.tight_layout()
                st.pyplot(fig)

def render_appliance_disaggregation(house_id):
    """Render the appliance disaggregation UI"""
    st.title("Appliance-Level Load Disaggregation")
    
    # Check if models are trained
    if not st.session_state.model_training_complete:
        st.warning("Please train the models first from the Dashboard.")
        return
    
    # Explanation
    st.info("""
    This section shows detailed appliance-level power consumption. 
    The MLP model disaggregates the total power into individual appliances.
    """)
    
    # Show model architecture
    with st.expander("Model Architecture - MLP Model for NILM"):
        st.write("""
        The Non-Intrusive Load Monitoring (NILM) model uses Multi-Layer Perceptrons:
        
        1. **Input**: Flattened time series data of energy consumption
        2. **Hidden Layers**: Dense neural networks with 64 and 32 neurons
        3. **Individual output heads**: One MLP model for each appliance
        
        This architecture can effectively learn the unique signatures of different appliances from aggregate data.
        """)
    
    # Select time range
    st.subheader("Time Range Selection")
    
    # Filter data for selected house
    house_data = st.session_state.dataset[st.session_state.dataset['house_id'] == house_id]
    
    # Get date range
    min_date = house_data['timestamp'].min()
    max_date = house_data['timestamp'].max()
    
    # Allow user to select date range
    date_range = st.date_input(
        "Select date range",
        value=[max_date - timedelta(days=7), max_date],
        min_value=min_date,
        max_value=max_date
    )
    
    if len(date_range) == 2:
        start_date, end_date = date_range
        start_date = datetime.combine(start_date, datetime.min.time())
        end_date = datetime.combine(end_date, datetime.max.time())
        
        filtered_data = house_data[
            (house_data['timestamp'] >= start_date) & 
            (house_data['timestamp'] <= end_date)
        ].copy()
        
        if len(filtered_data) > 0:
            # Show appliance breakdown
            st.subheader("Appliance Power Breakdown")
            
            # Get appliance columns
            appliance_columns = [app.lower() for app in APPLIANCES]
            
            # Prepare data for visualizations
            appliance_data = filtered_data[appliance_columns].copy()
            appliance_data.columns = APPLIANCES  # Use nice names
            
            # Calculate percentages
            appliance_totals = appliance_data.sum()
            total_power = appliance_totals.sum()
            appliance_percentages = (appliance_totals / total_power * 100).round(1)
            
            # Create columns for metrics
            cols = st.columns(4)
            for i, (app, percentage) in enumerate(zip(APPLIANCES, appliance_percentages)):
                col_idx = i % 4
                cols[col_idx].metric(
                    app,
                    f"{appliance_totals[i]:.1f} W",
                    f"{percentage:.1f}% of total"
                )
            
            # Create time series of appliance power
            st.subheader("Appliance Power Over Time")
            
            # Set up figure with subplots
            fig, axs = plt.subplots(4, 2, figsize=(15, 15), sharex=True)
            axs = axs.flatten()
            
            for i, app in enumerate(APPLIANCES):
                # Plot power over time
                axs[i].plot(
                    filtered_data['timestamp'], 
                    filtered_data[app.lower()], 
                    color=APPLIANCE_COLORS[app],
                    label='Actual'
                )
                
                # Add formatting
                axs[i].set_title(app)
                axs[i].set_ylabel('Power (W)')
                axs[i].grid(True, alpha=0.3)
                
                # If we have model predictions, show them
                if st.session_state.nilm_results:
                    # We need to align the predictions with the filtered data
                    # This is simplified - in a real app you'd need to match timestamps properly
                    if len(st.session_state.nilm_results['predicted']) == len(house_data):
                        # Get indices matching the filter
                        filter_indices = house_data.index[
                            (house_data['timestamp'] >= start_date) & 
                            (house_data['timestamp'] <= end_date)
                        ]
                        
                        # Get predictions for these indices
                        pred_indices = [house_data.index.get_loc(idx) for idx in filter_indices]
                        if pred_indices:
                            predictions = st.session_state.nilm_results['predicted'][pred_indices, i]
                            
                            # Plot predictions
                            axs[i].plot(
                                filtered_data['timestamp'],
                                predictions,
                                color='red',
                                linestyle='--',
                                alpha=0.7,
                                label='Predicted'
                            )
                            axs[i].legend()
            
            # Format x-axis for dates
            for ax in axs:
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Show disaggregation accuracy
            st.subheader("Disaggregation Accuracy")
            
            if st.session_state.nilm_results:
                # Show model metrics
                if 'nilm' in st.session_state.metrics:
                    metrics = st.session_state.metrics['nilm']
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Training Time", f"{metrics['efficiency']['training_time_seconds']:.2f} seconds")
                        st.metric("Model Size", f"{metrics['efficiency']['model_size_parameters']:,} parameters")
                    
                    with col2:
                        st.metric("Inference Time", f"{metrics['efficiency']['inference_time_seconds']:.4f} seconds")
                        
                        # Calculate mean absolute error if we have history
                        if 'history' in metrics and metrics['history'] is not None:
                            mae = np.mean(metrics['history']['val_mae'])
                            st.metric("Mean Absolute Error", f"{mae:.4f}")
                
                # Plot the learning curves
                if 'nilm' in st.session_state.metrics and 'history' in st.session_state.metrics['nilm']:
                    history = st.session_state.metrics['nilm']['history']
                    
                    if history is not None:
                        st.subheader("Learning Curves")
                        
                        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
                        
                        # Plot loss
                        ax1.plot([1], history['loss'], label='Train')
                        ax1.plot([1], history['val_loss'], label='Validation')
                        ax1.set_title('Model Loss')
                        ax1.set_ylabel('Loss')
                        ax1.set_xlabel('Epoch')
                        ax1.legend()
                        ax1.grid(True, alpha=0.3)
                        
                        # Plot MAE
                        ax2.plot([1], history['mae'], label='Train')
                        ax2.plot([1], history['val_mae'], label='Validation')
                        ax2.set_title('Model MAE')
                        ax2.set_ylabel('MAE')
                        ax2.set_xlabel('Epoch')
                        ax2.legend()
                        ax2.grid(True, alpha=0.3)
                        
                        plt.tight_layout()
                        st.pyplot(fig)

def render_anomaly_detection(house_id, anomaly_type="theft"):
    """Render the anomaly detection UI"""
    if anomaly_type == "theft":
        st.title("Energy Theft Detection")
        explanation = """
        Energy theft detection uses an Isolation Forest model to identify unusual patterns in consumption.
        Theft typically appears as unexplained drops in reported energy usage.
        """
    else:  # fault
        st.title("Smart Grid Fault Detection")
        explanation = """
        Grid fault detection uses an Isolation Forest model to identify abnormal patterns in voltage, frequency, and power quality.
        Faults appear as unusual spikes or drops in these measurements.
        """
    
    # Explanation
    st.info(explanation)
    
    # Check if models are trained
    if not st.session_state.model_training_complete:
        st.warning("Please train the models first from the Dashboard.")
        return
    
    # Show model architecture
    with st.expander("Model Architecture - Isolation Forest"):
        st.write("""
        The anomaly detection model uses an Isolation Forest architecture:
        
        1. **Feature Extraction**: Extract important features from time series data
        2. **Isolation Forest**: Detects anomalies by isolating outliers in the feature space
        3. **Decision Function**: Scores each data point based on how anomalous it is
        
        Anomalies are detected when the anomaly score is high, indicating the data point contains unusual patterns.
        """)
    
    # Filter data for selected house
    house_data = st.session_state.dataset[st.session_state.dataset['house_id'] == house_id]
    
    # Show overall detection results
    st.subheader("Detection Results")
    
    if st.session_state.anomaly_results:
        anomalies = st.session_state.anomaly_results['anomalies']
        num_anomalies = sum(anomalies)
        total_points = len(anomalies)
        
        # Get theft or fault flags, if available
        true_flags = None
        if anomaly_type == "theft" and 'metadata' in st.session_state.anomaly_results:
            if 'theft_flags' in st.session_state.anomaly_results['metadata']:
                true_flags = st.session_state.anomaly_results['metadata']['theft_flags']
        elif anomaly_type == "fault" and 'metadata' in st.session_state.anomaly_results:
            if 'fault_flags' in st.session_state.anomaly_results['metadata']:
                true_flags = st.session_state.anomaly_results['metadata']['fault_flags']
        
        # Calculate detection metrics
        if true_flags is not None:
            true_positives = sum(anomalies & true_flags)
            false_positives = sum(anomalies & ~true_flags)
            true_negatives = sum(~anomalies & ~true_flags)
            false_negatives = sum(~anomalies & true_flags)
            
            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Precision", f"{precision:.2f}")
            with col2:
                st.metric("Recall", f"{recall:.2f}")
            with col3:
                st.metric("F1 Score", f"{f1:.2f}")
            with col4:
                st.metric("Detected Anomalies", f"{num_anomalies} / {total_points}")
        else:
            st.metric("Detected Anomalies", f"{num_anomalies} / {total_points}")
        
        # Plot reconstruction errors
        st.subheader("Anomaly Scores")
        
        fig = plt.figure(figsize=(12, 6))
        plt.plot(st.session_state.anomaly_results['reconstruction_errors'], label="Anomaly Score")
        plt.axhline(
            y=st.session_state.models['anomaly'].threshold, 
            color='red', 
            linestyle='--', 
            label="Threshold"
        )
        
        # Mark anomalies
        anomaly_indices = np.where(anomalies)[0]
        plt.scatter(
            anomaly_indices, 
            st.session_state.anomaly_results['reconstruction_errors'][anomaly_indices], 
            color='red', 
            marker='x', 
            label="Detected Anomalies"
        )
        
        # Mark true flags if available
        if true_flags is not None:
            true_flag_indices = np.where(true_flags)[0]
            plt.scatter(
                true_flag_indices, 
                st.session_state.anomaly_results['reconstruction_errors'][true_flag_indices], 
                color='green', 
                marker='o', 
                s=100,
                facecolors='none',
                label=f"True {anomaly_type.capitalize()}"
            )
        
        plt.legend()
        plt.xlabel("Time")
        plt.ylabel("Anomaly Score")
        plt.title(f"{anomaly_type.capitalize()} Detection Results")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        
        # Show detection cases
        st.subheader(f"Detected {anomaly_type.capitalize()} Cases")
        
        # Filter for the last week (testing period)
        test_data = house_data[house_data['day'] > 21].copy()
        test_data = test_data.reset_index(drop=True)
        
        # Find which days have anomalies
        if len(anomaly_indices) > 0 and len(test_data) >= len(anomaly_indices):
            anomaly_days = test_data.iloc[anomaly_indices]['day'].unique()
            
            if len(anomaly_days) > 0:
                st.write(f"Anomalies detected on days: {', '.join(map(str, anomaly_days))}")
                
                for day in anomaly_days:
                    day_data = test_data[test_data['day'] == day]
                    day_anomaly_indices = anomaly_indices[
                        np.isin(test_data.iloc[anomaly_indices]['day'], [day])
                    ]
                    
                    with st.expander(f"Day {day} Analysis"):
                        
                        # Create plot
                        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
                        
                        # Plot 1: Total power consumption
                        ax1.plot(day_data['hour'], day_data['total_power'], label='Total Power', color='blue')
                        ax1.plot(day_data['hour'], day_data['reported_power'], label='Reported Power', color='green', linestyle='--')
                        ax1.set_title(f"Power Consumption on Day {day}")
                        ax1.set_ylabel("Power (W)")
                        ax1.legend()
                        ax1.grid(True, alpha=0.3)
                        
                        # Highlight anomaly hours
                        anomaly_hours = day_data.iloc[day_anomaly_indices]['hour'].values
                        for hour in anomaly_hours:
                            ax1.axvspan(hour-0.5, hour+0.5, color='red', alpha=0.2)
                        
                        # Plot 2 depends on anomaly type
                        if anomaly_type == "theft":
                            # For theft, show the difference between total and reported power
                            power_diff = day_data['total_power'] - day_data['reported_power']
                            ax2.bar(day_data['hour'], power_diff, label='Power Difference', color='orange')
                            ax2.set_title("Difference between Total and Reported Power")
                            ax2.set_ylabel("Power Difference (W)")
                            ax2.set_xlabel("Hour of Day")
                            
                            for hour in anomaly_hours:
                                ax2.axvspan(hour-0.5, hour+0.5, color='red', alpha=0.2)
                                
                        else:  # fault
                            # For faults, show voltage and frequency
                            ax2.plot(day_data['hour'], day_data['voltage'], label='Voltage', color='purple')
                            ax2.set_title("Voltage Variation")
                            ax2.set_ylabel("Voltage (V)")
                            ax2.set_xlabel("Hour of Day")
                            
                            # Add frequency as second y-axis
                            ax3 = ax2.twinx()
                            ax3.plot(day_data['hour'], day_data['frequency'], label='Frequency', color='brown', linestyle='-.')
                            ax3.set_ylabel("Frequency (Hz)")
                            
                            # Combine legends
                            lines1, labels1 = ax2.get_legend_handles_labels()
                            lines2, labels2 = ax3.get_legend_handles_labels()
                            ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
                            
                            for hour in anomaly_hours:
                                ax2.axvspan(hour-0.5, hour+0.5, color='red', alpha=0.2)
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                        
                        # Show data for the day
                        if anomaly_type == "theft":
                            # For theft, show the relevant columns
                            theft_columns = ['hour', 'total_power', 'reported_power', 'theft_flag', 'theft_reason']
                            st.dataframe(day_data[theft_columns])
                        else:  # fault
                            # For faults, show the relevant columns
                            fault_columns = ['hour', 'voltage', 'frequency', 'thd', 'fault_flag', 'fault_reason']
                            st.dataframe(day_data[fault_columns])
            else:
                st.write("No anomalies detected in the testing period.")
        else:
            st.write("No anomalies detected or test data not available.")

def render_efficiency_analysis(house_id):
    """Render the efficiency analysis UI"""
    st.title("Energy Efficiency Analysis")
    
    # Check if models are trained
    if not st.session_state.model_training_complete:
        st.warning("Please train the models first from the Dashboard.")
        return
    
    # Explanation
    st.info("""
    This section analyzes the energy efficiency of appliances and provides recommendations for improvement.
    It uses machine learning models and traditional statistical analysis.
    """)
    
    # Filter data for selected house
    house_data = st.session_state.dataset[st.session_state.dataset['house_id'] == house_id]
    
    # Calculate efficiency metrics
    st.subheader("Appliance Efficiency Metrics")
    
    # Appliance usage and efficiency
    appliance_columns = [app.lower() for app in APPLIANCES]
    appliance_data = house_data[appliance_columns].copy()
    
    # Calculate metrics
    appliance_metrics = {}
    for i, app in enumerate(APPLIANCES):
        app_lower = app.lower()
        data = house_data[app_lower]
        
        # Calculate metrics
        metrics = {
            'peak_power': data.max(),
            'average_power': data.mean(),
            'usage_hours': (data > 10).sum() / 24,  # Approximate hours used per day (when power > 10W)
            'total_energy': data.sum() / 1000,  # kWh
            'standby_power': data[data < data.max() * 0.1].mean(),  # Approximate standby power
        }
        
        # Calculate efficiency score (lower is better)
        # This is a simplified metric combining usage patterns and power consumption
        efficiency_score = 0
        if metrics['usage_hours'] > 0:
            efficiency_score = metrics['total_energy'] / metrics['usage_hours']
        
        metrics['efficiency_score'] = efficiency_score
        
        appliance_metrics[app] = metrics
    
    # Calculate overall efficiency score
    total_energy = sum(metrics['total_energy'] for metrics in appliance_metrics.values())
    total_usage_hours = sum(metrics['usage_hours'] for metrics in appliance_metrics.values())
    overall_efficiency = total_energy / total_usage_hours if total_usage_hours > 0 else 0
    
    # Create radar chart for efficiency scores
    st.subheader("Efficiency Comparison")
    
    # Convert efficiency scores to a 0-100 scale where 100 is most efficient
    max_score = max(metrics['efficiency_score'] for metrics in appliance_metrics.values())
    efficiency_normalized = {
        app: 100 * (1 - (metrics['efficiency_score'] / max_score)) 
        for app, metrics in appliance_metrics.items()
    }
    
    # Create radar chart
    categories = list(efficiency_normalized.keys())
    values = list(efficiency_normalized.values())
    
    # Close the loop
    values.append(values[0])
    categories = [*categories, categories[0]]
    
    # Create figure
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, polar=True)
    
    # Plot radar chart
    angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]  # Close the loop
    
    ax.plot(angles, values, 'o-', linewidth=2, label='Efficiency Score')
    ax.fill(angles, values, alpha=0.25)
    
    # Set category labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories[:-1])
    
    # Set y limits
    ax.set_ylim(0, 100)
    
    # Add title
    plt.title('Appliance Efficiency Scores', size=15)
    
    # Show the plot
    st.pyplot(fig)
    
    # Show metrics in a table
    st.subheader("Detailed Metrics")
    
    # Create DataFrame for display
    metrics_df = pd.DataFrame(
        {app: metrics for app, metrics in appliance_metrics.items()}
    ).T
    
    st.dataframe(metrics_df)
    
    # Recommendations based on efficiency
    st.subheader("Efficiency Recommendations")
    
    # Sort appliances by efficiency score (higher values indicate less efficient)
    sorted_apps = sorted(
        [(app, metrics['efficiency_score']) for app, metrics in appliance_metrics.items()],
        key=lambda x: x[1],
        reverse=True
    )
    
    # Generate recommendations for least efficient appliances
    for app, score in sorted_apps[:3]:
        metrics = appliance_metrics[app]
        
        with st.expander(f"Recommendations for {app}"):
            st.write(f"### {app} Efficiency Analysis")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Peak Power", f"{metrics['peak_power']:.1f} W")
            with col2:
                st.metric("Average Power", f"{metrics['average_power']:.1f} W")
            with col3:
                st.metric("Standby Power", f"{metrics['standby_power']:.1f} W")
            
            # Generate recommendations based on the metrics
            recommendations = []
            
            if metrics['standby_power'] > 5:
                recommendations.append(f"Consider unplugging the {app} when not in use to save {metrics['standby_power']:.1f} watts of standby power.")
            
            if metrics['peak_power'] > 2 * metrics['average_power']:
                recommendations.append(f"The {app} has high peak power ({metrics['peak_power']:.1f} W). Consider using it during off-peak hours to reduce demand charges.")
            
            if app.lower() == 'refrigerator' and metrics['average_power'] > 70:
                recommendations.append("Your refrigerator's power consumption is above average. Check the door seals and temperature settings.")
            
            if app.lower() == 'air_conditioner' and metrics['average_power'] > 1000:
                recommendations.append("Consider setting your air conditioner to a slightly higher temperature to reduce power consumption.")
            
            if app.lower() == 'lighting' and metrics['average_power'] > 100:
                recommendations.append("Consider switching to LED lighting to reduce power consumption by up to 75%.")
            
            # Add generic recommendation if none were generated
            if not recommendations:
                recommendations.append(f"The {app} is operating relatively efficiently, but regularly check for signs of wear or reduced performance.")
            
            # Display recommendations
            st.write("### Recommendations:")
            for rec in recommendations:
                st.write(f"- {rec}")
            
            # Show usage patterns
            st.write("### Usage Pattern:")
            
            # Create hourly usage heatmap
            usage_by_hour = []
            for day in range(1, 29):
                day_data = house_data[house_data['day'] == day]
                if len(day_data) > 0:
                    usage_by_hour.append(day_data[app.lower()].values)
            
            if usage_by_hour:
                # Convert to numpy array and reshape
                usage_array = np.vstack(usage_by_hour)
                
                # Create heatmap
                fig, ax = plt.subplots(figsize=(12, 6))
                sns.heatmap(
                    usage_array, 
                    cmap='YlOrRd', 
                    ax=ax,
                    xticklabels=range(24),
                    yticklabels=range(1, 29),
                    cbar_kws={'label': 'Power (W)'}
                )
                ax.set_title(f"{app} Usage Pattern (W)")
                ax.set_xlabel("Hour of Day")
                ax.set_ylabel("Day")
                st.pyplot(fig)
    
    # Show overall model performance
    st.subheader("Model Performance")
    
    # If we have metrics, display them
    if 'metrics' in st.session_state and 'overall' in st.session_state.metrics:
        overall_metrics = st.session_state.metrics['overall']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("### Training Performance")
            st.metric("Total Training Time", f"{overall_metrics['total_training_time']:.2f} seconds")
            
            # Show model sizes if available
            if 'model_sizes' in overall_metrics:
                for model, size in overall_metrics['model_sizes'].items():
                    st.metric(f"{model.upper()} Model Size", f"{size:,} parameters")
        
        with col2:
            st.write("### Inference Performance")
            st.metric("Total Inference Time", f"{overall_metrics['inference_time']:.4f} seconds")
            
            # Show accuracies if available
            if 'accuracies' in overall_metrics:
                for model, acc in overall_metrics['accuracies'].items():
                    if model == 'classification':
                        st.metric("Classification Accuracy", f"{acc:.2%}")
                    elif model == 'regression':
                        st.metric("Regression R²", f"{acc:.4f}")

def render_data_explorer(house_id):
    """Render the data explorer UI"""
    st.title("Energy Data Explorer")
    
    # Filter data for selected house
    house_data = st.session_state.dataset[st.session_state.dataset['house_id'] == house_id]
    
    # Allow user to select columns to view
    st.subheader("Data Selection")
    
    # Group columns by category
    column_groups = {
        "Time": ['timestamp', 'day', 'hour', 'is_weekend', 'day_of_week'],
        "Power": ['total_power', 'reported_power', 'reactive_power', 'apparent_power', 'baseline_power'],
        "Grid": ['voltage', 'current', 'frequency', 'power_factor', 'thd'],
        "Appliances": [app.lower() for app in APPLIANCES],
        "Anomalies": ['theft_flag', 'theft_factor', 'theft_reason', 'fault_flag', 'fault_reason']
    }
    
    # Let user select column groups
    selected_groups = st.multiselect(
        "Select data categories to view",
        options=list(column_groups.keys()),
        default=["Time", "Power"]
    )
    
    # Flatten selected columns
    selected_columns = ['house_id']  # Always include house_id
    for group in selected_groups:
        selected_columns.extend(column_groups[group])
    
    # Show the data
    st.subheader("Data Table")
    
    # Let user filter by day
    day_filter = st.slider(
        "Filter by day",
        min_value=1,
        max_value=st.session_state.data_generator.num_days,
        value=(1, st.session_state.data_generator.num_days)
    )
    
    # Apply day filter
    filtered_data = house_data[
        (house_data['day'] >= day_filter[0]) &
        (house_data['day'] <= day_filter[1])
    ][selected_columns].copy()
    
    # Show data
    st.dataframe(filtered_data)
    
    # Download link
    st.download_button(
        label="Download as CSV",
        data=filtered_data.to_csv(index=False).encode('utf-8'),
        file_name=f"house_{house_id}_data.csv",
        mime="text/csv"
    )
    
    # Visualizations
    st.subheader("Interactive Visualizations")
    
    # Let user choose columns to plot
    numeric_columns = filtered_data.select_dtypes(include=['float64', 'int64']).columns.tolist()
    
    if 'timestamp' in filtered_data.columns:
        # Time series visualization
        x_axis = 'timestamp'
        y_axes = st.multiselect(
            "Select columns to plot",
            options=numeric_columns,
            default=['total_power'] if 'total_power' in numeric_columns else numeric_columns[:1]
        )
        
        if y_axes:
            # Create time series plot
            fig, ax = plt.subplots(figsize=(12, 6))
            
            for col in y_axes:
                ax.plot(filtered_data[x_axis], filtered_data[col], label=col)
            
            ax.set_xlabel("Time")
            ax.set_ylabel("Value")
            ax.set_title("Time Series Visualization")
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Format x-axis for dates
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
            
            plt.tight_layout()
            st.pyplot(fig)
    
    # Correlation analysis
    st.subheader("Correlation Analysis")
    
    # Select columns for correlation
    corr_columns = st.multiselect(
        "Select columns for correlation analysis",
        options=numeric_columns,
        default=[col for col in ['total_power', 'voltage', 'current', 'power_factor'] if col in numeric_columns]
    )
    
    if corr_columns and len(corr_columns) > 1:
        # Calculate correlation matrix
        corr_matrix = filtered_data[corr_columns].corr()
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            corr_matrix, 
            annot=True, 
            cmap='coolwarm', 
            vmin=-1, 
            vmax=1, 
            center=0,
            ax=ax
        )
        ax.set_title("Correlation Matrix")
        plt.tight_layout()
        st.pyplot(fig)
    
    # Statistical summary
    st.subheader("Statistical Summary")
    
    # Show summary statistics
    if numeric_columns:
        stats_df = filtered_data[numeric_columns].describe()
        st.dataframe(stats_df)

# ------------------------------------
# Main Application
# ------------------------------------
def main():
    """Main application function"""
    # Initialize session state
    if 'deep_learning_models_ready' not in st.session_state:
        st.session_state.deep_learning_models_ready = False
    
    # Render sidebar for navigation
    page = render_sidebar()
    
    # Get current house ID
    house_id = st.session_state.current_house
    
    # Display selected page
    if page == "Dashboard Overview":
        render_dashboard(house_id)
    elif page == "Load Disaggregation":
        render_appliance_disaggregation(house_id)
    elif page == "Theft Detection":
        render_anomaly_detection(house_id, anomaly_type="theft")
    elif page == "Fault Detection":
        render_anomaly_detection(house_id, anomaly_type="fault")
    elif page == "Efficiency Analysis":
        render_efficiency_analysis(house_id)
    elif page == "Data Explorer":
        render_data_explorer(house_id)

if __name__ == "__main__":
    main()
