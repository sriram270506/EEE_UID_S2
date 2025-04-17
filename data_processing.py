import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

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
        Preprocess data for deep learning models
        
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
        # Check if there are 'theft_flag' and 'fault_flag' columns
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
        Create sequences for autoencoder anomaly detection
        
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
