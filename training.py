import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import time
import os

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
        
        # 1. Train CNN+BiLSTM for NILM
        print("Training CNN+BiLSTM model for NILM...")
        nilm_model = self._train_nilm_model(X_train, y_train, X_test, y_test)
        self.models['nilm'] = nilm_model
        
        # 2. Train Autoencoder+GRU for anomaly detection
        print("Training Autoencoder+GRU model for anomaly detection...")
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
        """Train and evaluate CNN+BiLSTM model for NILM"""
        from models import CNNBiLSTM
        
        # Get input shape from data
        sequence_length, n_features = X_train.shape[1], X_train.shape[2]
        n_appliances = y_train.shape[1]
        
        # Create and train model
        model = CNNBiLSTM(sequence_length=sequence_length, n_features=n_features, n_appliances=n_appliances)
        model.build_model()
        model.train(X_train, y_train, X_test, y_test, epochs=10, batch_size=32)
        
        # Get metrics
        self.metrics['nilm'] = {
            'efficiency': model.get_efficiency_metrics(),
            'history': model.history.history if model.history else None
        }
        
        return model
    
    def _train_anomaly_model(self, X_train, X_test):
        """Train and evaluate Autoencoder+GRU model for anomaly detection"""
        from models import AutoencoderGRU
        
        # Get input shape from data
        sequence_length, n_features = X_train.shape[1], X_train.shape[2]
        
        # Create and train model
        model = AutoencoderGRU(sequence_length=sequence_length, n_features=n_features)
        model.build_model()
        model.train(X_train, epochs=10, batch_size=32)
        
        # Test model
        test_results = model.detect_anomalies(X_test)
        
        # Get metrics
        self.metrics['anomaly'] = {
            'efficiency': model.get_efficiency_metrics(),
            'anomalies_detected': sum(test_results['anomalies']),
            'history': model.history.history if model.history else None
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
        
        # Collect metrics from deep learning models
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
        
        # Add overall metrics to the metrics dictionary
        self.metrics['overall'] = overall_metrics
        
        return overall_metrics
    
    def save_all_models(self, base_path="models"):
        """
        Save all trained models to disk
        
        Args:
            base_path: Base directory to save models
        """
        os.makedirs(base_path, exist_ok=True)
        
        # Save deep learning models
        if 'nilm' in self.models:
            self.models['nilm'].save(f"{base_path}/nilm_cnn_bilstm.h5")
        
        if 'anomaly' in self.models:
            self.models['anomaly'].save(f"{base_path}/anomaly_autoencoder_gru.h5")
        
        # Save traditional ML models
        if 'rf_classification' in self.models:
            self.models['rf_classification'].save(f"{base_path}/rf_classification.joblib")
        
        if 'xgb_regression' in self.models:
            self.models['xgb_regression'].save(f"{base_path}/xgb_regression.joblib")
        
        # Save metrics
        joblib.dump(self.metrics, f"{base_path}/model_metrics.joblib")
        
        print(f"All models saved to {base_path}")
    
    def load_all_models(self, base_path="models"):
        """
        Load all models from disk
        
        Args:
            base_path: Base directory to load models from
            
        Returns:
            Dictionary of loaded models
        """
        from models import CNNBiLSTM, AutoencoderGRU
        
        # Load deep learning models if files exist
        nilm_path = f"{base_path}/nilm_cnn_bilstm.h5"
        if os.path.exists(nilm_path):
            self.models['nilm'] = CNNBiLSTM()
            self.models['nilm'].load(nilm_path)
        
        anomaly_path = f"{base_path}/anomaly_autoencoder_gru.h5"
        if os.path.exists(anomaly_path):
            self.models['anomaly'] = AutoencoderGRU()
            self.models['anomaly'].load(anomaly_path)
        
        # Load traditional ML models
        rf_path = f"{base_path}/rf_classification.joblib"
        if os.path.exists(rf_path):
            self.models['rf_classification'] = DecisionModel(model_type="classification", algorithm="rf")
            self.models['rf_classification'].load(rf_path)
        
        xgb_path = f"{base_path}/xgb_regression.joblib"
        if os.path.exists(xgb_path):
            self.models['xgb_regression'] = DecisionModel(model_type="regression", algorithm="xgb")
            self.models['xgb_regression'].load(xgb_path)
        
        # Load metrics if available
        metrics_path = f"{base_path}/model_metrics.joblib"
        if os.path.exists(metrics_path):
            self.metrics = joblib.load(metrics_path)
        
        return self.models
