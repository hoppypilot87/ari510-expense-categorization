"""
Baseline Model for Expense Categorization Challenge

This is a simple baseline solution using Random Forest.
Use this as a starting point for your own solution!
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler


class Model:
    """
    Baseline Random Forest model for expense categorization.

    This model:
    - Scales features using StandardScaler
    - Uses a Random Forest classifier with default parameters
    - Achieves reasonable baseline performance

    You can improve this by:
    - Feature engineering
    - Hyperparameter tuning
    - Using different algorithms (XGBoost, Neural Networks, etc.)
    - Ensemble methods
    """

    def __init__(self):
        """Initialize the model and scaler."""
        self.scaler = StandardScaler()
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )

    def fit(self, X_train, y_train):
        """
        Train the model on the provided data.

        Parameters
        ----------
        X_train : numpy.ndarray
            Training features, shape (n_samples, n_features)
        y_train : numpy.ndarray
            Training labels, shape (n_samples,) with integer values 0-11
        """
        # Handle 1D arrays (single feature)
        if len(X_train.shape) == 1:
            X_train = X_train.reshape(-1, 1)

        # Scale the features
        X_scaled = self.scaler.fit_transform(X_train)

        # Train the model
        self.model.fit(X_scaled, y_train)

    def predict(self, X_test):
        """
        Generate predictions for test data.

        Parameters
        ----------
        X_test : numpy.ndarray
            Test features, shape (n_samples, n_features)

        Returns
        -------
        numpy.ndarray
            Predicted labels, shape (n_samples,) with integer values 0-11
        """
        # Handle 1D arrays (single feature)
        if len(X_test.shape) == 1:
            X_test = X_test.reshape(-1, 1)

        # Scale the features
        X_scaled = self.scaler.transform(X_test)

        # Generate predictions
        predictions = self.model.predict(X_scaled)

        return predictions
