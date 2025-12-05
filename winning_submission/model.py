"""
High-performance model for expense categorization.
Trains a Gradient Boosting classifier on the full feature set.
"""
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler


class Model:
    """Gradient Boosting model for expense categorization."""

    def __init__(self):
        """Initialize model and scaler."""
        self.scaler = StandardScaler()
        self.model = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
            verbose=0
        )
        print("Model initialized: Gradient Boosting (200 trees, depth=5)")

    def fit(self, X_train, y_train):
        """Train the model on provided data."""
        print(f"Training on {X_train.shape[0]} samples with {X_train.shape[1]} features")

        # Scale features
        X_scaled = self.scaler.fit_transform(X_train)

        # Train model
        self.model.fit(X_scaled, y_train)

        print("Training complete!")

    def predict(self, X_test):
        """Generate predictions."""
        print(f"Predicting {X_test.shape[0]} samples")

        # Scale features
        X_scaled = self.scaler.transform(X_test)

        # Predict
        predictions = self.model.predict(X_scaled)

        return predictions
