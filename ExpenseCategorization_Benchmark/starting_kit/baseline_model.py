"""
Baseline Model for Expense Categorization Competition

This is a simple baseline using a Random Forest classifier.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


class Model:
    """Simple Random Forest baseline for expense categorization."""

    def __init__(self):
        # Columns expected in X
        self.text_cols = ["vendor", "description"]
        self.num_cols = ["amount"]
        self.cat_cols = ["payment_method", "city", "state"]

        # Build a simple pipeline
        self.pipeline = Pipeline([
            ("preprocess", ColumnTransformer([
                ("vendor_tfidf", TfidfVectorizer(max_features=1000), "vendor"),
                ("desc_tfidf", TfidfVectorizer(max_features=2000), "description"),
                ("num", StandardScaler(), self.num_cols),
                ("cat", OneHotEncoder(handle_unknown="ignore"), self.cat_cols),
            ])),
            ("clf", RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)),
        ])

    def _ensure_dataframe(self, X):
        """Convert to DataFrame if needed."""
        if isinstance(X, pd.DataFrame):
            return X
        cols = self.text_cols + self.num_cols + self.cat_cols
        return pd.DataFrame(X, columns=cols)

    def fit(self, X_train, y_train):
        """Train the model."""
        X_train_df = self._ensure_dataframe(X_train)
        self.pipeline.fit(X_train_df, y_train)
        return self

    def predict(self, X_test):
        """Generate predictions."""
        X_test_df = self._ensure_dataframe(X_test)
        return np.asarray(self.pipeline.predict(X_test_df))
