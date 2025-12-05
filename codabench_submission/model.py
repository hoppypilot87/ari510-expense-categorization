"""
Codabench model for expense categorization.

This implementation mirrors the final course project pipeline:

- Text features:
    - vendor        -> TfidfVectorizer
    - description   -> TfidfVectorizer
- Numeric features:
    - amount        -> StandardScaler
- Categorical features:
    - payment_method, city, state -> OneHotEncoder
cp model.py codabench_submission/model.py
The classifier is a multinomial Logistic Regression model.
Codabench will call:

    model = Model()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

where X_* are tables with the columns above and y_* are
integer-encoded labels (e.g., category_encoded).
"""

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


class Model:
    """Logistic Regression text-aware model for expense categorization."""

    def __init__(self):
        # Columns expected in X (Codabench input)
        self.text_cols = ["vendor", "description"]
        self.num_cols = ["amount"]
        self.cat_cols = ["payment_method", "city", "state"]

        # Build the preprocessing + model pipeline
        self.pipeline = Pipeline(
            steps=[
                (
                    "preprocess",
                    ColumnTransformer(
                        transformers=[
                            # TF-IDF for vendor and description
                            ("vendor_tfidf", TfidfVectorizer(max_features=3000), "vendor"),
                            ("desc_tfidf", TfidfVectorizer(max_features=5000), "description"),
                            # Scale numeric amount
                            ("num", StandardScaler(), self.num_cols),
                            # One-hot encode categorical fields
                            ("cat", OneHotEncoder(handle_unknown="ignore"), self.cat_cols),
                        ],
                        remainder="drop",
                    ),
                ),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=1000,
                        multi_class="multinomial",
                        n_jobs=-1,
                    ),
                ),
            ]
        )

    # -------------------------------
    # Internal helper: ensure DataFrame
    # -------------------------------
    def _ensure_dataframe(self, X):
        """
        Codabench may pass a NumPy array or a DataFrame.
        We prefer a DataFrame with known column names so
        the ColumnTransformer can select the right features.
        """
        if isinstance(X, pd.DataFrame):
            return X

        # If X is not a DataFrame, assume the columns are in the
        # following order (must match the Codabench task setup):
        #   vendor, description, amount, payment_method, city, state
        cols = self.text_cols + self.num_cols + self.cat_cols
        return pd.DataFrame(X, columns=cols)

    # -------------------------------
    # Required Codabench API methods
    # -------------------------------
    def fit(self, X_train, y_train):
        """
        Train the model on the provided training data.

        Parameters
        ----------
        X_train : array-like or DataFrame
            Training features with columns:
            vendor, description, amount, payment_method, city, state.
        y_train : array-like
            Integer-encoded category labels.
        """
        X_train_df = self._ensure_dataframe(X_train)
        self.pipeline.fit(X_train_df, y_train)
        return self

    def predict(self, X_test):
        """
        Generate predictions for the provided test data.

        Parameters
        ----------
        X_test : array-like or DataFrame
            Test features with the same columns as X_train.

        Returns
        -------
        np.ndarray
            1D array of predicted integer labels.
        """
        X_test_df = self._ensure_dataframe(X_test)
        y_pred = self.pipeline.predict(X_test_df)
        return np.asarray(y_pred)