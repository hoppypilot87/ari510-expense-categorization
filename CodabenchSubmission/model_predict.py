"""
model_predict.py

Utility functions to load the trained expense category model and
make predictions for new transactions.

This script assumes:
- The tuned model + scaler were saved in ../models/ from notebooks/03_model_tuning.ipynb
- Label indices follow the mapping printed during training:
    0: Bills
    1: DiningOut
    2: Education
    3: Entertainment
    4: Groceries
    5: Healthcare
    6: Miscellaneous
    7: PersonalCare
    8: Shopping
    9: Transport
   10: Travel
   11: Utilities
"""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd


# -------------------------------------------------------------------
# Paths and global config
# -------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR

MODEL_PATH = MODELS_DIR / "best_category_model.pkl"
SCALER_PATH = MODELS_DIR / "category_scaler.pkl"

# This must match the label encoder mapping used in training
LABEL_CLASSES = [
    "Bills",
    "DiningOut",
    "Education",
    "Entertainment",
    "Groceries",
    "Healthcare",
    "Miscellaneous",
    "PersonalCare",
    "Shopping",
    "Transport",
    "Travel",
    "Utilities",
]


# -------------------------------------------------------------------
# Load artifacts
# -------------------------------------------------------------------

def load_artifacts():
    """Load trained model and scaler from disk."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
    if not SCALER_PATH.exists():
        raise FileNotFoundError(f"Scaler file not found at {SCALER_PATH}")

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler


# -------------------------------------------------------------------
# Core prediction helper
# -------------------------------------------------------------------

def predict_category_from_vector(x_numeric: np.ndarray) -> str:
    """
    Predict a category given a 1D array of numeric features in the
    SAME ORDER as used during training.

    Parameters
    ----------
    x_numeric : np.ndarray
        Shape (n_features,) – raw numeric features.

    Returns
    -------
    str
        Predicted category label.
    """
    model, scaler = load_artifacts()

    # Ensure shape (1, n_features)
    x_numeric = np.asarray(x_numeric).reshape(1, -1)

    # Build a DataFrame with the SAME feature names the scaler was fit on
    # This avoids the "X does not have valid feature names" warning.
    if hasattr(scaler, "feature_names_in_"):
        feature_names = list(scaler.feature_names_in_)
        x_df = pd.DataFrame(x_numeric, columns=feature_names)
    else:
        # Fallback: no feature name info (shouldn't happen with our training setup)
        x_df = pd.DataFrame(x_numeric)

    x_scaled = scaler.transform(x_df)

    # Model was trained on encoded labels 0..11
    pred_idx = int(model.predict(x_scaled)[0])
    return LABEL_CLASSES[pred_idx]


# -------------------------------------------------------------------
# Convenience: predict for a random sample row from the dataset
# (useful for quick smoke tests)
# -------------------------------------------------------------------

def _build_feature_matrix(df: pd.DataFrame):
    """
    Replicate the feature construction used in 03_model_tuning.ipynb:
    - Drop ID-like columns
    - Drop any category / encoded columns
    - Keep only numeric features
    """
    TARGET_COL = "category"

    # Add transaction_id here so it doesn't go into the scaler
    ID_CANDIDATES = [
        "entity_id",
        "txn_id",
        "user_id",
        "index",
        "id",
        "transaction_id",   # <--- NEW
    ]

    drop_cols = []
    for c in df.columns:
        if c in ID_CANDIDATES:
            drop_cols.append(c)
        if "category" in c.lower() and c != TARGET_COL:
            drop_cols.append(c)

    df_num = (
        df.drop(columns=drop_cols, errors="ignore")
          .select_dtypes(include="number")
          .copy()
    )

    return df_num


def demo_random_prediction(n_samples: int = 5) -> None:
    """
    Load a few random rows from the processed dataset and print
    human-readable predictions for sanity checking.
    """
    data_path = BASE_DIR / "data" / "processed" / "transactions_long.csv"
    if not data_path.exists():
        raise FileNotFoundError(f"Processed data not found at {data_path}")

    df = pd.read_csv(data_path)
    df_num = _build_feature_matrix(df)

    # Align indices so we can access the true category
    df_num["__row_index__"] = df_num.index
    df_merged = df_num.merge(
        df[["category"]],
        left_on="__row_index__",
        right_index=True,
        how="left",
    )
    X = df_merged.drop(columns=["__row_index__", "category"])
    y_true = df_merged["category"]

    model, scaler = load_artifacts()

    # IMPORTANT: keep X as a DataFrame so the scaler sees column names.
    X_scaled = scaler.transform(X)

    # Model predicts on the scaled DataFrame
    y_pred_idx = model.predict(X_scaled)

    print("=== Demo predictions on random subset ===")
    sample_idx = np.random.choice(len(X), size=min(n_samples, len(X)), replace=False)
    for i in sample_idx:
        true_label = y_true.iloc[i]
        pred_label = LABEL_CLASSES[int(y_pred_idx[i])]
        print(f"Row {i}: true = {true_label:12s} | predicted = {pred_label:12s}")


# -------------------------------------------------------------------
# CLI entry point for quick testing
# -------------------------------------------------------------------

if __name__ == "__main__":
    demo_random_prediction(n_samples=5)