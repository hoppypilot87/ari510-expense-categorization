"""
model_predict.py

Utility functions to load the trained expense category pipeline
and make predictions for new transactions.

This version reflects the UPDATED MODEL:
- The saved model is a FULL PIPELINE (TF-IDF + OHE + Scaler + Logistic Regression)
- No external scaler is needed
- Model expects a DataFrame with columns:
    vendor, description, amount, payment_method, city, state
"""

from pathlib import Path
import joblib
import pandas as pd

# -------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"

MODEL_PATH = MODELS_DIR / "best_category_model.pkl"

# Updated label classes (same order as during training)
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
# Load model
# -------------------------------------------------------------------

def load_model():
    """Load the trained pipeline."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")

    model = joblib.load(MODEL_PATH)
    return model


# -------------------------------------------------------------------
# Core prediction functions
# -------------------------------------------------------------------

def _build_single_row(
    vendor: str,
    description: str,
    amount: float,
    payment_method: str,
    city: str,
    state: str,
) -> pd.DataFrame:
    """Build a one-row DataFrame in the same shape as training data."""
    return pd.DataFrame(
        [
            {
                "vendor": vendor,
                "description": description,
                "amount": amount,
                "payment_method": payment_method,
                "city": city,
                "state": state,
            }
        ]
    )


def predict_category_with_proba(
    vendor: str,
    description: str,
    amount: float,
    payment_method: str,
    city: str,
    state: str,
):
    """
    Predict spending category AND return per-class probabilities.

    Returns
    -------
    (str, dict)
        (best_label, {label: probability_float})
    """
    model = load_model()
    df = _build_single_row(vendor, description, amount, payment_method, city, state)

    # Predict class probabilities
    proba = model.predict_proba(df)[0]  # shape (n_classes,)

    # Best index + label
    best_idx = int(proba.argmax())
    best_label = LABEL_CLASSES[best_idx]

    # Map probabilities to label names for Gradio's Label component
    proba_dict = {
        label: float(p) for label, p in zip(LABEL_CLASSES, proba)
    }

    return best_label, proba_dict


def predict_category(
    vendor: str,
    description: str,
    amount: float,
    payment_method: str,
    city: str,
    state: str,
) -> str:
    """
    Backwards-compatible helper that only returns the label.
    """
    best_label, _ = predict_category_with_proba(
        vendor, description, amount, payment_method, city, state
    )
    return best_label


# -------------------------------------------------------------------
# Convenience test
# -------------------------------------------------------------------

def quick_test():
    """Run a quick test to verify model loads and predicts with probs."""

    example = {
        "vendor": "Shell Gas Station",
        "description": "Fuel purchase - premium",
        "amount": 42.18,
        "payment_method": "Credit Card",
        "city": "Detroit",
        "state": "MI",
    }

    label, proba = predict_category_with_proba(**example)
    print("Predicted Category:", label)
    print("Top probabilities:")
    for k, v in sorted(proba.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"  {k:12s}: {v:.3f}")


if __name__ == "__main__":
    quick_test()