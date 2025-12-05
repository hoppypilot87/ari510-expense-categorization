# Starting Kit - Expense Categorization Challenge

This starting kit contains a baseline solution to help you get started.

## Files

- `model.py`: Baseline Random Forest model
- `requirements.txt`: Required Python packages

## How to Use

1. **Test locally** (optional):
   - Download the public data
   - Run the ingestion and scoring programs locally
   - Verify your model works

2. **Modify the model**:
   - Edit `model.py` to implement your own solution
   - Keep the same class structure (Model with fit() and predict() methods)
   - Add any required packages to `requirements.txt`

3. **Submit**:
   - Zip `model.py` and `requirements.txt`
   - Upload to the competition platform
   - Check the leaderboard for results

## Model Requirements

Your `Model` class must have:

```python
class Model:
    def __init__(self):
        # Initialize your model
        pass

    def fit(self, X_train, y_train):
        # Train on data
        # X_train: numpy array (n_samples, n_features)
        # y_train: numpy array (n_samples,) with integers 0-11
        pass

    def predict(self, X_test):
        # Return predictions
        # X_test: numpy array (n_samples, n_features)
        # Returns: numpy array (n_samples,) with integers 0-11
        return predictions
```

## Tips for Improvement

1. **Feature Engineering**: Create additional features from the data
2. **Hyperparameter Tuning**: Optimize model parameters
3. **Different Algorithms**: Try XGBoost, LightGBM, Neural Networks
4. **Ensemble Methods**: Combine multiple models
5. **Cross-validation**: Use proper validation to avoid overfitting

## Testing Locally

You can test your model before submitting:

```python
import numpy as np
from model import Model

# Load data
X_train = np.genfromtxt('training_data')
y_train = np.genfromtxt('training_label')
X_test = np.genfromtxt('testing_data')

# Train and predict
m = Model()
m.fit(X_train, y_train)
predictions = m.predict(X_test)

# Evaluate
from sklearn.metrics import accuracy_score
y_test = np.genfromtxt('testing_label')  # Not available in competition
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy:.4f}")
```

Good luck!
