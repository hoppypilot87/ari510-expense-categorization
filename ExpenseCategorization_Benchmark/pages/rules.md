# Competition Rules

## Submission Requirements

### File Structure

Your submission must be a ZIP file containing:
- **model.py** (required): Python file with your Model class
- **requirements.txt** (optional): Python package dependencies

### Model Class Requirements

Your `model.py` must contain a class named `Model` with the following methods:

```python
class Model:
    def __init__(self):
        # Initialize your model
        pass

    def fit(self, X_train, y_train):
        # Train your model
        # X_train: numpy array of shape (n_samples, n_features)
        # y_train: numpy array of shape (n_samples,) with integer labels
        pass

    def predict(self, X_test):
        # Generate predictions
        # X_test: numpy array of shape (n_samples, n_features)
        # Returns: numpy array of shape (n_samples,) with integer predictions
        return predictions
```

## Allowed Libraries

You may use any Python libraries available in the competition Docker environment, including:
- numpy
- pandas
- scikit-learn
- scipy
- xgboost
- lightgbm
- tensorflow
- pytorch

If you need additional libraries, specify them in `requirements.txt`.

## Submission Limits

- **Maximum submissions per day**: 5
- **Total maximum submissions**: 100
- **Execution time limit**: 10 minutes
- **Memory limit**: 8 GB

## Code of Conduct

- Do not attempt to access or manipulate test labels
- Do not submit pre-trained models that use external data
- Model must train from scratch using only the provided training data
- Fair play and academic integrity are expected

## Disqualification

Submissions will be disqualified for:
- Attempting to hack the evaluation system
- Using test set labels during training
- Sharing solutions publicly during the competition
- Violating submission limits

## Questions

If you have questions about the rules or technical issues, please use the competition forum.
