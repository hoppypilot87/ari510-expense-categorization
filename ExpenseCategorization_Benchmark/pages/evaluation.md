# Evaluation Criteria

## Metrics

Your model will be evaluated using multiple classification metrics:

### Primary Metric: Accuracy
The main ranking metric is classification accuracy - the percentage of correctly classified transactions.

### Secondary Metrics

1. **F1 Score (Weighted)**: Weighted average F1 score across all categories
2. **F1 Score (Macro)**: Unweighted average F1 score across all categories
3. **Precision (Weighted)**: Weighted average precision
4. **Recall (Weighted)**: Weighted average recall
5. **Duration**: Training + prediction time (lower is better)

## Scoring Process

1. Your model is initialized with the `Model()` constructor
2. The `fit(X_train, y_train)` method is called with training data
3. The `predict(X_test)` method is called to generate predictions
4. Predictions are compared against ground truth labels
5. All metrics are calculated and displayed on the leaderboard

## Leaderboard

The leaderboard displays all metrics and ranks participants by accuracy.

### Tiebreakers

If two submissions have the same accuracy:
1. Higher F1 Score (Weighted)
2. Lower Duration

## What Gets Scored

- Integer predictions from 0 to 11
- Predictions must match the number of test samples
- Invalid predictions will receive a score of 0
