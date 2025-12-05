"""
Scoring Program for Expense Categorization Competition

This program:
1. Loads predictions and ground truth
2. Calculates performance metrics
3. Saves scores
"""
import json
import os
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# Codabench standard paths
reference_dir = os.path.join('/app/input/', 'ref')
prediction_dir = os.path.join('/app/input/', 'res')
score_dir = '/app/output/'


def main():
    """Main scoring workflow."""
    try:
        print('=' * 50)
        print('EXPENSE CATEGORIZATION - SCORING PROGRAM')
        print('=' * 50)

        # Load predictions and ground truth
        print('\n[1/3] Loading predictions and ground truth...')
        prediction = np.genfromtxt(os.path.join(prediction_dir, 'prediction'))
        truth = np.genfromtxt(os.path.join(reference_dir, 'testing_label'))

        print(f'  Predictions: {len(prediction)} samples')
        print(f'  Ground truth: {len(truth)} samples')

        # Load metadata
        try:
            with open(os.path.join(prediction_dir, 'metadata.json')) as f:
                metadata = json.load(f)
                duration = metadata.get('duration', -1)
        except:
            duration = -1

        # Calculate metrics
        print('\n[2/3] Calculating metrics...')
        accuracy = accuracy_score(truth, prediction)

        # Try to use zero_division if available (scikit-learn >= 0.22)
        try:
            f1_macro = f1_score(truth, prediction, average='macro', zero_division=0)
            f1_weighted = f1_score(truth, prediction, average='weighted', zero_division=0)
            precision = precision_score(truth, prediction, average='weighted', zero_division=0)
            recall = recall_score(truth, prediction, average='weighted', zero_division=0)
        except TypeError:
            # Fallback for older scikit-learn versions
            f1_macro = f1_score(truth, prediction, average='macro')
            f1_weighted = f1_score(truth, prediction, average='weighted')
            precision = precision_score(truth, prediction, average='weighted')
            recall = recall_score(truth, prediction, average='weighted')

        print(f'  Accuracy: {accuracy:.4f}')
        print(f'  F1 (macro): {f1_macro:.4f}')
        print(f'  F1 (weighted): {f1_weighted:.4f}')
        print(f'  Precision: {precision:.4f}')
        print(f'  Recall: {recall:.4f}')
        print(f'  Duration: {duration:.2f}s')

        # Prepare scores
        scores = {
            'accuracy': float(accuracy),
            'f1_macro': float(f1_macro),
            'f1_weighted': float(f1_weighted),
            'precision': float(precision),
            'recall': float(recall),
            'duration': float(duration)
        }

        # Save scores
        print('\n[3/3] Saving scores...')
        with open(os.path.join(score_dir, 'scores.json'), 'w') as score_file:
            json.dump(scores, score_file, indent=2)

        print('\n' + '=' * 50)
        print('SCORING COMPLETED SUCCESSFULLY')
        print('=' * 50)

    except Exception as e:
        print(f'\n!!! SCORING FAILED !!!')
        print(f'Error: {type(e).__name__}: {str(e)}')
        import traceback
        traceback.print_exc()
        # Still write a scores file with error indicator
        scores = {'error': str(e)}
        with open(os.path.join(score_dir, 'scores.json'), 'w') as score_file:
            json.dump(scores, score_file)


if __name__ == '__main__':
    main()
