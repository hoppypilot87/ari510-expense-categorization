"""
Ingestion Program for Expense Categorization Competition

This program:
1. Loads training and testing data
2. Imports the participant's Model class
3. Trains the model
4. Generates predictions
5. Saves predictions and metadata
"""
import json
import os
import sys
import time
import numpy as np

# Codabench standard paths
input_dir = '/app/input_data/'
output_dir = '/app/output/'
program_dir = '/app/program'
submission_dir = '/app/ingested_program'

sys.path.append(program_dir)
sys.path.append(submission_dir)


def get_training_data():
    """Load training data and labels."""
    X_train = np.genfromtxt(os.path.join(input_dir, 'training_data'))
    y_train = np.genfromtxt(os.path.join(input_dir, 'training_label'))
    return X_train, y_train


def get_prediction_data():
    """Load test data."""
    return np.genfromtxt(os.path.join(input_dir, 'testing_data'))


def main():
    """Main ingestion workflow."""
    try:
        print('=' * 50)
        print('EXPENSE CATEGORIZATION - INGESTION PROGRAM')
        print('=' * 50)

        # Import participant's model
        print('\n[1/5] Importing Model...')
        from model import Model
        print('Model imported successfully')

        # Load data
        print('\n[2/5] Loading Data...')
        X_train, y_train = get_training_data()
        X_test = get_prediction_data()
        print(f'  Training samples: {X_train.shape[0]}')
        print(f'  Testing samples: {X_test.shape[0]}')
        print(f'  Features: {X_train.shape[1] if len(X_train.shape) > 1 else 1}')

        # Initialize model
        print('\n[3/5] Initializing Model...')
        start = time.time()
        m = Model()

        # Train model
        print('\n[4/5] Training Model...')
        m.fit(X_train, y_train)

        # Generate predictions
        print('\n[5/5] Generating Predictions...')
        predictions = m.predict(X_test)
        duration = time.time() - start

        print(f'\nTotal duration: {duration:.2f} seconds')
        print(f'Predictions shape: {predictions.shape}')

        # Save predictions
        print('\nSaving results...')
        np.savetxt(os.path.join(output_dir, 'prediction'), predictions, fmt='%d')

        # Save metadata
        with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
            json.dump({
                'duration': duration,
                'train_samples': int(X_train.shape[0]),
                'test_samples': int(X_test.shape[0]),
                'features': int(X_train.shape[1] if len(X_train.shape) > 1 else 1)
            }, f, indent=2)

        print('\n' + '=' * 50)
        print('INGESTION COMPLETED SUCCESSFULLY')
        print('=' * 50)

    except Exception as e:
        print(f'\n!!! INGESTION FAILED !!!')
        print(f'Error: {type(e).__name__}: {str(e)}')
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
