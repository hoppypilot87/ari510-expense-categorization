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
import pandas as pd

# Codabench standard paths
input_dir = '/app/input_data/'
output_dir = '/app/output/'
program_dir = '/app/program'
submission_dir = '/app/ingested_program'

sys.path.append(program_dir)
sys.path.append(submission_dir)


def get_training_data():
    """Load training data and labels (CSV format with text features)."""
    print(f'  Reading from: {input_dir}')
    print(f'  Files in input_dir: {os.listdir(input_dir)}')

    train_data_path = os.path.join(input_dir, 'training_data.csv')
    train_label_path = os.path.join(input_dir, 'training_label')

    print(f'  Loading training data from: {train_data_path}')
    X_train = pd.read_csv(train_data_path)
    print(f'  Training data shape: {X_train.shape}')
    print(f'  Training data columns: {list(X_train.columns)}')

    print(f'  Loading training labels from: {train_label_path}')
    y_train = np.genfromtxt(train_label_path)
    print(f'  Training labels shape: {y_train.shape}')

    return X_train, y_train


def get_prediction_data():
    """Load test data (CSV format with text features)."""
    test_data_path = os.path.join(input_dir, 'testing_data.csv')
    print(f'  Loading test data from: {test_data_path}')
    X_test = pd.read_csv(test_data_path)
    print(f'  Test data shape: {X_test.shape}')
    print(f'  Test data columns: {list(X_test.columns)}')
    return X_test


def main():
    """Main ingestion workflow."""
    try:
        print('=' * 50)
        print('EXPENSE CATEGORIZATION - INGESTION PROGRAM')
        print('=' * 50)

        # Debug: Show environment info
        print('\n[DEBUG] Environment Info:')
        print(f'  Python version: {sys.version}')
        print(f'  Input dir: {input_dir}')
        print(f'  Output dir: {output_dir}')
        print(f'  Submission dir: {submission_dir}')
        print(f'  Files in submission dir: {os.listdir(submission_dir)}')

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
        print('Model initialized')

        # Train model
        print('\n[4/5] Training Model...')
        print(f'  X_train type: {type(X_train)}')
        print(f'  y_train type: {type(y_train)}')
        print(f'  First row of X_train: {X_train.iloc[0].to_dict()}')
        m.fit(X_train, y_train)
        print('Model training complete')

        # Generate predictions
        print('\n[5/5] Generating Predictions...')
        predictions = m.predict(X_test)
        duration = time.time() - start

        print(f'\nTotal duration: {duration:.2f} seconds')
        print(f'Predictions shape: {predictions.shape}')
        print(f'Predictions dtype: {predictions.dtype}')
        print(f'Predictions range: {predictions.min()} to {predictions.max()}')
        print(f'Unique predictions: {len(np.unique(predictions))}')

        # Save predictions
        print('\nSaving results...')
        output_file = os.path.join(output_dir, 'prediction')
        print(f'  Saving predictions to: {output_file}')
        np.savetxt(output_file, predictions, fmt='%d')
        print(f'  Predictions saved. File size: {os.path.getsize(output_file)} bytes')

        # Save metadata
        metadata_file = os.path.join(output_dir, 'metadata.json')
        print(f'  Saving metadata to: {metadata_file}')
        with open(metadata_file, 'w') as f:
            json.dump({
                'duration': duration,
                'train_samples': int(X_train.shape[0]),
                'test_samples': int(X_test.shape[0]),
                'features': int(X_train.shape[1] if len(X_train.shape) > 1 else 1)
            }, f, indent=2)
        print(f'  Metadata saved')

        # Verify files exist
        print('\n[VERIFICATION] Checking output files...')
        print(f'  Files in output dir: {os.listdir(output_dir)}')

        print('\n' + '=' * 50)
        print('INGESTION COMPLETED SUCCESSFULLY')
        print('=' * 50)

    except Exception as e:
        print(f'\n!!! INGESTION FAILED !!!')
        print(f'Error: {type(e).__name__}: {str(e)}')
        import traceback
        traceback.print_exc()

        # Create an error indicator file
        try:
            with open(os.path.join(output_dir, 'error.txt'), 'w') as f:
                f.write(f'{type(e).__name__}: {str(e)}\n')
                f.write(traceback.format_exc())
        except:
            pass

        sys.exit(1)


if __name__ == '__main__':
    main()
