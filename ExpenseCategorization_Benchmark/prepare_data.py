"""
Prepare training and testing data for the Codabench competition.
This script splits the processed data and creates the required format.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os

# Load the processed data
data_path = '../data/processed/transactions_long.csv'
df = pd.read_csv(data_path)

print(f"Loaded {len(df)} transactions")
print(f"Categories: {df['category'].unique()}")

# Prepare features (drop non-feature columns)
feature_columns = ['amount']  # Add more features as needed
target_column = 'category'

# For this example, we'll use just the amount as a feature
# You can expand this to include engineered features from your notebooks
X = df[feature_columns].values
y = df[target_column].values

# Encode labels to integers
le = LabelEncoder()
y_encoded = le.fit_transform(y)

print(f"\nLabel mapping:")
for idx, label in enumerate(le.classes_):
    print(f"  {idx}: {label}")

# Split into train/test (70/30 split)
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
)

print(f"\nTrain size: {len(X_train)}")
print(f"Test size: {len(X_test)}")

# Save the data files
output_dir = 'public_data'
os.makedirs(output_dir, exist_ok=True)

# Save training data (space-separated, no header)
np.savetxt(os.path.join(output_dir, 'training_data'), X_train, fmt='%.6f')
np.savetxt(os.path.join(output_dir, 'training_label'), y_train, fmt='%d')

# Save testing data (space-separated, no header)
np.savetxt(os.path.join(output_dir, 'testing_data'), X_test, fmt='%.6f')
np.savetxt(os.path.join(output_dir, 'testing_label'), y_test, fmt='%d')

# Save label mapping for reference
with open(os.path.join(output_dir, 'label_mapping.txt'), 'w') as f:
    for idx, label in enumerate(le.classes_):
        f.write(f"{idx}: {label}\n")

print(f"\n✓ Data files created in '{output_dir}/' directory")
print("Files created:")
print("  - training_data")
print("  - training_label")
print("  - testing_data")
print("  - testing_label")
print("  - label_mapping.txt")
