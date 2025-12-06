"""
Prepare data WITH text features (vendor, description) for Codabench.
This matches the updated model.py that uses TF-IDF.
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
print(f"Columns: {list(df.columns)}")

# Select features that match the model
# Model expects: vendor, description, amount, payment_method, city, state
feature_columns = ['vendor', 'description', 'amount', 'payment_method', 'city', 'state']
target_column = 'category'

print(f"\nFeatures: {feature_columns}")

# Prepare X and y
X = df[feature_columns]
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

# Save the data files as CSV (not space-separated, because we have text)
output_dir = 'public_data'
os.makedirs(output_dir, exist_ok=True)

# Save training data (CSV with headers)
X_train.to_csv(os.path.join(output_dir, 'training_data.csv'), index=False)
np.savetxt(os.path.join(output_dir, 'training_label'), y_train, fmt='%d')

# Save testing data (CSV with headers)
X_test.to_csv(os.path.join(output_dir, 'testing_data.csv'), index=False)
np.savetxt(os.path.join(output_dir, 'testing_label'), y_test, fmt='%d')

# Save label mapping
with open(os.path.join(output_dir, 'label_mapping.txt'), 'w') as f:
    for idx, label in enumerate(le.classes_):
        f.write(f"{idx}: {label}\n")

# Save feature info
with open(os.path.join(output_dir, 'feature_info.txt'), 'w') as f:
    f.write("Features (6 total):\n")
    f.write("  - vendor (text): Merchant/vendor name\n")
    f.write("  - description (text): Transaction description\n")
    f.write("  - amount (numeric): Transaction amount\n")
    f.write("  - payment_method (categorical): Payment method\n")
    f.write("  - city (categorical): City\n")
    f.write("  - state (categorical): State\n")

print(f"\nData files created in '{output_dir}/' directory")
print("Files created:")
print("  - training_data.csv (with headers)")
print("  - training_label")
print("  - testing_data.csv (with headers)")
print("  - testing_label")
print("  - label_mapping.txt")
print("  - feature_info.txt")
print("\nNow the data matches your teammate's updated model!")
