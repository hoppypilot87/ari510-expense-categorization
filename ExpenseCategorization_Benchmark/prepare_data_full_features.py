"""
Prepare training and testing data with engineered features.
This creates a richer feature set for better model performance.
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
print(f"Categories: {df['category'].unique()}")

# Feature engineering
print("\nEngineering features...")

# 1. Amount features
df['log_amount'] = np.log1p(df['amount'])
df['amount_squared'] = df['amount'] ** 2

# 2. Categorical encodings
# Payment method
payment_encoder = LabelEncoder()
df['payment_method_encoded'] = payment_encoder.fit_transform(df['payment_method'])

# State
state_encoder = LabelEncoder()
df['state_encoded'] = state_encoder.fit_transform(df['state'])

# 3. Vendor features (simple encoding)
vendor_encoder = LabelEncoder()
df['vendor_encoded'] = vendor_encoder.fit_transform(df['vendor'])

# 4. Description length
df['description_length'] = df['description'].str.len()

# 5. Time features (extract from date)
df['date'] = pd.to_datetime(df['date'])
df['day_of_week'] = df['date'].dt.dayofweek
df['month'] = df['date'].dt.month
df['day_of_month'] = df['date'].dt.day

# Select features for model
feature_columns = [
    'amount',
    'log_amount',
    'amount_squared',
    'payment_method_encoded',
    'state_encoded',
    'vendor_encoded',
    'description_length',
    'day_of_week',
    'month',
    'day_of_month'
]

target_column = 'category'

print(f"\nFeatures ({len(feature_columns)}): {feature_columns}")

# Prepare X and y
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
print(f"Feature shape: {X_train.shape}")

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

# Save feature names
with open(os.path.join(output_dir, 'feature_names.txt'), 'w') as f:
    for i, feat in enumerate(feature_columns):
        f.write(f"{i}: {feat}\n")

print(f"\nData files created in '{output_dir}/' directory")
print("Files created:")
print("  - training_data")
print("  - training_label")
print("  - testing_data")
print("  - testing_label")
print("  - label_mapping.txt")
print("  - feature_names.txt")
print(f"\nNow participants can achieve much higher accuracy with {len(feature_columns)} features!")
