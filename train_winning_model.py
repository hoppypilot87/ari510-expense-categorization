"""
Train a high-performance model on the full feature set.
This will be the winning submission!
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
import joblib

print("Loading data...")
X_train = np.genfromtxt('ExpenseCategorization_Benchmark/public_data/training_data')
y_train = np.genfromtxt('ExpenseCategorization_Benchmark/public_data/training_label')
X_test = np.genfromtxt('ExpenseCategorization_Benchmark/public_data/testing_data')
y_test = np.genfromtxt('ExpenseCategorization_Benchmark/public_data/testing_label')

print(f"Training shape: {X_train.shape}")
print(f"Testing shape: {X_test.shape}")

# Scale features
print("\nScaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a strong model
print("\nTraining Gradient Boosting model...")
model = GradientBoostingClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.1,
    random_state=42
)
model.fit(X_train_scaled, y_train)

# Evaluate
print("\nEvaluating...")
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')

print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1:.4f}")

# Save model for submission
print("\nSaving model...")
joblib.dump(model, 'winning_model.pkl')
joblib.dump(scaler, 'winning_scaler.pkl')

print("\nModel saved! Ready to create submission.")
