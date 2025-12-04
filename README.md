ARI 510 Final Project – Automated Expense Categorization

University of Michigan–Flint • Fall 2025

Overview

This project implements a complete machine learning pipeline for automatic expense categorization.
Given a set of numeric features about a transaction (amount, user profile, simulated metadata), our model predicts one of 12 spending categories, such as:
	•	Groceries
	•	DiningOut
	•	Utilities
	•	Transport
	•	Shopping
	•	Bills
	•	Travel
	•	Healthcare
	•	Education
	•	Entertainment
	•	PersonalCare
	•	Miscellaneous

The system includes:

✔ A synthetic dataset of 10,000 transactions
✔ Human annotation workflow for HW3
✔ Full modeling pipeline (EDA → Baselines → Tuning)
✔ A final tuned Logistic Regression model (best accuracy & inference time)
✔ A standalone model_predict.py script for real-time predictions
✔ A Gradio web interface + Codabench evaluation (Grant)

Project Architecture

ari510-expense-categorization/
│
├── data/
│   ├── raw/                     # Synthetic dataset (transactions_synthetic.csv)
│   ├── annotation/              # Annotation sample + filled HW3 labels
│   └── processed/               # (Deprecated in final pipeline—raw is used directly)
│
├── notebooks/
│   ├── 01_eda.ipynb             # Exploratory data analysis & dataset creation
│   ├── 02_baselines.ipynb       # Baseline models (LR, RF)
│   ├── 03_model_tuning.ipynb    # Hyperparameter tuning & final model export
│   └── 05_model_predict.ipynb   # (Optional) End-to-end demo for presentation
│
├── models/
│   ├── best_category_model.pkl  # Final tuned Logistic Regression model
│   └── category_scaler.pkl      # StandardScaler trained on training data
│
├── model_predict.py             # Standalone prediction script (used for demo)
├── LICENSE.txt
└── README.md                    # (This file)

 Dataset

Because the original Indian Personal Finance dataset did not contain real transaction descriptions, we generated a synthetic but realistic dataset with:
	•	10,000 individual transactions
	•	12 spending categories
	•	User metadata (income, occupation, dependents, city tier)
	•	Transaction metadata
	•	Amount distributions based on real-world spending patterns

A 50-row random subset was provided to annotators for HW3.

Modeling Pipeline

1. EDA (01_eda.ipynb)
	•	Reads synthetic transaction data
	•	Cleans, validates, and inspects category distribution
	•	Prepares data for modeling (feature matrix + labels)

2. Baseline Models (02_baselines.ipynb)
	•	Logistic Regression (multinomial)
	•	Random Forest Classifier
	•	Evaluation using accuracy, precision, recall, F1

3. Model Tuning (03_model_tuning.ipynb)
	•	Hyperparameter search
	•	Trains + scores advanced models
	•	Selects Logistic Regression as final based on:
	•	Highest accuracy
	•	Fastest inference
	•	Easiest deployment

4. Final Model Export

Saves two artifacts:
	•	best_category_model.pkl
	•	category_scaler.pkl

5. Final Prediction Script (model_predict.py)

Provides:
	•	Single prediction function
	•	Random-sample demo
	•	Preprocessing consistent with training
	•	Clean, readable CLI output

Example Run:

python model_predict.py

=== Demo predictions on random subset ===
Row 8338: true = Shopping     | predicted = DiningOut
Row 5516: true = Utilities    | predicted = Utilities

Gradio
 run:
 pip install gradio
 run app.py file
 Open in browser
 
HW3 (Annotation)

Included:
	•	A 50-row annotation sample
	•	Full annotation guidelines
	•	Revised dropdown categories
	•	Completed labeling file
	•	Code for comparing human labels vs. model predictions

Team
	•	Philip Haapala — End-to-end pipeline, synthetic dataset, notebooks 1–3, inference script, documentation
	•	Grant Finn — Deployment (Codabench, Gradio), testing, review

License

MIT License — see LICENSE.txt

