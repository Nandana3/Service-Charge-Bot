# train_model.py

import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib
import pickle

# Load the dataset
df = pd.read_csv('kuwait_expat_labor_data.csv')

# --- Feature Selection and Engineering ---
features = ['reason_for_leaving', 'industry', 'job_category', 'monthly_salary_kwd']
target = 'filed_dispute'

X = df[features]
y = df[target]

# --- Data Preprocessing (One-Hot Encoding) ---
X_encoded = pd.get_dummies(X, columns=['reason_for_leaving', 'industry', 'job_category'], drop_first=True)

# --- Save the column list ---
# This is crucial for ensuring the live app uses the exact same columns
with open('model_columns.pkl', 'wb') as f:
    pickle.dump(X_encoded.columns.tolist(), f)

# --- Model Training (on the entire dataset) ---
# Now we train on all data since the goal is to have the most knowledgeable model for the app
model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_encoded, y)

# --- Save the trained model ---
joblib.dump(model, 'dispute_model.joblib')

print("Model trained and saved as 'dispute_model.joblib'")
print("Model columns saved as 'model_columns.pkl'")