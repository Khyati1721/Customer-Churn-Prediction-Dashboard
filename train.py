import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# Load Dataset
df = pd.read_csv("Telco-Customer-Churn.csv")

# Convert TotalCharges to numeric (some rows are empty strings)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Fill missing numeric values only
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

# Drop ID column
if 'customerID' in df.columns:
    df.drop('customerID', axis=1, inplace=True)


# Encode Categorical Variables
df = pd.get_dummies(df, drop_first=True)


# Features & Target
X = df.drop('Churn_Yes', axis=1)
y = df['Churn_Yes']

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# Train Model
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)


# Save Model & Columns
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(X.columns.tolist(), open("columns.pkl", "wb"))

print("✅ Model trained and saved successfully!")
