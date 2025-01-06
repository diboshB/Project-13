# train_model.py

import sqlite3
import pandas as pd
import xgboost as xgb
from sklearn.metrics import classification_report, roc_auc_score, f1_score, confusion_matrix
import joblib
import matplotlib.pyplot as plt

# Connecting to the database
conn = sqlite3.connect('/Users/diboshbaruah/Desktop/Database.db')
data = pd.read_sql_query('SELECT * FROM Fraud_Detection', conn)
conn.close()

# Data Pre-processing
numeric_columns = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']
data[numeric_columns] = data[numeric_columns].apply(pd.to_numeric, errors='coerce')
data['step'] = pd.to_numeric(data['step'], errors='coerce')

# Dropping unique identifier columns
data.drop(columns=['nameOrig', 'nameDest'], inplace=True)

# Using One-hot encoding for 'type' column
data = pd.get_dummies(data, columns=['type'], drop_first=True)

# Converting other columns to integer type
data['isFraud'] = data['isFraud'].astype(int)
data['isFlaggedFraud'] = data['isFlaggedFraud'].astype(int)

# Splitting data into train, validation, and live sets
train_data = data.iloc[:4000000]
validation_data = data.iloc[4000000:5000000]
live_data = data.iloc[5000000:]

X_train = train_data.drop(columns=['isFraud', 'isFlaggedFraud'])
y_train = train_data['isFraud']

X_validation = validation_data.drop(columns=['isFraud', 'isFlaggedFraud'])
y_validation = validation_data['isFraud']

# Training XGBoost model
model = xgb.XGBClassifier(
    eval_metric='logloss', 
    n_estimators=100, 
    max_depth=6, 
    learning_rate=0.1, 
    subsample=0.8, 
    colsample_bytree=0.8
) 
model.fit(X_train, y_train)

# Evaluating Model on Validation Data
y_pred_eval = model.predict(X_validation)
y_pred_prob = model.predict_proba(X_validation)[:, 1]

# Classification Report and Metrics
print("Classification Report on Validation Dataset:")
print(classification_report(y_validation, y_pred_eval))

print("ROC AUC Score (default threshold):", roc_auc_score(y_validation, y_pred_prob))
print("F1-score:", f1_score(y_validation, y_pred_eval))
print("Confusion Matrix:")
print(confusion_matrix(y_validation, y_pred_eval))

# Saving the trained model to disk
joblib.dump(model, 'fraud_model.xgb')
print("Trained model is saved!!")
