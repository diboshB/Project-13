# predict_model.py

import joblib
import pandas as pd
import sqlite3
import numpy as np

# Loading the saved model
model = joblib.load('fraud_model.xgb')

# Connecting to the database and fetching the live dataset (post-5 million records)
conn = sqlite3.connect('/Users/diboshbaruah/Desktop/Database.db')
data = pd.read_sql_query('SELECT * FROM Fraud_Detection', conn)
conn.close()

# Pre-process data as done in the training phase
numeric_columns = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']
data[numeric_columns] = data[numeric_columns].apply(pd.to_numeric, errors='coerce')
data['step'] = pd.to_numeric(data['step'], errors='coerce')

# Dropping unnecessary columns
data.drop(columns=['nameOrig', 'nameDest'], inplace=True)

# Using One-hot encoding for 'type' column
data = pd.get_dummies(data, columns=['type'], drop_first=True)

# Converting other columns to integers
data['isFraud'] = data['isFraud'].astype(int)
data['isFlaggedFraud'] = data['isFlaggedFraud'].astype(int)

# Extracting the features
X_live = data.drop(columns=['isFraud', 'isFlaggedFraud'])

# Making predictions using the trained model
predictions = model.predict(X_live)

# Displaying predictions 
print("Predictions on live data:")
print(predictions)
