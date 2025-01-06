# app.py

from flask import Flask, request, jsonify
import joblib
import pandas as pd

# Initializing the Flask application
app = Flask(__name__)

# Loading the trained model
model = joblib.load('fraud_model.xgb')

# List of expected one-hot encoded columns for 'type'
expected_columns = [
    'step', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest',
    'type_CASH_IN', 'type_CASH_OUT', 'type_DEBIT', 'type_PAYMENT', 'type_TRANSFER'
]

# Pre-process the input data
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Getting data from POST request
        data = request.get_json()

        # Converting the data into a DataFrame
        df = pd.DataFrame(data)

        # Dropping 'nameOrig' and 'nameDest' columns
        if 'nameOrig' in df.columns:
            df.drop(columns=['nameOrig'], inplace=True)
        if 'nameDest' in df.columns:
            df.drop(columns=['nameDest'], inplace=True)

        # Ensuring the numeric columns are processed correctly
        numeric_columns = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']
        df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')

        # Converting 'step' to numeric
        df['step'] = pd.to_numeric(df['step'], errors='coerce')

        # Using One-hot encoding for 'type' column 
        df = pd.get_dummies(df, columns=['type'], drop_first=True)

        # Adding missing one-hot encoded columns (with 0 values) 
        for col in expected_columns:
            if col not in df.columns:
                df[col] = 0

        # Ensuring that the columns are in the same order as the model was trained
        df = df[expected_columns]

        # Now, we remove the target columns from the input before passing to the model
        # Ensure 'isFraud' and 'isFlaggedFraud' are not present in the input data.
        if 'isFraud' in df.columns:
            df.drop(columns=['isFraud'], inplace=True)
        if 'isFlaggedFraud' in df.columns:
            df.drop(columns=['isFlaggedFraud'], inplace=True)

        # Making prediction using the cleaned and preprocessed input
        prediction = model.predict(df)

        # Returning the prediction as JSON
        return jsonify({'prediction': prediction.tolist()})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
