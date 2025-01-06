# Project-13
Capstone_Project - Classification - Fraud_Detection

1. Introduction
Financial fraud detection is a critical problem in banking and payment systems,
where the primary objective is to identify fraudulent transactions in real-time
to minimize financial losses and protect users. Fraudulent transactions can
lead to significant financial harm, so detecting them early is crucial for both
financial institutions and their customers.
For this project, we used a dataset consisting of 6 million transaction records
from a bank's transaction system. Each record includes several features such
as the transaction type, transaction amount, and the balance before and after
the transaction. The target variable, isFraud, indicates whether a transaction is
fraudulent (1) or non-fraudulent (0). The goal of this project was to build a
machine learning model that could predict fraud with high accuracy, using
XGBoost, a gradient boosting machine learning algorithm known for its
performance and efficiency in classification problems.
2. Data Collection
The data for this project was collected from a financial transaction system in
the form of a relational database, specifically SQLite format, containing
records of transactions. The dataset provided includes several key features
that help describe the transaction details:
• Step: The time step of the transaction (each transaction occurs on a
different step, which essentially represents a day).
• Type: The type of the transaction (e.g., PAYMENT, TRANSFER, CASH-
IN, CASH-OUT).
• Amount: The amount of money involved in the transaction.
• OldBalanceOrg: The balance of the origin account before the
transaction occurred.
• NewBalanceOrg: The balance of the origin account after the
transaction.
Additionally, the dataset includes fraud labels indicating whether a transaction
was fraudulent or not (the isFraud column). Once collected, the dataset was
loaded into a Pandas DataFrame for further analysis and processing.
3. Data Preprocessing
Before feeding the data into the machine learning model, several
preprocessing steps were necessary to clean and transform the data:
3.1 Handling Missing Values
The first step was to check if there were any missing values in the dataset.
For the dataset used in this project, there were no missing values, so no
imputation was required. This ensured that we could directly work with the
full dataset.
3.2 Feature Engineering
The dataset contains both categorical and numerical features. We performed
the following transformations:
• Categorical Variables: The Type feature was one-hot encoded into
multiple binary columns, where each type of transaction (e.g.,
PAYMENT, TRANSFER, etc.) becomes a separate feature.
• Numerical Features: Features such as Amount, OldBalanceOrg, and
NewBalanceOrg were already in numerical form, so no transformation was
required for these variables.
Additionally, we created new features such as the transaction balance change
by subtracting OldBalanceOrg from NewBalanceOrg. This additional feature
provided more insight into the financial activity surrounding each transaction.
3.3 Feature Scaling
Although XGBoost can handle raw, unscaled data efficiently, we performed
standardization on features like Amount, OldBalanceOrg, and NewBalanceOrg to
ensure that no feature would disproportionately influence the model due to its
scale.
4. Data Collection & Feature Selection
4.1 Feature Inspection & Selection
After exploring the dataset, the next step was to identify and select the most
relevant features for the model. The dataset initially contained 14 features,
including the target variable isFraud. Key steps in the feature selection process
included:
• Identifying Redundant Features: After inspecting the dataset, we
noted that some features, such as OldBalanceOrg and NewBalanceOrg, were
closely related, with their difference representing the amount involved
in the transaction.
• Removing Irrelevant Features: Features like nameOrig and nameDest
(representing the account names) were found to be non-contributory to
fraud detection and were removed.
• Feature Importance: Using XGBoost's feature importance techniques,
we identified key features like Amount, OldBalanceOrg, NewBalanceOrg, and
Type as the most influential predictors.
4.2 Feature Engineering
In addition to the original features, new features were engineered to
potentially improve the model performance:
• Balance Change: A feature capturing the difference between
NewBalanceOrg and OldBalanceOrg, representing the financial impact of the
transaction.
• Transaction Type Encodings: One-hot encoding was applied to the
Type feature, transforming it into binary columns for each transaction
type (e.g., PAYMENT, TRANSFER, CASH-IN, CASH-OUT).
4.3 Handling Class Imbalance
Given the highly imbalanced dataset (with fraudulent transactions being much
fewer than non-fraudulent ones), we used SMOTE (Synthetic Minority Over-
sampling Technique) to generate synthetic samples for the minority class
(fraudulent transactions). This technique helped to improve the model’s ability
to detect fraud.
5. Model Building
For the predictive model, we chose XGBoost, a powerful gradient boosting
algorithm known for its performance and efficiency in classification problems.
The model was trained on the processed data and evaluated using cross-
validation.
5.1 Model Evaluation Metrics
We used the following evaluation metrics for this binary classification problem:
• Accuracy
• Precision
• Recall
• F1-Score
• AUC-ROC Curve
The AUC (Area Under Curve) was particularly important for assessing the
model’s ability to distinguish between fraudulent and non-fraudulent
transactions.
6. Results & Discussion
6.1 Model Performance
The XGBoost model performed exceptionally well across all metrics:
• Accuracy: 1.00 (Perfect accuracy, correctly classifying all non-
fraudulent transactions and most fraudulent transactions)
• Precision:
o Class 0 (Non-fraudulent transactions): 1.00 (Perfect
precision for class 0, meaning almost all predicted non-fraudulent
transactions were correct)
o Class 1 (Fraudulent transactions): 0.97 (97% of predicted
fraudulent transactions were indeed fraudulent)
• Recall:
o Class 0 (Non-fraudulent transactions): 1.00 (Perfect recall for
class 0, meaning all non-fraudulent transactions were correctly
identified)
o Class 1 (Fraudulent transactions): 0.68 (68% of all fraudulent
transactions were correctly identified)
• F1-Score: 0.80 (The overall F1-Score, a weighted average of both class
F1 scores, reflects strong performance but highlights a slight trade-off
due to the lower recall for class 1)
• ROC AUC: 0.9996 (Indicates near-perfect ability to distinguish between
fraudulent and non-fraudulent transactions)
Confusion Matrix:
• True negatives (TN): 999,436 (Non-fraudulent transactions correctly
predicted as non-fraudulent)
• False positives (FP): 10 (Non-fraudulent transactions incorrectly
predicted as fraudulent)
• False negatives (FN): 179 (Fraudulent transactions incorrectly predicted
as non-fraudulent)
• True positives (TP): 375 (Fraudulent transactions correctly predicted as
fraudulent)
Precision-Recall AUC: 0.8305 (Shows strong performance for both precision
and recall, though with some room for improvement on recall for class 1).
6.2 Feature Importance
The most important features contributing to the model's decision-making
process are:
• newbalanceDest: 0.1754
• newbalanceOrig: 0.1711
• type_CASH_OUT: 0.1477
• type_CASH_IN: 0.1252
• oldbalanceOrg: 0.1044
• type_PAYMENT: 0.0813
• oldbalanceDest: 0.0566
• amount: 0.0551
• step: 0.0384
• type_TRANSFER: 0.0361
• type_DEBIT: 0.0086
These features were found to be the most influential in predicting fraudulent
transactions, with newbalanceDest, newbalanceOrig, and type_CASH_OUT emerging
as the top contributors.
7. Conclusion
The project demonstrates the effectiveness of XGBoost for predicting
fraudulent financial transactions. By preprocessing the data, engineering new
features, and handling class imbalance, the model achieved excellent
performance, with near-perfect accuracy and strong precision for non-
fraudulent transactions.
While the model showed high performance overall, there is still room for
improvement, especially in increasing the recall for fraudulent transactions.
Future work could explore more advanced techniques for handling rare fraud
patterns, anomaly detection, and leveraging ensemble methods to further
enhance the model's predictive accuracy.
Deploying such a model in real-world banking systems would significantly
reduce fraud-related losses and help protect customers from financial crimes.
