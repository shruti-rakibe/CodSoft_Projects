Credit Card Fraud Detection using Machine Learning

Project Overview:
This project builds a Machine Learning model to detect fraudulent credit card transactions. Fraud detection is an important real-world application of machine learning used by banks and financial institutions to prevent financial loss.
The dataset contains transaction details, and the model predicts whether a transaction is fraudulent or legitimate.
This project helps in understanding:

Dataset:
The dataset contains anonymized transaction data.
Features:
Time – Time elapsed between transactions
V1 to V28 – Numerical features obtained using PCA transformation
Amount – Transaction amount
Class – Target variable
0 → Normal Transaction
1 → Fraudulent Transaction
The dataset is highly imbalanced, meaning fraudulent transactions are very rare compared to normal transactions.
Dataset is too large to upload. You can download it from Kaggle:https:// https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

How to Run the Project:
Step 1: Install Required Libraries
pip install pandas numpy matplotlib seaborn scikit-learn

Step 2: Place Dataset
Make sure the file creditcard.csv is in the project folder.

Step 3: Run Python Script
python fraud_detection.py

Results:
Identifies fraud transactions.

Technologies Used:
Python 
Pandas
NumPy
Matplotlib
Seaborn
Scikit-learn
