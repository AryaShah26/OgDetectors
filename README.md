Fraud Detection Using Machine Learning
This project aims to detect fraudulent transactions using machine learning models. It leverages both unsupervised learning for anomaly detection using Random Cut Forest (RCF) and supervised learning for classification using XGBoost. The goal is to accurately identify fraudulent transactions from a highly imbalanced dataset of credit card transactions.

Table of Contents
Project Overview
Features
Data
Requirements
Installation
Usage
Model Training
Evaluation
Future Enhancements
License
Project Overview
The project focuses on using machine learning techniques to detect fraudulent credit card transactions. With only a small percentage of fraudulent transactions in the dataset, the challenge lies in handling imbalanced data and building models that can generalize well.

We use two different approaches:

Unsupervised Learning: Detects anomalies in transaction data using Random Cut Forest (RCF).
Supervised Learning: Classifies transactions as fraudulent or legitimate using XGBoost.
The dataset contains 284,807 transactions, of which only 492 are fraudulent, making the task of fraud detection challenging but essential for building reliable financial systems.

Features
Anomaly Detection: Detects outliers in transactions without labeled data using RCF.
Fraud Classification: Uses XGBoost to classify transactions as fraudulent or legitimate.
Imbalanced Data Handling: Employs techniques such as upsampling and weight balancing to handle imbalanced data.
Data
The dataset used for this project is a public, anonymized credit card transactions dataset containing transactions from European cardholders. The dataset has the following key attributes:

Time: Seconds elapsed between any transaction and the first transaction.
Amount: Transaction amount.
PCA Components: 28 features that have been transformed using Principal Component Analysis (PCA).
Class: The target variable where 1 indicates a fraudulent transaction and 0 indicates a legitimate transaction.
The dataset is highly imbalanced, with a very small number of fraudulent transactions compared to legitimate ones.

Requirements
Python 3.x with the following libraries:
pandas
numpy
scikit-learn
imbalanced-learn
xgboost
matplotlib
seaborn
You can install the required packages by running:

bash
Copy code
pip install -r requirements.txt
Installation
Clone the repository:

bash
Copy code
git clone https://github.com/your-repo/fraud-detection.git
cd fraud-detection
Set up the environment: Install the required dependencies:

bash
Copy code
pip install -r requirements.txt
Prepare the dataset: Download the dataset from Kaggle and place it in the data/ directory.

Usage
Data Preprocessing:

Load the dataset and perform basic preprocessing such as scaling and handling missing values (if any).
Handle imbalanced data using upsampling techniques or weighting schemes.
Train the Models:

Use the train_xgboost.py script to train the XGBoost classifier.
Use the train_rcf.py script to train the Random Cut Forest model.
Make Predictions: After training the models, you can use them to make predictions on new transactions:

python
Copy code
from model import predict_transaction
result = predict_transaction(transaction_data)
print(result)
Evaluate the Models: Evaluate the performance of the models on a test dataset using accuracy, precision, recall, and F1-score for the XGBoost model. For the RCF model, you can analyze the anomaly scores.

Model Training
Random Cut Forest (RCF):

The unsupervised RCF model is trained to detect anomalies in transaction data. It provides an anomaly score for each transaction, which can be used to flag suspicious transactions.
XGBoost:

A supervised learning model that is trained on labeled data to classify transactions as fraudulent or legitimate. The model is optimized using techniques like upsampling of the minority class and setting scale weights for imbalanced data.
Evaluation
XGBoost Model:

Accuracy
Precision
Recall
F1-Score
RCF Model:

Analyze the anomaly scores generated for each transaction and determine a suitable threshold to classify a transaction as fraudulent.
Confusion Matrix: Helps in understanding how well the model is classifying the fraudulent and legitimate transactions.

You can use the provided evaluate_model.py script to evaluate the performance of the models.

Future Enhancements
Feature Engineering: Explore more advanced features from raw transaction data to improve model performance.
Hyperparameter Tuning: Experiment with hyperparameters of the XGBoost model for better performance.
Threshold Optimization: Optimize the anomaly score threshold in RCF to reduce false positives and negatives.
Real-Time Integration: Deploy the models into a real-time system for fraud detection.
License
This project is licensed under the MIT License - see the LICENSE file for details.

This README provides clear instructions and insights into the project's structure, goals, and usage without AWS content.
