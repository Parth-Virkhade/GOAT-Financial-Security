import pandas as pd
import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score

def detect_fraud(df):
    # Check if required columns exist
    required_columns = ['amount', 'location', 'time']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"CSV must contain '{col}' column for LOF detection.")

    # Encode categorical columns
    encoder = LabelEncoder()
    df['location_encoded'] = encoder.fit_transform(df['location'].astype(str))
    df['time_encoded'] = encoder.fit_transform(df['time'].astype(str))

    # Prepare features
    X = df[['amount', 'location_encoded', 'time_encoded']]

    # Fit LOF model
    lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
    preds = lof.fit_predict(X)

    # LOF predicts -1 for outliers (fraud), 1 for normal (non-fraud)
    df['Fraud'] = np.where(preds == -1, 'Fraud', 'Non-Fraud')

    # Prepare labels for metrics
    y_true = np.where(df['Fraud'] == 'Fraud', 1, 0)
    y_pred = np.where(preds == -1, 1, 0)

    # Calculate real accuracy and precision
    accuracy = round(accuracy_score(y_true, y_pred) * 100, 2)
    precision = round(precision_score(y_true, y_pred, zero_division=0) * 100, 2)

    fraud_count = (df['Fraud'] == 'Fraud').sum()
    total = len(df)
    anomaly_percent = round(100 * fraud_count / total, 2)

    return df.drop(columns=['location_encoded', 'time_encoded']), {
    'anomaly_percent': anomaly_percent,
    'accuracy': accuracy,
    'precision': precision,
    'details': {
        'n_neighbors': 20,
        'contamination': 0.1,
        'total_transactions': total,
        'fraudulent_transactions': fraud_count
    }
}

