import pandas as pd
import numpy as np

def detect_fraud(df):
    # Check if 'amount' column exists
    if 'amount' not in df.columns:
        raise ValueError("CSV must contain 'amount' column for Z-Score detection.")

    # Calculate mean and std
    mean = df['amount'].mean()
    std = df['amount'].std()

    # Compute Z-scores
    df['z_score'] = (df['amount'] - mean) / std

    # Set threshold
    threshold = 3

    # Label fraud
    df['Fraud'] = df['z_score'].apply(lambda x: 'Fraud' if abs(x) > threshold else 'Non-Fraud')

    # Count stats
    fraud_count = (df['Fraud'] == 'Fraud').sum()
    total = len(df)
    anomaly_percent = round(100 * fraud_count / total, 2)

    # Mock accuracy & precision for demo
    summary = {
        'anomaly_percent': anomaly_percent,
        'accuracy': round(95 + np.random.rand() * 2, 2),      # mock: 95–97%
        'precision': round(85 + np.random.rand() * 5, 2)      # mock: 85–90%
    }
    print("Z-Score detected:", summary)


    return df.drop(columns='z_score'), summary
