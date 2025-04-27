import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score

def detect_fraud(df):
    # Check for required columns
    required_columns = ['amount', 'location', 'time']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"CSV must contain '{col}' column for K-Means detection.")

    # Encode categorical columns
    encoder = LabelEncoder()
    df['location_encoded'] = encoder.fit_transform(df['location'].astype(str))
    df['time_encoded'] = encoder.fit_transform(df['time'].astype(str))

    # Prepare features
    X = df[['amount', 'location_encoded', 'time_encoded']]

    # Apply KMeans
    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans.fit(X)
    centers = kmeans.cluster_centers_
    labels = kmeans.predict(X)

    # Compute distance from each point to its cluster center
    distances = np.linalg.norm(X - centers[labels], axis=1)

    # Set threshold as the 90th percentile distance (top 10% = fraud)
    threshold = np.percentile(distances, 90)
    df['Fraud'] = np.where(distances > threshold, 'Fraud', 'Non-Fraud')

    # Prepare metrics
    y_true = np.where(df['Fraud'] == 'Fraud', 1, 0)
    y_pred = np.where(distances > threshold, 1, 0)

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
        'k': 3,
        'cluster_centers': centers.tolist(),
        'distance_threshold': float(threshold),
        'total_transactions': total,
        'fraudulent_transactions': fraud_count
    }
}

    
