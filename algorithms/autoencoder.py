import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def detect_fraud(df):
    # Check for required columns
    required_columns = ['amount', 'location', 'time']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"CSV must contain '{col}' column for Autoencoder detection.")

    # Encode categorical columns
    encoder = LabelEncoder()
    df['location_encoded'] = encoder.fit_transform(df['location'].astype(str))
    df['time_encoded'] = encoder.fit_transform(df['time'].astype(str))

    # Prepare features
    X = df[['amount', 'location_encoded', 'time_encoded']]

    # Scale the data
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # Build Autoencoder model
    model = Sequential([
        Dense(4, activation='relu', input_shape=(X_scaled.shape[1],)),
        Dense(2, activation='relu'),
        Dense(4, activation='relu'),
        Dense(X_scaled.shape[1], activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='mse')
    
    # Train the model
    model.fit(X_scaled, X_scaled, epochs=30, batch_size=32, verbose=0)

    # Get reconstruction error
    X_pred = model.predict(X_scaled, verbose=0)
    reconstruction_error = np.mean(np.power(X_scaled - X_pred, 2), axis=1)

    # Set threshold (90th percentile error)
    threshold = np.percentile(reconstruction_error, 90)

    # Label frauds
    df['Fraud'] = np.where(reconstruction_error > threshold, 'Fraud', 'Non-Fraud')

    # Prepare labels for metrics
    y_true = np.where(df['Fraud'] == 'Fraud', 1, 0)
    y_pred = np.where(reconstruction_error > threshold, 1, 0)

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
        'model': '3 Layer Autoencoder',
        'loss_threshold_percentile': 90,
        'total_transactions': total,
        'fraudulent_transactions': fraud_count
    }
}

