def predict_unemployment_lstm(file_name, lookback=3):

    # =============================
    # Imports
    # =============================
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import accuracy_score
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping

    # =============================
    # 1. Load Data
    # =============================
    df = pd.read_csv(file_name)
    df['date'] = pd.to_datetime(df['date'].str.strip(), format='%b %d, %Y')

    # Reverse because most recent is on top
    df = df.iloc[::-1].reset_index(drop=True)

    # =============================
    # 2. Create Target (0 or 1)
    # =============================
    df['is_positive'] = np.where(
        df['actual'] < df['forecast'], 1,
        np.where(df['actual'] > df['forecast'], 0, np.nan)
    )

    # Create surprise column
    df['surprise'] = df['actual'] - df['forecast']

    # =============================
    # 3. Create Lag Features
    # =============================
    for i in range(1, lookback + 1):
        df[f'actual_lag_{i}'] = df['actual'].shift(i)
        df[f'surprise_lag_{i}'] = df['surprise'].shift(i)

    # Drop rows with missing values (except last row which we predict)
    train_df = df.dropna().copy()

    # =============================
    # 4. Feature Selection
    # =============================
    feature_cols = ['forecast', 'previous'] + \
                   [f'actual_lag_{i}' for i in range(1, lookback + 1)] + \
                   [f'surprise_lag_{i}' for i in range(1, lookback + 1)]

    X = train_df[feature_cols].values
    y = train_df['is_positive'].values

    # =============================
    # 5. Scale Features
    # =============================
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # Reshape for LSTM
    X_scaled = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))

    # =============================
    # 6. Train-Test Split (80/20)
    # =============================
    split_index = int(len(X_scaled) * 0.90)

    X_train, X_test = X_scaled[:split_index], X_scaled[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    # =============================
    # 7. Build LSTM Model
    # =============================
    model = Sequential()
    model.add(LSTM(64, input_shape=(1, X_scaled.shape[2])))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )

    early_stop = EarlyStopping(patience=7, restore_best_weights=True)

    model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=8,
        verbose=0,
        validation_data=(X_test, y_test),
        callbacks=[early_stop]
    )

    # =============================
    # 8. Evaluate
    # =============================
    y_pred_prob = model.predict(X_test, verbose=0)
    y_pred = (y_pred_prob > 0.5).astype(int).flatten()

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

    # =============================
    # 9. Max Consecutive Failures
    # =============================
    failures = (y_pred != y_test).astype(int)

    max_fail = 0
    current = 0

    for f in failures:
        if f == 1:
            current += 1
            max_fail = max(max_fail, current)
        else:
            current = 0

    print(f"Max Consecutive Failures: {max_fail}")

    # =============================
    # 10. Predict Most Recent Row
    # =============================

    # Latest row (most recent release with missing actual)
    latest = df.iloc[-1:].copy()

    # Create lag features for latest row manually
    for i in range(1, lookback + 1):
        latest[f'actual_lag_{i}'] = df['actual'].iloc[-(i+1)]
        latest[f'surprise_lag_{i}'] = df['surprise'].iloc[-(i+1)]

    latest_features = latest[feature_cols].values
    latest_scaled = scaler.transform(latest_features)
    latest_scaled = latest_scaled.reshape((1, 1, latest_scaled.shape[1]))

    latest_pred_prob = model.predict(latest_scaled, verbose=0)
    latest_pred = int(latest_pred_prob.flatten()[0] > 0.5)

    print(f"Predicted Latest is_positive: {latest_pred}")

    return latest_pred




#here call the function predict_unemployment_lstm(file_name, lookback=3)

print(predict_unemployment_lstm('usunemployementdatalong.csv', lookback=3))