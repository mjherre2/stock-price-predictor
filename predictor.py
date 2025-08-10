# predictor.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

from utils import fetch_last_six_months_data_cached

# --- Config ---
TICKER = "AAPL"
SEQ_LEN = 60
EPOCHS = 10
BATCH_SIZE = 32

# --- Step 1: Load Data ---
df = fetch_last_six_months_data_cached(TICKER)

# --- Step 2: Create Label (1 if next day up, 0 if down) ---
df['Future'] = df['Close'].shift(-1)
df['Target'] = (df['Future'] > df['Close']).astype(int)
df.dropna(inplace=True)

# --- Step 3: Scale & Sequence ---
scaler = MinMaxScaler()
df['Close_scaled'] = scaler.fit_transform(df[['Close']])

def create_sequences(df, seq_len):
    X, y = [], []
    for i in range(len(df) - seq_len):
        X.append(df['Close_scaled'].values[i:i+seq_len])
        y.append(df['Target'].values[i+seq_len])
    return np.array(X), np.array(y)

X, y = create_sequences(df, SEQ_LEN)
X = X.reshape((X.shape[0], X.shape[1], 1))

# --- Step 4: Train/Test Split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# --- Step 5: Build Model ---
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(SEQ_LEN, 1)),
    Dropout(0.2),
    LSTM(50),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# --- Step 6: Train ---
model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(X_test, y_test))

# --- Step 7: Evaluate ---
loss, acc = model.evaluate(X_test, y_test)
print(f"\n✅ Test Accuracy: {acc:.2f}")

# --- Step 8: Plot Predictions ---
preds = model.predict(X_test).flatten()
pred_labels = (preds > 0.5).astype(int)

plt.figure(figsize=(10, 4))
plt.plot(pred_labels[:100], label="Predicted")
plt.plot(y_test[:100], label="Actual", alpha=0.7)
plt.legend()
plt.title(f"{TICKER} Trend Prediction (0=Down, 1=Up)")
plt.xlabel("Time Step")
plt.ylabel("Trend")
plt.tight_layout()
plt.show()

model.save("model/stock_lstm.h5")
