import yfinance as yf
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# ========== Hyperparams ==========
TICKER = "AAPL"
START = "2018-01-01"
END = "2024-01-01"
SEQUENCE_LENGTH = 60
EPOCHS = 100
BATCH_SIZE = 32
LR = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========== Step 1: Get Data ==========
df = yf.download(TICKER, start=START, end=END)

# ========== Step 2: Add Indicators ==========
def add_technical_indicators(df):
    # RSI
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # MACD
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2

    df.dropna(inplace=True)
    return df

df = add_technical_indicators(df)

# ========== Step 3: Feature Scaling ==========
features = ['Close', 'RSI', 'MACD']
scalers = {feature: MinMaxScaler() for feature in features}

for feature in features:
    df[feature] = scalers[feature].fit_transform(df[feature].values.reshape(-1, 1))

# ========== Step 4: Sequence Prep ==========
def create_sequences(df, seq_len, target_col='Close'):
    sequences = []
    targets = []
    data = df[features].values
    for i in range(seq_len, len(df)):
        seq = data[i - seq_len:i]
        target = data[i][0]  # predict scaled Close
        sequences.append(seq)
        targets.append(target)
    return torch.tensor(sequences, dtype=torch.float32), torch.tensor(targets, dtype=torch.float32)

X, y = create_sequences(df, SEQUENCE_LENGTH)

train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# ========== Step 5: Model ==========
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.linear(out[:, -1])
        return out.squeeze()

model = LSTMModel(input_size=len(features)).to(DEVICE)
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# ========== Step 6: Training ==========
model.train()
for epoch in range(EPOCHS):
    optimizer.zero_grad()
    outputs = model(X_train.to(DEVICE))
    loss = loss_fn(outputs, y_train.to(DEVICE))
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {loss.item():.4f}")

# ========== Step 7: Evaluation ==========
model.eval()
with torch.no_grad():
    preds = model(X_test.to(DEVICE)).cpu().numpy()
    y_true = y_test.numpy()

# Inverse transform
preds_actual = scalers['Close'].inverse_transform(preds.reshape(-1, 1)).flatten()
y_actual = scalers['Close'].inverse_transform(y_true.reshape(-1, 1)).flatten()

# ========== Step 8: Plot ==========
plt.figure(figsize=(14, 6))
plt.plot(y_actual, label='Actual')
plt.plot(preds_actual, label='Predicted')
plt.title(f'{TICKER} Stock Price Prediction')
plt.xlabel('Days')
plt.ylabel('Price')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()