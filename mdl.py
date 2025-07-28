import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

device = torch.device('cpu')

# Fetch Data
ticker = input("Enter ticker symbol: ")
df = yf.download(ticker, '2020-01-01')

# Scale prices
scaler = StandardScaler()
df['Close_scaled'] = scaler.fit_transform(df[['Close']])

# Create sequences
seq_length = 30
data = []
for i in range(len(df) - seq_length):
    data.append(df['Close_scaled'].values[i:i + seq_length])

data = np.array(data)

# Split into train and test
train_size = int(0.8 * len(data))
X_train = torch.from_numpy(data[:train_size, :-1]).float().to(device)
Y_train = torch.from_numpy(data[:train_size, -1]).float().to(device)
X_test = torch.from_numpy(data[train_size:, :-1]).float().to(device)
Y_test = torch.from_numpy(data[train_size:, -1]).float().to(device)

# Reshape for LSTM: (batch_size, seq_length, input_dim)
X_train = X_train.unsqueeze(-1)
X_test = X_test.unsqueeze(-1)

# LSTM Model
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(2, x.size(0), 32).to(device)
        c0 = torch.zeros(2, x.size(0), 32).to(device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])  # last time step
        return out

# Initialize model
model = LSTMModel(input_dim=1, hidden_dim=32, num_layers=2, output_dim=1).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop
model.train()
num_epochs = 200
for epoch in range(num_epochs):
    y_train_pred = model(X_train)
    loss = criterion(y_train_pred.squeeze(), Y_train)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 25 == 0:
        print(f"Epoch {epoch:03d} | Loss: {loss.item():.4f}")

# Predict on test set
model.eval()
with torch.no_grad():
    y_test_pred = model(X_test).squeeze()

# Convert predictions and true values back to original price scale
y_test_pred_np = y_test_pred.cpu().numpy().reshape(-1, 1)
y_test_true_np = Y_test.cpu().numpy().reshape(-1, 1)

y_test_pred_rescaled = scaler.inverse_transform(y_test_pred_np)
y_test_true_rescaled = scaler.inverse_transform(y_test_true_np)

# RMSE using NumPy sqrt for compatibility
rmse = np.sqrt(mean_squared_error(y_test_true_rescaled, y_test_pred_rescaled))
print(f"\nTest RMSE: {rmse:.2f}")

# Index for plotting
index = df.iloc[-len(Y_test):].index

# Plot
fig = plt.figure(figsize=(14, 10))
gs = fig.add_gridspec(4, 1)

# Price prediction plot
ax1 = fig.add_subplot(gs[:3, 0])
ax1.plot(index, y_test_true_rescaled, color='blue', label='Actual Price')
ax1.plot(index, y_test_pred_rescaled, color='green', label='Predicted Price')
ax1.legend()
ax1.set_title(f'{ticker} Stock Price Prediction')
ax1.set_ylabel('Price')

# Residuals plot
ax2 = fig.add_subplot(gs[3, 0])
residuals = y_test_true_rescaled.flatten() - y_test_pred_rescaled.flatten()
ax2.plot(index, residuals, color='red', label='Residuals')
ax2.axhline(0, color='black', linestyle='--')
ax2.set_title('Prediction Residuals')
ax2.set_xlabel('Date')
ax2.set_ylabel('Residual')
ax2.legend()

plt.tight_layout()
plt.show()