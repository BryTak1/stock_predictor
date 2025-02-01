# train_model.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
import random

# Configuration
SEQUENCE_LENGTH = 60
EPOCHS = 100
BATCH_SIZE = 64
HIDDEN_SIZE = 64
LEARNING_RATE = 0.001
TICKERS = ['AAPL', 'GOOG', 'MSFT', 'AMZN', 'META', 'TSLA', 'NVDA', 'PYPL', 'NFLX', 'INTC', 'SOL']

# LSTM Model Definition
class StockPredictor(nn.Module):
    def __init__(self, input_size=1, hidden_size=HIDDEN_SIZE, output_size=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.linear(out[:, -1, :])
        return out

# Dataset Class
class StockDataset(Dataset):
    def __init__(self, sequences, targets):
        self.sequences = sequences
        self.targets = targets
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]

def download_stock_data(tickers, years=5):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365*years)
    
    all_data = []
    for ticker in tickers:
        data = yf.download(ticker, start=start_date, end=end_date)
        if not data.empty:
            closes = data['Close'].values.reshape(-1, 1)
            all_data.append(closes)
    return np.vstack(all_data)

def create_sequences(data, sequence_length):
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled_data = scaler.fit_transform(data)
    
    sequences = []
    targets = []
    for i in range(len(scaled_data) - sequence_length):
        seq = scaled_data[i:i+sequence_length]
        target = scaled_data[i+sequence_length]
        sequences.append(seq)
        targets.append(target)
    
    return np.array(sequences), np.array(targets), scaler

def train():
    # Load and prepare data
    stock_data = download_stock_data(TICKERS)
    sequences, targets, scaler = create_sequences(stock_data, SEQUENCE_LENGTH)
    
    # Convert to PyTorch tensors
    sequences = torch.FloatTensor(sequences)
    targets = torch.FloatTensor(targets)
    
    # Create DataLoader
    dataset = StockDataset(sequences, targets)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Initialize model, loss function, and optimizer
    model = StockPredictor()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Training loop
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        
        for batch_seq, batch_target in dataloader:
            optimizer.zero_grad()
            
            outputs = model(batch_seq)
            loss = criterion(outputs, batch_target)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        print(f'Epoch [{epoch+1}/{EPOCHS}], Loss: {avg_loss:.6f}')
    
    # Save the trained model
    torch.save(model.state_dict(), 'stock_predictor.pth')
    print("Model saved to stock_predictor.pth")

if __name__ == '__main__':
    train()