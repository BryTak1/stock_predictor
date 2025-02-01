# app.py
from flask import Flask, render_template, request, jsonify
import yfinance as yf
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta

app = Flask(__name__)

# LSTM Model Definition
class StockPredictor(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, output_size=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.linear(out[:, -1, :])
        return out

model = StockPredictor()
model.load_state_dict(torch.load('stock_predictor.pth', map_location=torch.device('cpu')))
model.eval()

def get_historical_data(ticker):
    end = datetime.now()
    start = end - timedelta(days=5*365)  # Changed from 1 year to 5 years
    data = yf.download(ticker, start=start, end=end + timedelta(days=1))
    return data[['Close']]
def prepare_input(data):
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled_data = scaler.fit_transform(data)
    
    sequence = []
    for i in range(len(scaled_data)-60):
        sequence.append(scaled_data[i:i+60])
        
    sequence = np.array(sequence)
    return torch.FloatTensor(sequence[-1].reshape(1, 60, 1)), scaler

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    ticker = request.form['ticker'].upper()
    
    try:
        data = get_historical_data(ticker)
        if len(data) < 60:
            return jsonify({'error': f'Need at least 60 days of data for {ticker}'})
        
        input_tensor, scaler = prepare_input(data.values)
        with torch.no_grad():
            prediction = model(input_tensor)
        
        # Get prediction date (next trading day)
        last_date = data.index[-1].date()
        prediction_date = last_date + timedelta(days=1)

        
        # Convert to weekend-aware date
        while prediction_date.weekday() >= 5:  # 5 = Saturday, 6 = Sunday
            prediction_date += timedelta(days=1)
        predicted_price = float(scaler.inverse_transform(prediction.numpy())[0][0])
        last_price = float(data['Close'].iloc[-1])
        percent_change = float(((predicted_price - last_price) / last_price) * 100)
        
        dates = data.index.tz_localize(None).strftime('%Y-%m-%d').tolist()
        prices = [float(price) for price in data['Close'].values]
        
        return jsonify({
            'ticker': ticker,
            'prediction_date': prediction_date.strftime('%Y-%m-%d'),
            'prediction': round(predicted_price, 2),
            'change': round(percent_change, 2),
            'history': {
                'dates': dates,
                'prices': prices
            }
        })
    
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)