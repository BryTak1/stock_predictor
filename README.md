<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
</head>
<body>
    <div class="container">
        <h1>📈 Stock Price Prediction Web Application</h1>
        <p>Welcome to the <strong>Stock Price Prediction Web Application</strong>! This project leverages a <strong>Long Short-Term Memory (LSTM)</strong> neural network to predict stock prices. Users can input a stock ticker symbol and receive a prediction for the next day's closing price, along with an interactive visualization of historical price data. Built with <strong>Flask</strong>, <strong>PyTorch</strong>, and <strong>Tailwind CSS</strong>, this application is both powerful and user-friendly.</p>
        <h2>✨ Features</h2>
        <ul>
            <li><strong>📊 Stock Price Prediction</strong>: Predicts the next day's closing price for any given stock ticker.</li>
            <li><strong>📅 Historical Data Visualization</strong>: Displays historical price data in an interactive chart powered by <strong>Plotly</strong>.</li>
            <li><strong>🎨 User-Friendly Interface</strong>: A clean and intuitive web interface designed with <strong>Tailwind CSS</strong>.</li>
            <li><strong>🤖 LSTM Model</strong>: Utilizes a pre-trained LSTM model for accurate stock price predictions.</li>
        </ul>
        <h2>🚀 Requirements</h2>
        <ul>
            <li>Python 3.8+</li>
            <li>Flask</li>
            <li>PyTorch</li>
            <li>yfinance</li>
            <li>scikit-learn</li>
            <li>numpy</li>
            <li>Plotly (for chart visualization)</li>
            <li>Tailwind CSS (for styling)</li>
        </ul>
        <h2>🔧 Installation</h2>
        <ol>
            <li><strong>Clone the repository</strong>:
                <div class="highlight">
                    <code>git clone https://github.com/yourusername/stock-price-predictor.git</code><br>
                    <code>cd stock-price-predictor</code>
                </div>
            </li>
            <li><strong>Create a virtual environment</strong>:
                <div class="highlight">
                    <code>python -m venv venv</code><br>
                    <code>source venv/bin/activate</code> <em>(On Windows use <code>venv\Scripts\activate</code>)</em>
                </div>
            </li>
            <li><strong>Install the required packages</strong>:
                <div class="highlight">
                    <code>pip install -r requirements.txt</code>
                </div>
            </li>
            <li><strong>Train the model</strong> (optional):
                <div class="highlight">
                    <code>python train_model.py</code>
                </div>
                This will train the LSTM model and save it as <code>stock_predictor.pth</code>.
            </li>
            <li><strong>Run the application</strong>:
                <div class="highlight">
                    <code>python app.py</code>
                </div>
                The application will be available at <a href="http://127.0.0.1:5000/" target="_blank">http://127.0.0.1:5000/</a>.
            </li>
        </ol>
        <h2>📖 Usage</h2>
        <ol>
            <li><strong>Access the Web Interface</strong>: Open your web browser and navigate to <a href="http://127.0.0.1:5000/" target="_blank">http://127.0.0.1:5000/</a>.</li>
            <li><strong>Enter a Stock Ticker</strong>: In the input field, enter a stock ticker symbol (e.g., AAPL, GOOG, MSFT) and click "Predict".</li>
            <li><strong>View the Prediction</strong>: The application will display the predicted closing price for the next trading day, along with the percentage change from the last closing price.</li>
            <li><strong>Explore Historical Data</strong>: Use the buttons above the chart to view historical price data for different time periods (1 week, 1 month, 1 year, 5 years).</li>
        </ol>
        <h2>📂 Project Structure</h2>
        <ul>
            <li><code>app.py</code>: The main Flask application that handles web requests and serves the prediction results.</li>
            <li><code>train_model.py</code>: Script to train the LSTM model using historical stock data.</li>
            <li><code>templates/index.html</code>: The HTML template for the web interface, styled with Tailwind CSS.</li>
            <li><code>stock_predictor.pth</code>: Pre-trained LSTM model weights.</li>
        </ul>
        <h2>🤝 Contributing</h2>
        <p>Contributions are welcome! Please feel free to submit a pull request or open an issue for any bugs or feature requests.</p>
        <h2>📜 License</h2>
        <p>This project is licensed under the MIT License. See the <a href="LICENSE" target="_blank">LICENSE</a> file for details.</p>
        <h2>🙏 Acknowledgments</h2>
        <ul>
            <li><strong>yfinance</strong>: For fetching historical stock data.</li>
            <li><strong>PyTorch</strong>: For building and training the LSTM model.</li>
            <li><strong>Flask</strong>: For creating the web application.</li>
            <li><strong>Tailwind CSS</strong>: For styling the web interface.</li>
            <li><strong>Plotly</strong>: For interactive chart visualization.</li>
        </ul>
        <h2>⚠️ Disclaimer</h2>
        <p>This application is for educational purposes only. The stock price predictions are based on historical data and should not be considered as financial advice. Always do your own research before making any investment decisions.</p>
        <div class="button" onclick="window.location.href='https://https://github.com/BryTak1/stock_predictor'">
            View on GitHub
        </div>
    </div>
</body>
</html>