# Stock_Price_Pred

https://stockpricepred-5cr9tap38zxwj2f3hfgzxf.streamlit.app/

Welcome to the Stock Market Dashboard! This web application provides a comprehensive analysis of stock market data, including price movements, forecasts, moving averages, fundamental data, and related news. It is designed to help users gain insights into their favorite stocks and make informed investment decisions.

Features
Summary: Overview of the selected stock, including sector, industry, market cap, and business summary.
Pricing Data: Visualizes stock price movements through various types of charts.
Opening and Closing Price Chart
High and Low Price Chart
All Prices Chart
Candlestick Chart
Forecast: Uses the Prophet model to forecast future stock prices.
Displays forecast data and components.
Moving Averages: Plots moving averages and percentage changes.
Model Prediction: Predicts stock prices using an LSTM model.
Displays original test data against predictions.
Calculates RMSE and R2 score for model evaluation.
Fundamental Data: Provides balance sheet, income statement, and cash flow statement.
News: Fetches and displays the top 10 news articles related to the stock.
Personal Information Sidebar
The sidebar provides personal information about the developer, Jaykumar Pal:

Photo
Contact information
GitHub and LinkedIn links
Brief bio
Getting Started
Prerequisites
Python 3.7 or above
Streamlit
pandas
yfinance
plotly
matplotlib
mplfinance
Prophet
cufflinks
requests
scikit-learn
keras
alpha_vantage
stocknews
lightweight_charts
pandas_ta
Usage
Enter the stock ticker symbol in the input box.
Select the start and end dates for the data.
Explore various tabs to analyze the stock data:
Summary: Get an overview of the company.
Pricing Data: Visualize price movements.
Forecast: See future price predictions.
Moving Averages: Analyze moving averages.
Model Prediction: View predictions from the LSTM model.
Fundamental Data: Check financial statements.
News: Read the latest news articles.
Contributing
Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

License
This project is licensed under the MIT License - see the LICENSE file for details.
