import streamlit as st
from datetime import date, datetime
import pandas as pd
import yfinance as yf
import plotly.express as px
import numpy as np
import matplotlib.pyplot as plt
import mplfinance as mpf
import plotly.graph_objects as go
from alpha_vantage.fundamentaldata import FundamentalData
from stocknews import StockNews
from lightweight_charts import Chart
from prophet import Prophet
from prophet.plot import plot_plotly
import requests
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, InputLayer

# -------------------- Streamlit Config --------------------
st.set_page_config(page_title='StockPrice Prediction', layout='wide')
st.snow()

st.markdown("<h1 style='text-align: center;'>Stock Dashboard</h1>", unsafe_allow_html=True)

# -------------------- Sidebar for personal information --------------------


# -------------------- Data Fetching Utility --------------------
def fetch_data(symbol, start_dt, end_dt):
    """
    Download data from Yahoo Finance with auto_adjust=False,
    flatten any MultiIndex, rename columns to standard names,
    move the date index to a 'Date' column,
    and reorder columns so 'Date' is first.
    """
    df = yf.download(symbol, start=start_dt, end=end_dt, auto_adjust=False)

    # Flatten if multi-index columns (common with single ticker downloads)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join(col).strip() for col in df.columns.values]

    # Also rename columns that might appear as 'Close_MSFT' to 'Close', etc.
    rename_dict = {
        f'Adj Close_{symbol}': 'Adj Close',
        f'Close_{symbol}': 'Close',
        f'Open_{symbol}': 'Open',
        f'High_{symbol}': 'High',
        f'Low_{symbol}': 'Low',
        f'Volume_{symbol}': 'Volume',
        f'Adj_Close_{symbol}': 'Adj Close',
        f'Close_{symbol.upper()}': 'Close',   # If ticker is uppercase
        f'Open_{symbol.upper()}': 'Open',
        f'High_{symbol.upper()}': 'High',
        f'Low_{symbol.upper()}': 'Low',
        f'Volume_{symbol.upper()}': 'Volume',
        f'Adj_Close_{symbol.upper()}': 'Adj Close',
    }
    for old_col, new_col in rename_dict.items():
        if old_col in df.columns:
            df.rename(columns={old_col: new_col}, inplace=True)

    # Move the DateIndex to a 'Date' column
    df['Date'] = df.index
    df.reset_index(drop=True, inplace=True)

    # Reorder columns so 'Date' is at the front
    cols = ['Date'] + [col for col in df.columns if col != 'Date']
    df = df[cols]

    return df

# A separate function for forecast data, if needed
def fetch_forecast_data(symbol, start_dt, end_dt):
    """
    Same logic for forecast downloads (Prophet).
    """
    df = yf.download(symbol, start=start_dt, end=end_dt, auto_adjust=False)

    # Flatten if multi-index
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join(col).strip() for col in df.columns.values]

    rename_dict = {
        f'Adj Close_{symbol}': 'Adj Close',
        f'Close_{symbol}': 'Close',
        f'Open_{symbol}': 'Open',
        f'High_{symbol}': 'High',
        f'Low_{symbol}': 'Low',
        f'Volume_{symbol}': 'Volume',
        f'Adj_Close_{symbol}': 'Adj Close',
        f'Close_{symbol.upper()}': 'Close',
        f'Open_{symbol.upper()}': 'Open',
        f'High_{symbol.upper()}': 'High',
        f'Low_{symbol.upper()}': 'Low',
        f'Volume_{symbol.upper()}': 'Volume',
        f'Adj_Close_{symbol.upper()}': 'Adj Close',
    }
    for old_col, new_col in rename_dict.items():
        if old_col in df.columns:
            df.rename(columns={old_col: new_col}, inplace=True)

    df['Date'] = df.index
    df.reset_index(drop=True, inplace=True)

    cols = ['Date'] + [col for col in df.columns if col != 'Date']
    df = df[cols]

    return df

# -------------------- Streamlit Inputs --------------------
ticker = st.text_input('Ticker')
start_date = st.date_input('Start Date', value=pd.to_datetime('2024-01-01'))
end_date = st.date_input('End Date', value=pd.to_datetime('today'))

if ticker:
    @st.cache_data
    def load_data(symbol):
        return fetch_data(symbol, start_date, end_date)

    data_load_state = st.text("Loading data...")
    data = load_data(ticker)
    data_load_state.text("Loading data...done!")

    # Check if DataFrame is empty or missing critical columns
    required_cols = {"Open", "High", "Low", "Close"}
    if data.empty or not required_cols.issubset(data.columns):
        st.warning("No valid data found for the specified ticker/date range or missing columns.")
        st.stop()

    # Ticker object for summary info
    tickerData = yf.Ticker(ticker)

    # Create tabs
    Summary, pricing_data, Forecast, Moving_Averages, Model_Prediction, news, fundamental_data = st.tabs(
        ["Summary", "Pricing Data", "Forecast", "Moving_Averages", "Model_Prediction", "Top 10 News", "Fundamental Data"]
    )

    # -------------------- Summary Tab --------------------
    with Summary:
        buttonClicked = st.button('Set')
        if buttonClicked:
            try:
                info = tickerData.get_info()  # Updated approach to get info
                st.header("Profile")
                st.metric("Sector", info.get("sector", "N/A"))
                st.metric("Industry", info.get("industry", "N/A"))
                st.metric("Website", info.get("website", "N/A"))
                st.metric("Market Cap", info.get("marketCap", "N/A"))
            except Exception as e:
                st.error(f"Error fetching data: {e}")

        try:
            info = tickerData.get_info()
            logo_url = info.get('logo_url')
        except Exception as e:
            st.error(f"Error fetching logo: {e}")
            logo_url = None

        if logo_url:
            st.markdown(f'<img src="{logo_url}" alt="Company Logo">', unsafe_allow_html=True)
        else:
            st.markdown("No company logo available")

        string_name = info.get('longName', 'Company Name Unavailable')
        st.header(f'**{string_name}**')
        string_summary = info.get('longBusinessSummary', 'Business Summary Unavailable')
        st.info(string_summary)

    # -------------------- Pricing Data Tab --------------------
    with pricing_data:
        st.header('Price Movement')
        st.write(data)  # Show raw data if needed

        # Combined Opening and Closing Price Chart
        st.subheader('Opening and Closing Price Chart')
        fig_price = go.Figure()
        fig_price.add_trace(go.Scatter(x=data['Date'], y=data['Close'], mode='lines', name='Closing Price'))
        fig_price.add_trace(go.Scatter(x=data['Date'], y=data['Open'], mode='lines', name='Opening Price'))
        fig_price.update_layout(title=f'{ticker} Opening and Closing Price')
        st.plotly_chart(fig_price)

        # High and Low Price Chart
        st.subheader('High and Low Price Chart')
        fig_high_low_price = go.Figure()
        fig_high_low_price.add_trace(go.Scatter(x=data['Date'], y=data['High'], mode='lines', name='High'))
        fig_high_low_price.add_trace(go.Scatter(x=data['Date'], y=data['Low'], mode='lines', name='Low'))
        fig_high_low_price.update_layout(title=f'{ticker} High and Low Price')
        st.plotly_chart(fig_high_low_price)

        # All Four Prices Chart
        st.subheader('Prices Chart')
        fig_all_prices = go.Figure()
        fig_all_prices.add_trace(go.Scatter(x=data['Date'], y=data['High'], mode='lines', name='High'))
        fig_all_prices.add_trace(go.Scatter(x=data['Date'], y=data['Low'], mode='lines', name='Low'))
        fig_all_prices.add_trace(go.Scatter(x=data['Date'], y=data['Open'], mode='lines', name='Open'))
        fig_all_prices.add_trace(go.Scatter(x=data['Date'], y=data['Close'], mode='lines', name='Close'))
        fig_all_prices.update_layout(title=f'{ticker} All Prices', xaxis=dict(showgrid=True))
        st.plotly_chart(fig_all_prices)

        # Candlestick Chart
        st.header('Candlestick Chart')
        fig_candlestick = go.Figure(
            data=[go.Candlestick(x=data['Date'],
                                 open=data['Open'],
                                 high=data['High'],
                                 low=data['Low'],
                                 close=data['Close'])]
        )
        fig_candlestick.update_layout(title=f'{ticker} Candlestick Chart')
        st.plotly_chart(fig_candlestick)

        # Pricing Data
        st.subheader('Pricing Data')
        st.write(data.sort_values(by='Date', ascending=False))

        # Calculations
        df_temp = data.copy()
        df_temp['% Change'] = df_temp['Close'] / df_temp['Close'].shift(1)
        df_temp.dropna(inplace=True)
        annual_return = df_temp['% Change'].mean() * 252 * 100
        st.write('Annual Return:', annual_return, '%')
        stdev = np.std(df_temp['% Change']) * np.sqrt(252)
        st.write('Standard Deviation:', stdev * 100, '%')
        st.write('Risk Adj. Return:', annual_return / (stdev * 100))

    # -------------------- Forecast Tab --------------------
    with Forecast:
        starting = "2015-01-01"
        today_str = date.today().strftime("%Y-%m-%d")
        n_years = 1
        period = n_years * 365

        @st.cache_data
        def load_forecast_data(symbol):
            return fetch_forecast_data(symbol, starting, today_str)

        data_load_state = st.text("Loading forecast data...")
        forecast_data = load_forecast_data(ticker)
        data_load_state.text("Loading forecast data...done!")

        # Check for valid columns
        if forecast_data.empty or 'Close' not in forecast_data.columns:
            st.warning("No valid forecast data found. Check the ticker or date range.")
            st.stop()

        st.subheader('Raw Data')
        st.write(forecast_data.sort_values(by='Date', ascending=False))

        def plot_raw_data():
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=forecast_data['Date'], y=forecast_data['Open'], name='Stock_Open'))
            fig.add_trace(go.Scatter(x=forecast_data['Date'], y=forecast_data['Close'], name='Stock_Close'))
            fig.layout.update(title_text='Time Series Data', xaxis_rangeslider_visible=True)
            st.plotly_chart(fig)

        plot_raw_data()

        # Prepare data for Prophet
        df_train = forecast_data[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
        # Ensure 'y' is numeric
        df_train['y'] = pd.to_numeric(df_train['y'], errors='coerce')
        df_train.dropna(subset=['ds', 'y'], inplace=True)

        m = Prophet()
        m.fit(df_train)
        future = m.make_future_dataframe(periods=period)
        forecast = m.predict(future)

        st.subheader('Forecast Data')
        st.write(forecast.tail())

        fig1 = plot_plotly(m, forecast)
        st.plotly_chart(fig1)

        st.write('Forecast Components')
        fig2 = m.plot_components(forecast)
        st.write(fig2)

    # -------------------- Moving Averages Tab --------------------
    with Moving_Averages:
        end_dt = datetime.now()
        start_dt = datetime(end_dt.year - 20, end_dt.month, end_dt.day)

        # Reuse the fetch_data function
        newdata = fetch_data(ticker, start_dt, end_dt)

        def plot_graph(figsize, values, column_name):
            plt.figure()
            values.plot(figsize=figsize)
            plt.xlabel('Years')
            plt.ylabel(column_name)
            plt.title(f"{column_name} of {ticker}")
            plt.show()
            st.pyplot(plt.gcf())

        for column in ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']:
            if column in newdata.columns:
                plot_graph((15, 5), newdata[column], column)

        if 'Close' in newdata.columns:
            newdata['MA_for_250_days'] = newdata['Close'].rolling(250).mean()
            plot_graph((15, 5), newdata[['Close', 'MA_for_250_days']], 'MA_for_250_days')

            newdata['MA_for_100_days'] = newdata['Close'].rolling(100).mean()
            plot_graph((15, 5), newdata[['Close', 'MA_for_100_days']], 'MA_for_100_days')

            plot_graph((15, 5), newdata[['Close', 'MA_for_100_days', 'MA_for_250_days']], 'MA')
            newdata['percentage_change_cp'] = newdata['Close'].pct_change()
            plot_graph((15, 5), newdata['percentage_change_cp'], 'Percentage_Change')

        Adj_close_price = newdata[['Close']] if 'Close' in newdata.columns else None

    # -------------------- Model Prediction Tab --------------------
    with Model_Prediction:
        if st.button('Predict'):
            if Adj_close_price is None or Adj_close_price.empty:
                st.error("No 'Close' data available for modeling.")
            else:
                # Data preprocessing
                scaler = MinMaxScaler(feature_range=(0, 1))
                scaled_data = scaler.fit_transform(Adj_close_price)

                x_data = []
                y_data = []
                for i in range(100, len(scaled_data)):
                    x_data.append(scaled_data[i - 100:i])
                    y_data.append(scaled_data[i])
                x_data, y_data = np.array(x_data), np.array(y_data)

                splitting_len = int(len(x_data) * 0.8)
                x_train, y_train = x_data[:splitting_len], y_data[:splitting_len]
                x_test, y_test = x_data[splitting_len:], y_data[splitting_len:]

                model = Sequential()
                model.add(InputLayer(input_shape=(x_train.shape[1], 1)))
                model.add(LSTM(128, return_sequences=True))
                model.add(LSTM(64, return_sequences=False))
                model.add(Dense(25))
                model.add(Dense(1))
                model.compile(optimizer='adam', loss='mean_squared_error')

                model.fit(x_train, y_train, batch_size=1, epochs=1)

                predictions = model.predict(x_test)
                inv_predictions = scaler.inverse_transform(predictions)
                inv_y_test = scaler.inverse_transform(y_test)

                # Build a DataFrame to compare predictions vs original
                idx_range = newdata.index[splitting_len+100:] if splitting_len+100 < len(newdata) else range(len(inv_predictions))
                ploting_data = pd.DataFrame({
                    'original_test_data': inv_y_test.reshape(-1),
                    'predictions': inv_predictions.reshape(-1)
                }, index=idx_range)

                st.subheader('Plotting Data')
                st.write(ploting_data)

                def plot_graph(figsize, df_plot, title):
                    plt.figure()
                    df_plot.plot(figsize=figsize)
                    plt.title(title)
                    plt.xlabel('Index')
                    plt.ylabel('Price')
                    plt.show()
                    st.pyplot(plt.gcf())

                plot_graph((15, 6), ploting_data, 'Test Data')

                # Combine with original close data up to splitting_len+100
                if splitting_len+100 <= len(Adj_close_price):
                    combined = pd.concat([Adj_close_price[:splitting_len+100], ploting_data], axis=0)
                else:
                    combined = ploting_data
                plot_graph((15, 6), combined, 'Whole Data')

                rmse = np.sqrt(mean_squared_error(inv_y_test, inv_predictions))
                r2 = r2_score(inv_y_test, inv_predictions)
                st.write('Root Mean Squared Error (RMSE):', rmse)
                st.write('R-squared (R2) score:', r2)

                # Predict tomorrow's price using the last 100 days
                last_100_days = scaled_data[-100:]
                last_100_days = last_100_days.reshape((1, 100, 1))
                predicted_price_scaled = model.predict(last_100_days)
                predicted_price = scaler.inverse_transform(predicted_price_scaled)
                formatted_price = "{:.2f}".format(predicted_price[0][0])
                st.write('Predicted Closing Price for Tomorrow:', formatted_price)

    # -------------------- News Tab --------------------
    with news:
        st.header('News')
        sn = StockNews(ticker, save_news=False)
        df_news = sn.read_rss()
        # If df_news is empty, handle gracefully
        if df_news.empty:
            st.write("No news found.")
        else:
            for i in range(min(10, len(df_news))):
                st.subheader(f'News {i+1}')
                st.write(df_news['published'][i])
                st.write(df_news['title'][i])
                st.write(df_news['summary'][i])
                st.write(f'Title Sentiment: {df_news["sentiment_title"][i]}')
                st.write(f'News Sentiment: {df_news["sentiment_summary"][i]}')

    # -------------------- Fundamental Data Tab --------------------
    with fundamental_data:
        st.header('Fundamental Data')
        key = 'PJWPPYH1QF8YECF9'
        fd = FundamentalData(key, output_format='pandas')

        st.subheader('Balance Sheet')
        balance_sheet = fd.get_balance_sheet_annual(ticker)[0]
        balance_sheet.columns = balance_sheet.columns.str.replace(' ', '_')
        st.write(balance_sheet)

        st.subheader('Income Statement')
        income_statement = fd.get_income_statement_annual(ticker)[0]
        income_statement.columns = income_statement.columns.str.replace(' ', '_')
        st.write(income_statement)

        st.subheader('Cash Flow Statement')
        cash_flow = fd.get_cash_flow_annual(ticker)[0]
        cash_flow.columns = cash_flow.columns.str.replace(' ', '_')
        st.write(cash_flow)

else:
    st.write("Please Enter a Ticker")
