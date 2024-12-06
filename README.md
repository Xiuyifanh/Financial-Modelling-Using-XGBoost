# AAPL Stock Price Analysis, Prediction, and Forecasting

Welcome to the **AAPL Stock Price Analysis and Prediction** project! This repository contains a comprehensive pipeline for analyzing, predicting, and forecasting Apple Inc. (AAPL) stock prices. The project includes implementations for **long-term analysis**, **short-term prediction**, and **future forecasting** using various data science techniques, including machine learning, feature engineering, and visualization.

---

## Project Overview

The primary goal of this project is to build a robust framework for understanding stock price trends and making accurate predictions. This repository contains:

1. **Long-Term Analysis**: Identifies macro trends over a longer time horizon using historical data and advanced visualizations.
2. **Short-Term Analysis**: Focuses on short-term price movements and trends for tactical decision-making.
3. **Stock Price Prediction**: Employs machine learning models to predict the next trading day's price.
4. **Future Forecasting**: Uses predictive models to forecast stock prices for the next 5–7 trading days.

---

## Features

- **Data Retrieval**: Fetches historical stock data using the Yahoo Finance API (`yfinance`).
- **Feature Engineering**: Adds meaningful indicators such as moving averages (SMA, EMA), volatility, and Relative Strength Index (RSI).
- **Machine Learning Models**: 
  - Gradient Boosting with **XGBoost** for prediction and forecasting.
  - Optimized hyperparameters using **GridSearchCV**.
- **Data Visualization**: Plots training vs. testing data, actual vs. predicted stock prices, and more for insightful analysis.

---

## Technologies Used

- **Python**: The programming language used for data analysis, machine learning, and visualizations.
- **Libraries**:
  - `yfinance`: For downloading financial data.
  - `pandas`: For data manipulation and preprocessing.
  - `numpy`: For numerical operations.
  - `matplotlib`: For plotting and visualizations.
  - `sklearn`: For machine learning models, preprocessing, and evaluation.
  - `xgboost`: For gradient boosting regression model.
  - `seaborn`: For improved data visualizations.

---

## Project Structure

This repository is organized into the following parts:

1. **long_term_analysis.py**: Long-term analysis of stock price trends, with features like SMA, EMA, and volatility.
2. **short_term_prediction.py**: Short-term predictions using the same features, optimized using grid search and cross-validation.
3. **forecasting.py**: Forecasts the next few trading days using XGBoost, including prediction and feature selection.
4. **data_preprocessing.py**: Contains the data fetching and feature engineering scripts.
5. **visualization.py**: Includes various visualizations for better insight into stock trends and predictions.

---

## Requirements

To run this project, you will need the following Python libraries:

- `yfinance`
- `pandas`
- `numpy`
- `matplotlib`
- `sklearn`
- `xgboost`
- `seaborn`

---

## Detailed Breakdown of the Scripts

### 1. **XGBoost_Long_Term_Validation.ipynb**

This script focuses on identifying long-term trends in the stock price data. It uses historical data to calculate moving averages (SMA, EMA), volatility, and other statistical indicators. The visualizations help in understanding the broader price trends over a long period.

### 2. **XGBoost_Short_Term_Validation.ipynb**

This script is designed for short-term predictions. It focuses on predicting stock prices for the next day by utilizing machine learning models (XGBoost). The model is trained using historical stock data, and hyperparameters are optimized using **GridSearchCV**. The results are evaluated using metrics like **R-squared**, **MSE**, **MAE**, and **MAPE**.

### 3. **XGBoost_Long_Term_Forecasting.ipynb**

This script is designed for forecasting future stock prices for the next 5–7 trading days. It uses **XGBoost** along with **feature selection** to improve the performance of the model. It provides predictions for the next week's worth of stock prices.

This script is responsible for generating visualizations such as:

- Training vs. Testing Split plot
- Actual vs. Predicted Prices plot
- Scatter Plot of Actual vs. Predicted prices
- Forecasting visualizations for the next trading week

These visualizations provide insights into the model's performance and help in identifying any discrepancies between the predicted and actual stock prices.

---

## Example Outputs

### 1. **Long-Term Analysis**:

- Visualizations of AAPL's price movements with **SMA** and **EMA** overlaid to show longer-term trends.
- Rolling volatility and RSI calculations for detecting overbought or oversold conditions.

### 2. **Short-Term Prediction**:

- Predicted prices for the next day's stock price.
- Performance metrics like **R-squared**, **MAE**, and **MAPE** showing the model's prediction accuracy.

### 3. **Forecasting**:

- Forecasted prices for the next 5–7 trading days.
- Performance analysis comparing predicted vs. actual results for future stock price trends.

---

## Future Improvements

- **Model Enhancement**: Implement more advanced models such as **LSTM (Long Short-Term Memory)** for time series forecasting.
- **Data Sources**: Integrate additional data sources such as **news sentiment analysis** or **economic indicators** for improving model predictions.
- **Interactive Visualizations**: Add interactive visualizations using libraries like **Plotly** or **Dash** for better user engagement.
- **Deployment**: Create a web application or API that allows users to predict and forecast AAPL stock prices in real-time.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgements

- **Yahoo Finance** for providing free stock data.
- **XGBoost** for the powerful gradient boosting algorithm.
- The **scikit-learn** library for providing essential tools for machine learning and data preprocessing.

---

## Contact

For any questions or feedback, feel free to reach out:

- Email: emiratenihaar+github@gmail.com
- GitHub: (https://github.com/users/Xiuyifanh)
