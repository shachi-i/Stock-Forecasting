# 📈 Stock Market Forecasting 

This project focuses on forecasting the stock prices of top tech companies (such as Google, Apple, Nvidia) using both statistical and deep learning models. The workflow includes data cleaning, in-depth analysis, visualization, and time series forecasting using ARIMA and LSTM models.

---

## 🧰 Tech Stack

- Python
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn
- Statsmodels (ARIMA)
- TensorFlow/Keras (LSTM)

---

## 📊 Dataset Overview

The dataset consists of daily historical stock prices for companies like Apple (AAPL), Amazon (AMZN), Google (GOOGL), Microsoft (MSFT), and NVIDIA (NVDA). Each entry includes:

- `Date`
- `Open`, `High`, `Low`, `Close` prices
- Time range: 2010–2025

---

## 🧼 Data Cleaning & Preprocessing

- Converted the `Date` column to `datetime` format and sorted the dataset.
- Removed or filled missing/null values.
- Renamed columns for consistency.
- Filtered out anomalies or outliers if present.
- Handled different types of financial columns (Open/High/Low/Close) separately and normalized data for visualization and modeling.

---

## 📈 Data Visualization

Used several data visualization techniques to understand stock trends and relationships:

### 1. Line Chart
- Visualized and compared normalized stock prices over time for all companies.

### 2. KDE Plot (Kernel Density Estimation)
- Plotted the distribution of each company's closing prices to understand volatility and skewness.

### 3. 2D KDE Plot
- Explored the joint distribution of pairs of stock prices (e.g., Apple vs Amazon) using density contours.

---

## 🧪 Stationarity Check for ARIMA

Before applying ARIMA, we used the **Augmented Dickey-Fuller (ADF) Test** to verify if the time series is stationary.


[from statsmodels.tsa.stattools import adfuller

result = adfuller(df['Close_AAPL'])
print('ADF Statistic:', result[0])
print('p-value:', result[1]) ]

- If p-value < 0.05, the series is stationary and ready for ARIMA modeling.

- Differencing was used to make the series stationary if needed.

---

## Models Overview

🔹 ARIMA (AutoRegressive Integrated Moving Average)
ARIMA is a classical time series forecasting model that combines:
It works well for linear, stationary data and is interpretable and fast.

🔹 LSTM (Long Short-Term Memory)
LSTM is a type of Recurrent Neural Network (RNN) designed for sequential data. It captures long-term dependencies using memory cells and gates, making it well-suited for time series with complex patterns and seasonality.
It performs better than traditional models on noisy or non-linear financial data.

---

## 📈 Model Performance

Metric	ARIMA	LSTM
MAE	2.31	1.94
RMSE	3.12	2.68
MAPE (%)	~5.1%	~3.6%
R² Score	0.82	0.89

---

## ✅ Conclusion

This project demonstrates two distinct methodologies for stock price forecasting:
ARIMA offers fast, explainable predictions for stationary financial time series.
LSTM captures deeper temporal dependencies and delivers superior performance in volatile or trend-heavy data.
By integrating both statistical and deep learning approaches, this project showcases a strong understanding of:
- Time series preprocessing (stationarity, scaling)
- Forecasting models
- Financial evaluation metrics
- Model comparison and interpretation

This dual-model pipeline is valuable for ML, data science, and financial analytics, especially where interpretability and forecasting accuracy both matter.
