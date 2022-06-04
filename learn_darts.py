import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from darts import TimeSeries
from darts.models import Theta
from darts.metrics import mape
from darts.utils.missing_values import fill_missing_values
import yfinance as yf


def get_data(ticker: str) -> pd.DataFrame:
    ticker = ticker.upper()
    try:
        df = pd.read_csv(f"{ticker}.csv")
    except FileNotFoundError:
        df = yf.download(ticker, start="2020-01-01", verbose=False)
        df.to_csv(f"{ticker}.csv")
    return df


df = get_data("IWV")
df["Date"] = pd.to_datetime(df["Date"])

series = TimeSeries.from_dataframe(
    df, time_col="Date", value_cols=["Close"], freq="B", fill_missing_dates=True
)
series = fill_missing_values(series)

train, test = series.split_before(0.75)

thetas = np.linspace(-10, 10, 50)

best_mape = float("inf")
best_theta = 0

for theta in thetas:
    model = Theta(theta)
    model.fit(train)
    pred_theta = model.predict(len(test))
    res = mape(test, pred_theta)

    if res < best_mape:
        best_mape = res
        best_theta = theta

best_theta_model = Theta(best_theta)
best_theta_model.fit(train)

historical_fcast_theta = best_theta_model.historical_forecasts(
        series, start=0.6, forecast_horizon=30, verbose=True
)

train.plot(label="train")
test.plot(label="test")
historical_fcast_theta.plot(label="Martin is a G, eh")
plt.show()
