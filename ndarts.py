from typing import List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from darts import TimeSeries, concatenate
from darts.models import NBEATSModel
from darts.dataprocessing.transformers import Scaler
from darts.utils.missing_values import fill_missing_values
from darts.metrics import mape, r2_score
import yfinance as yf


def get_data(ticker: str) -> pd.DataFrame:
    ticker = ticker.upper()
    try:
        df = pd.read_csv(f"data/{ticker}.csv")
    except FileNotFoundError:
        df = yf.download(ticker, start="2020-01-01", verbose=False)
        df.to_csv(f"data/{ticker}.csv")
    return df


def display_forecast(pred_series, ts_transformed, forecast_type, start_date=None):
    plt.figure(figsize=(8, 5))
    if start_date:
        ts_transformed = ts_transformed.drop_before(start_date)
    ts_transformed.univariate_component(0).plot(label="actual")
    pred_series.plot(label=("historic " + forecast_type + " forecasts"))
    plt.title(f"R2: {r2_score(ts_transformed.univariate_component(0), pred_series)}")
    plt.legend()


def create_series(df: pd.DataFrame, column: str) -> TimeSeries:
    df["Date"] = pd.to_datetime(df["Date"])
    series = TimeSeries.from_dataframe(
        df, time_col="Date", value_cols=[column], freq="B", fill_missing_dates=True
    )
    series = fill_missing_values(series)
    return series


def create_covariates(base: TimeSeries, other: TimeSeries) -> TimeSeries:
    covariates = base.stack(other)
    scaler = Scaler()
    covariates = scaler.fit_transform(base)
    return covariates


def generate_data(base: List[str], covs: List[List[str]]):
    base_tick = create_series(get_data(base[0]), base[1])
    base_train, base_test = base_tick.split_before(0.9)
    cov_list, test_list, train_list = [], [], []
    for cov in covs:
        series = create_series(get_data(cov[0]), cov[1])
        the_cov = create_covariates(base_tick, series)
        cov_list.append(the_cov)
        train, test = the_cov.split_before(0.9)
        test_list.append(test)
        train_list.append(train)
        return {
            "base": base_tick,
            "base_train": base_train,
            "base_test": base_test,
            "covs": cov_list,
            "cov_train": train_list,
            "cov_test": test_list,
        }


covariates = [["AAPL", "Open"], ["MSFT", "Close"], ["TSLA", "Close"]]
data_dict = generate_data(["AAPL", "Close"], covariates)


parameters = {"forecast_horizon": [10, 20, 30, 40]}

model_nbeats = NBEATSModel(
    input_chunk_length=30,
    output_chunk_length=7,
    generic_architecture=True,
    num_stacks=10,
    num_blocks=1,
    num_layers=4,
    layer_widths=512,
    n_epochs=10,
    nr_epochs_val_period=1,
    batch_size=800,
    model_name="nbeats_run",
)
"""
parameters = dict(
    input_chunk_length=30,
    output_chunk_length=7,
    generic_architecture=True,
    num_stacks=10,
    num_blocks=1,
    num_layers=4,
    layer_widths=512,
    n_epochs=10,
    nr_epochs_val_period=1,
    batch_size=800,
    model_name="nbeats_run",
    forecast_horizon=7,
)

model_nbeats = NBEATSModel.gridsearch(parameters=parameters, series=train)
"""

model_nbeats.fit(
    data_dict["base_train"],
    val_series=data_dict["base_test"],
    past_covariates=data_dict["cov_train"],
    val_past_covariates=data_dict["cov_test"],
    verbose=True,
)

pred_series = model_nbeats.historical_forecasts(
    data_dict["base"],
    past_covariates=data_dict["covs"],
    forecast_horizon=7,
    stride=5,
    retrain=False,
    verbose=True,
)
display_forecast(pred_series, data_dict["base"], "7 day")
plt.show()
