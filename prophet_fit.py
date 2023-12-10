import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import TimeSeriesSplit
from read_data import df_data
from future_weather import df_future_wx

# Data Preparation
df = df_data.copy()
df = (
    df
    .assign(
        datetime=lambda df_: pd.to_datetime(df_.datetime).dt.tz_localize(None),
        ds=lambda df_: df_.datetime,
        y=lambda df_: df_.ercot
    )
    .set_index('ds')
    .loc['2011':]
    .reset_index()
)

# Train-Test Split
test_size = pd.Timedelta(days=60)
split_point = df.ds.max() - test_size
df_train = df[df.ds <= split_point]
df_test = df[df.ds > split_point]

#%%
# Timeseries k-folds validations prior to measuring accuracy on hold-out (df_test) data
n_splits = 3
tscv = TimeSeriesSplit(n_splits=n_splits)

# Lists to store MAPE and MAE for each fold
mape_scores = []
mae_scores = []

for train_idx, val_idx in tscv.split(df_train):
    # Splitting the data for the current fold
    train_fold = df_train.iloc[train_idx]
    val_fold = df_train.iloc[val_idx]

    # Model Configuration and Training (same as in your code)
    model = Prophet(holidays_prior_scale=10.0).add_country_holidays(country_name='US')
    for regressor in ['hdh', 'cdh', 'ghi']:
        model.add_regressor(regressor)
    for seasonality in [('hourly', 1/24, 3), ('daily', 1, 5), ('weekly', 7, 3), ('monthly', 30.5, 3)]:
        model.add_seasonality(name=seasonality[0], period=seasonality[1], fourier_order=seasonality[2])
    model.fit(train_fold[['ds', 'y', 'hdh', 'cdh', 'ghi']])

    # Prediction and Model Evaluation for the current fold
    df_full = val_fold.loc[:, ['ds', 'hdh', 'cdh', 'ghi']]
    df_forecast = model.predict(df_full)
    y_true = val_fold.y
    y_pred = df_forecast[df_forecast.ds.isin(val_fold.ds)].yhat
    mae = mean_absolute_error(y_true, y_pred)
    mae_scores.append(mae)

    print(f"Fold {len(mae_scores)} - MAE: {mae:.2f}")

# Calculate and print the average MAE across all folds
average_mae = np.mean(mae_scores)
print(f"Average MAE across {n_splits} folds: {average_mae:.2f}")

#%%
# Model Configuration and Training for hold-out data accuracy measurement
model = Prophet(holidays_prior_scale=10.0).add_country_holidays(country_name='US')
for regressor in ['hdh', 'cdh', 'ghi']:
    model.add_regressor(regressor)
for seasonality in [('hourly', 1/24, 3), ('daily', 1, 5), ('weekly', 7, 3), ('monthly', 30.5, 3)]:
    model.add_seasonality(name=seasonality[0], period=seasonality[1],
                          fourier_order=seasonality[2])
model.fit(df_train[['ds', 'y', 'hdh', 'cdh', 'ghi']])

# Prediction and Model Evaluation
df_full = df.loc[:, ['ds', 'hdh', 'cdh', 'ghi']]
df_forecast = model.predict(df_full)
y_true = df_test.y
y_pred = df_forecast[df_forecast.ds.isin(df_test.ds)].yhat
mape = (np.abs(y_true - y_pred) / y_true).mean() * 100
mae = mean_absolute_error(y_true, y_pred)

# Output Accuracy Metrics
print(f"MAPE: {mape:.2f}%")
print(f"MAE: {mae:.2f}")

