#%%
import pandas as pd
from prophet import Prophet
from future_weather import df_future_wx
from prophet_fit import df, model

# Future DataFrame Creation
future_years = 5
hours_in_year = 24 * 365.25
future_periods = int(future_years * hours_in_year)

# Extending DataFrame 5 years into the future
df_future = model.make_future_dataframe(periods=future_periods, freq='H')

# Merging and Preparing Future Data
df_future = (
    df_future
    .merge(df.loc[:, ['ds', 'hdh', 'cdh', 'ghi', 'y']],
           on=['ds'],
           how='left'
           )
    .assign(
        month=lambda df_: df_.ds.dt.month,
        day=lambda df_: df_.ds.dt.day,
        hour_begin=lambda df_: df_.ds.dt.hour,
        hour_end=lambda df_: df_.hour_begin.add(1)
    )
    .merge(df_future_wx.loc[:, ['month', 'day', 'hour_begin', 'hour_end', 'hdh', 'cdh', 'ghi']],
           on=['month', 'day', 'hour_begin', 'hour_end'],
           how='left'
           )
    .assign(
            hdh_x=lambda df_: df_.hdh_x.fillna(df_.hdh_y).bfill(),
            cdh_x=lambda df_: df_.cdh_x.fillna(df_.cdh_y).bfill(),
            ghi_x=lambda df_: df_.ghi_x.fillna(df_.ghi_y).bfill(),
        )
    .rename(columns={'hdh_x': 'hdh', 'cdh_x': 'cdh', 'ghi_x': 'ghi'})
    .loc[:, ['ds', 'y', 'hdh', 'cdh', 'ghi']]
    .sort_values('ds')
)

#%%%
# Use the trained Prophet model to make predictions on the future DataFrame
df_future_forecast = model.predict(df_future)

#%%
# Create cumulative trend and  year over year trend using Prophet's decomposed trend component
df_future_forecast = (
    df_future_forecast
    .assign(
        trend_normalized=lambda df_: df_.trend.sub(df_.trend.iloc[0]),
        trend_yoy=lambda df_: ((df_.trend - df_.trend.shift(8760)) / df_.trend.shift(
            8760)).multiply(100).bfill().round(1)
    )
)

