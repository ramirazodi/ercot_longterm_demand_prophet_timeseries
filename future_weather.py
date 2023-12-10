import pandas as pd
from read_data import df_data

# Create weather assumptions for predictions of future periods.
# Normal weather can be achieved by removing the .loc section
df_future_wx = (
    df_data
    .set_index('datetime')
    .drop(columns=['date'])
    .loc['2022']
    .groupby(['month', 'day', 'hour_begin', 'hour_end'])
    .mean()
    .round(1)
    .reset_index()
    .loc[:, ['month', 'day', 'hour_begin', 'hour_end', 'temp', 'ghi']]
    .assign(
        hdh=lambda df_: df_.temp.sub(65).multiply(-1).clip(0),
        cdh=lambda df_: df_.temp.sub(65).clip(0),
    )
)

