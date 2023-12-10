import pandas as pd

# Load dataset
df_data = pd.read_csv('inputs/ercot_load_clean.csv')

# Data preparation for timeseries modeling
df_data = (
    df_data
    .assign(
        datetime=lambda df_: pd.to_datetime(df_.datetime),
        date=lambda df_: df_.datetime.dt.date
    )
    .set_index('datetime')
    .resample('H').ffill()
    .reset_index()
)


