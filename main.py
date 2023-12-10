from create_plots import prophet_fit_plot, lt_demand_pred_plot
from demand_prediction import df_future, df_future_forecast

def main():
    # Call the plot creation functions
    prophet_fit_plot()
    lt_demand_pred_plot()

    # Export dataframes to CSV files in the Outputs folder
    df_future_forecast.to_csv('outputs/prophet_output.csv', index=False)
    df_future.to_csv('outputs/demand_predictions_and_features.csv', index=False)

if __name__ == "__main__":
    main()
