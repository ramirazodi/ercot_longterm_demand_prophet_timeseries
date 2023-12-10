![longterm_demand_predictions_2023](https://github.com/ramirazodi/ercot_longterm_demand_prophet_timeseries/assets/106940649/9d86dc4f-fc8a-497b-9b0d-149b077aed13)
![longterm_demand_predictions_2017](https://github.com/ramirazodi/ercot_longterm_demand_prophet_timeseries/assets/106940649/c51ceb31-9a85-481f-b8fd-3197bc7055da)
![prophet_fit](https://github.com/ramirazodi/ercot_longterm_demand_prophet_timeseries/assets/106940649/8f9b37a0-b6ca-4335-ba81-ee188eebd18f)


# ERCOT Long-Term Trend & Demand Forecasting Using Facebook's Prophet Model

This project focuses on forecasting long-term electricity demand for ERCOT (Electric Reliability Council of Texas) using Facebook's Prophet time series forecasting tool. Electricity demand is generally highly correlated with weather, and most demand forecasting models—especially short-term ones—achieve high accuracy by predicting demand based on weather features. However, forecasting long-term electricity demand for rapidly growing grids like ERCOT, driven by electrification and economic growth, requires accurately modeling and decomposing the growth trend component. The ideal approach often involves decomposing the long-term demand into its trend and seasonal components, then utilizing a gradient boosting model that uses weather features to predict a stationary outlook of long-term demand, which is later adjusted to reflect the long-term trend.

For simplicity, this project uses a less conventional approach by employing a single model (Facebook Prophet) to predict long-term demand, incorporating both the long-term growth trend and weather drivers in a single machine learning model. Facebook's Prophet model is chosen for its ability to model time series qualities and handle regressors using key weather features such as Heating Degree Hours (HDH), Cooling Degree Hours (CDH), and Global Horizontal Irradiance (GHI). The model is simplified by training at a grid-level rather than a zonal-level granularity, but the results are still indicative, and the accuracy achieved is acceptable. Higher accuracy could certainly be attained by applying a more granular or bottom-up approach and including additional key regressors like economic indicators and commodity prices.

The analysis includes data preprocessing, model fitting, validations, predictions, and visualization of the electricity demand trends and forecast. Grid-level historical data, including key weather regressors, are provided in the 'inputs' folder of the project files.


# Project Overview

The project consists of several Python modules, each handling different aspects of the forecasting process:

 - read_data.py: Reads and preprocesses the input data for time series analysis.
 - future_weather.py: Prepares future weather data for predictive modeling.
 - prophet_fit.py: Configures and trains the Prophet model on historical demand data.
 - demand_prediction.py: Generates future demand predictions using the trained model.
 - create_plots.py: Visualizes the results and insights from the model.
 - main.py: Orchestrates the entire process from data preparation to visualization.


# How To Use

1) Install the required libraries by running: pip install -r requirements.txt.
2) Run the script: 'main.py' in your favorite IDE. The project was developed in PyCharm. 
3) The script will generate several output files:
	- 'longterm_demand_predictions.png': Image of historical grid demand as well as predicted grid demand projected 5 years out. The model also plots the decomposed trend line. 
	- 'prophet_fit.png': Image of the model fit on historical data as well as the train set, hold-out sets, and predictions of hold-out set. 
	- 'demand_predictions_and_features.csv': Model-ready data that was used to created predictions. Includes historical as well as 5 years out future data except for predicitons. 
	- 'prophet_output.csv': Typical Prophet output file which includes, predicitons, actuals, decompositions, encoding and other transformed features by the model. 


# Prerequisites

Before running this project, ensure you have Python installed on your system. This project was developed using Python 3.8.
