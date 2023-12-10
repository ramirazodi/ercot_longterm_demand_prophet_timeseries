import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
from prophet_fit import df_forecast, split_point, df
from demand_prediction import df_future_forecast

def setup_plot():
    """Configures common plot settings."""
    sns.set(style="whitegrid")
    plt.figure(figsize=(16, 10))
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['figure.facecolor'] = 'white'

def prophet_fit_plot():
    """Creates and saves the Prophet model fit plot."""
    setup_plot()

    # Plotting series with specified colors
    sns.lineplot(data=df, x='ds', y='y', label='Actual', color='black', alpha=0.5)
    sns.lineplot(data=df_forecast, x='ds', y='yhat', label='Prediction',
                 color='turquoise', alpha=0.7)
    sns.lineplot(data=df_forecast, x='ds', y='trend', label='Trend',
                 color='red', alpha=0.8, linestyle='dotted', linewidth=3)
    test_period = df_forecast[df_forecast['ds'] > split_point]
    sns.lineplot(data=test_period, x='ds', y='yhat',
                 color='orange', alpha=0.6, linestyle='-', linewidth=1)
    plt.axvline(x=split_point, color='black', linestyle='--', linewidth=2, alpha=0.8)

    # Adding titles and labels
    plt.title('ERCOT Long-Term Demand - Facebook Prophet Model Fit',
              fontsize=20, fontname='Helvetica', color='black', pad=15)
    plt.xlabel('Year', fontsize=16, fontname='Helvetica', color='black', labelpad=15)
    plt.ylabel('Demand (MW)', fontsize=16, fontname='Helvetica', color='black', labelpad=10)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=3, fontsize=12)

    # Tick Labels and Grid
    configure_axes()

    # Save and show the plot
    plt.savefig('outputs/prophet_fit.png', dpi=300, bbox_inches='tight')
    plt.show()

def lt_demand_pred_plot():
    """Creates and saves the long-term demand prediction plot."""
    setup_plot()

    # Plotting series with specified colors
    sns.lineplot(data=df, x='ds', y='y', label='Actual', color='black', alpha=0.5)
    sns.lineplot(data=df_future_forecast, x='ds', y='yhat', label='Prediction',
                 color='turquoise', alpha=0.7)
    sns.lineplot(data=df_future_forecast, x='ds', y='trend', label='Trend',
                 color='red', alpha=0.8, linestyle='dotted', linewidth=3)

    # Adding titles and labels
    plt.title('ERCOT Long-Term Demand - 2022 Weather', fontsize=20, fontname='Helvetica',
              color='black', pad=15)
    plt.xlabel('Year', fontsize=16, fontname='Helvetica', color='black', labelpad=15)
    plt.ylabel('Demand (MW)', fontsize=16, fontname='Helvetica', color='black', labelpad=10)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 0.98), ncol=4, fontsize=12)

    # Tick Labels and Grid
    configure_axes()

    # Save and show the plot
    plt.savefig('outputs/longterm_demand_predictions.png', dpi=300, bbox_inches='tight')
    plt.show()

def configure_axes():
    """Configures axes labels, ticks, and grid."""
    plt.xticks(fontsize=13, fontname='Helvetica', color='black')
    plt.yticks(fontsize=13, fontname='Helvetica', color='black')
    plt.gca().yaxis.set_major_formatter(mticker.StrMethodFormatter('{x:,.0f}'))
    plt.gca().xaxis.set_major_locator(mdates.YearLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.gca().yaxis.set_major_locator(mticker.MultipleLocator(10000))
    plt.ylim((0, 100000))
    plt.grid(axis='x', linestyle='-', linewidth=1, alpha=0.8)
    plt.grid(axis='y', linestyle='-', linewidth=1, alpha=0.8)
    plt.xticks(rotation=0)
