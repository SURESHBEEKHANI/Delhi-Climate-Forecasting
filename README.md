# ForecastMaster

ForecastMaster is a Streamlit application for time series forecasting of Delhi's climate. The application supports:
- Temperature Forecasting
- Humidity Forecasting
- Wind Speed Forecasting
- Mean Pressure Forecasting

## Features
- **Interactive Forecasts:** Upload CSV data and visualize forecasts using Plotly and Matplotlib.
- **Multiple Metrics:** Choose which forecast to display via sidebar options.
- **Forecast Components:** View trend and seasonality components for deeper insights.

## Requirements
- Python 3.x
- Streamlit
- Pandas
- Prophet
- Plotly
- Matplotlib

## Usage
1. Install the required packages using pip.
2. Run the application:
   ```
   streamlit run app.py
   ```
3. Upload your CSV file containing the required columns (e.g., date, meantemp, humidity, wind_speed, meanpressure).
4. Choose the forecast options from the sidebar to generate the desired forecasts.

## File Structure
- `app.py` - Main application file.
- `readme.md` - Project documentation.

<!-- ...additional notes... -->
