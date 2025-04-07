import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from prophet import Prophet

# Streamlit Page Configuration
st.set_page_config(page_title="ForecastMaster", layout="centered")

# Title and Description
#st.image("logo.png", width=100)  # Add a logo at the top
st.title("📈Delhi Climate Forecasting")
st.write("Please upload your CSV file for temperature trend forecasting.")


# Sidebar Options
st.sidebar.markdown("""**Application Features:**  
- Temperature Forecasting  
- Humidity Forecasting  
- Wind Speed Forecasting  
- Mean Pressure Forecasting""")
show_meantemp = st.sidebar.checkbox("Show Temperature Forecast", value=True)  # New checkbox for Temperature Forecast
show_humidity = st.sidebar.checkbox("Show Humidity Forecast", value=True)  # Added new checkbox
show_wind_speed = st.sidebar.checkbox("Show Wind Speed Forecast", value=True)  # existing checkbox
show_meanpressure = st.sidebar.checkbox("Show Mean Pressure Forecast", value=True)  # added new checkbox


# File Upload
uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file:
    # Load Data
    df = pd.read_csv(uploaded_file)
    df.rename(columns={"date": "ds", "meantemp": "y"}, inplace=True)
    df["ds"] = pd.to_datetime(df["ds"])

    # Display Data Preview
    st.write("Data Preview")
    st.dataframe(df.head())

    # Forecast Period Selection
    periods = st.slider("Select number of future days to predict", min_value=30, max_value=365, value=180)

    if show_meantemp:
        # Train Prophet Model for Temperature Forecast
        model = Prophet()
        model.fit(df)

        # Create Future Dataframe and Predict
        future = model.make_future_dataframe(periods=periods)
        forecast = model.predict(future)

        # Display Forecast Graph
        st.write("Forecast Graph")
        fig_forecast = px.line(forecast, x='ds', y='yhat', title='Temperature Forecast', 
                               labels={'yhat': 'Forecast Temperature', 'ds': 'Forecast Date'})
        st.plotly_chart(fig_forecast, use_container_width=True)

        # Show Forecast Data Table
        st.write("Forecast Data Table")
        st.dataframe(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(), 
                     column_config={'ds': 'Forecast Date', 'yhat': 'Forecast Temperature', 'yhat_lower': 'Lower Bound', 'yhat_upper': 'Upper Bound'})

        # Trend and Seasonality Components
        st.write("Trend and Seasonality Components")
        fig_components = model.plot_components(forecast)
        st.pyplot(fig_components)
    
    # New Humidity Forecast Section
    if show_humidity:
        if 'humidity' in df.columns:
            humidity_df = df[["ds", "humidity"]].copy()
            humidity_df.rename(columns={"humidity": "y"}, inplace=True)
            humidity_model = Prophet()
            humidity_model.fit(humidity_df)
            future_humidity = humidity_model.make_future_dataframe(periods=periods)
            humidity_forecast = humidity_model.predict(future_humidity)
            st.write("Humidity Forecast Graph")
            fig_humidity = px.line(humidity_forecast, x='ds', y='yhat', title='Humidity Forecast', 
                                   labels={'yhat': 'Forecasted Humidity', 'ds': 'Forecast Date', 'yhat_lower': 'Lower Bound', 'yhat_upper': 'Upper Bound'})
            st.plotly_chart(fig_humidity, use_container_width=True)
            st.write("Humidity Forecast Data Table")
            st.dataframe(humidity_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(),
                         column_config={'ds': 'Forecast Date', 'yhat': 'Forecasted Humidity',
                                        'yhat_lower': 'Lower-Bound', 'yhat_upper': 'Uper-Bound'})
            st.write("Humidity Trend and Seasonality Components")
            fig_humidity_components = humidity_model.plot_components(humidity_forecast)
            st.pyplot(fig_humidity_components)
        else:
            st.write("Humidity data not found in the uploaded CSV.")
    
    # New Wind Speed Forecast Section
    if show_wind_speed:
        if 'wind_speed' in df.columns:
            wind_df = df[["ds", "wind_speed"]].copy()
            wind_df.rename(columns={"wind_speed": "y"}, inplace=True)
            wind_model = Prophet()
            wind_model.fit(wind_df)
            future_wind = wind_model.make_future_dataframe(periods=periods)
            wind_forecast = wind_model.predict(future_wind)
            st.write("Wind Speed Forecast Graph")
            fig_wind = px.line(wind_forecast, x='ds', y='yhat', title='Wind Speed Forecast', 
                               labels={'yhat': 'Forecasted Wind Speed', 'ds': 'Forecast Date'})
            st.plotly_chart(fig_wind, use_container_width=True)
            st.write("Wind Speed Forecast Data Table")
            st.dataframe(wind_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(), 
                         column_config={'ds': 'Forecast Date', 'yhat': 'Forecasted Wind Speed', 
                                        'yhat_lower': 'Lower-Bound', 'yhat_upper': 'Uper-Bound'})
            st.write("Wind Speed Trend and Seasonality Components")
            fig_wind_components = wind_model.plot_components(wind_forecast)
            st.pyplot(fig_wind_components)
        else:
            st.write("Wind Speed data not found in the uploaded CSV.")

    # New Mean Pressure Forecast Section
    if show_meanpressure:
        if 'meanpressure' in df.columns:
            mp_df = df[["ds", "meanpressure"]].copy()
            mp_df.rename(columns={"meanpressure": "y"}, inplace=True)
            mp_model = Prophet()
            mp_model.fit(mp_df)
            future_mp = mp_model.make_future_dataframe(periods=periods)
            mp_forecast = mp_model.predict(future_mp)
            st.write("Mean Pressure Forecast Graph")
            fig_mp = px.line(mp_forecast, x='ds', y='yhat', title='Mean Pressure Forecast', 
                             labels={'yhat': 'Forecasted Mean Pressure', 'ds': 'Forecast Date'})
            st.plotly_chart(fig_mp, use_container_width=True)
            st.write("Mean Pressure Forecast Data Table")
            st.dataframe(mp_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(), 
                         column_config={'ds': 'Forecast Date', 'yhat': 'Forecasted Mean Pressure', 
                                        'yhat_lower': 'Lower-Bound', 'yhat_upper': 'Uper-Bound'})
            st.write("Mean Pressure Trend and Seasonality Components")
            fig_mp_components = mp_model.plot_components(mp_forecast)
            st.pyplot(fig_mp_components)
        else:
            st.write("Mean Pressure data not found in the uploaded CSV.")
