```python
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI AQI Prediction",
    page_icon="🌍",
    layout="wide"
)

# ============================================================
# LOAD MODELS
# ============================================================

@st.cache_resource
def load_models():
    model = joblib.load("aqi_prediction_model.pkl")
    forecast_model = joblib.load("aqi_forecasting_model.pkl")
    return model, forecast_model


# ============================================================
# LOAD DATASET
# ============================================================

@st.cache_data
def load_data():
    df = pd.read_csv("AQI_ML_Dataset.csv")
    df["AQI"] = df["AQI"].fillna(df["AQI"].median())
    return df


model, forecast_model = load_models()
df = load_data()


# ============================================================
# AQI CATEGORY
# ============================================================

def get_category(aqi):

    if aqi <= 50:
        return "Good"

    elif aqi <= 100:
        return "Satisfactory"

    elif aqi <= 200:
        return "Moderate"

    elif aqi <= 300:
        return "Poor"

    elif aqi <= 400:
        return "Very Poor"

    else:
        return "Severe"


# ============================================================
# HEALTH ADVISORY
# ============================================================

def get_advisory(aqi):

    if aqi <= 50:
        return "Air quality is good. Normal outdoor activities are generally safe."

    elif aqi <= 100:
        return "Air quality is satisfactory. Sensitive people should monitor conditions."

    elif aqi <= 200:
        return "Sensitive individuals should reduce prolonged outdoor exposure."

    elif aqi <= 300:
        return "Reduce prolonged outdoor activities."

    elif aqi <= 400:
        return "Avoid prolonged outdoor exposure."

    else:
        return "Avoid outdoor exposure as much as possible."


# ============================================================
# TITLE
# ============================================================

st.title("🌍 AI Environmental Intelligence System")

st.subheader(
    "AI-Based Air Quality Index Prediction and Forecasting"
)

st.write(
    "Enter environmental conditions to predict AQI and "
    "analyze future air quality."
)

st.divider()


# ============================================================
# SIDEBAR INPUT
# ============================================================

st.sidebar.header("📍 Location")

location = st.sidebar.text_input(
    "Area / City",
    value="Perundurai"
)

st.sidebar.header("🌫️ Environmental Parameters")

pm25 = st.sidebar.number_input(
    "PM2.5",
    min_value=0.0,
    max_value=1000.0,
    value=80.0,
    step=1.0
)

pm10 = st.sidebar.number_input(
    "PM10",
    min_value=0.0,
    max_value=1000.0,
    value=120.0,
    step=1.0
)

so2 = st.sidebar.number_input(
    "SO2",
    min_value=0.0,
    max_value=1000.0,
    value=15.0,
    step=1.0
)

no2 = st.sidebar.number_input(
    "NO2",
    min_value=0.0,
    max_value=1000.0,
    value=40.0,
    step=1.0
)

co = st.sidebar.number_input(
    "CO",
    min_value=0.0,
    max_value=100.0,
    value=1.2,
    step=0.1
)

o3 = st.sidebar.number_input(
    "O3",
    min_value=0.0,
    max_value=1000.0,
    value=60.0,
    step=1.0
)

temp = st.sidebar.number_input(
    "Temperature",
    min_value=-50.0,
    max_value=60.0,
    value=25.0,
    step=0.1
)

pres = st.sidebar.number_input(
    "Pressure",
    min_value=800.0,
    max_value=1200.0,
    value=1010.0,
    step=0.1
)

dewp = st.sidebar.number_input(
    "Dew Point",
    min_value=-50.0,
    max_value=60.0,
    value=15.0,
    step=0.1
)

rain = st.sidebar.number_input(
    "Rainfall",
    min_value=0.0,
    max_value=500.0,
    value=0.0,
    step=0.1
)

wspm = st.sidebar.number_input(
    "Wind Speed",
    min_value=0.0,
    max_value=50.0,
    value=2.0,
    step=0.1
)

# ============================================================
# PREDICT BUTTON
# ============================================================

predict_button = st.sidebar.button(
    "🔮 Predict AQI",
    type="primary",
    use_container_width=True
)


# ============================================================
# INPUT DATA
# ============================================================

input_data = pd.DataFrame(
    [[
        pm25,
        pm10,
        so2,
        no2,
        co,
        o3,
        temp,
        pres,
        dewp,
        rain,
        wspm
    ]],
    columns=[
        "PM2.5",
        "PM10",
        "SO2",
        "NO2",
        "CO",
        "O3",
        "TEMP",
        "PRES",
        "DEWP",
        "RAIN",
        "WSPM"
    ]
)


# ============================================================
# FUTURE FORECAST
# ============================================================

aqi = df["AQI"]

future_input = pd.DataFrame([{
    "AQI_Lag_1": aqi.iloc[-1],
    "AQI_Lag_2": aqi.iloc[-2],
    "AQI_Lag_3": aqi.iloc[-3],
    "AQI_Lag_6": aqi.iloc[-6],
    "AQI_Lag_12": aqi.iloc[-12],
    "AQI_Lag_24": aqi.iloc[-24]
}])

future_aqi = forecast_model.predict(
    future_input
)[0]

future_category = get_category(future_aqi)


# ============================================================
# DEFAULT VALUES BEFORE PREDICTION
# ============================================================

if "predicted_aqi" not in st.session_state:
    st.session_state.predicted_aqi = None


# ============================================================
# RUN PREDICTION
# ============================================================

if predict_button:

    predicted_aqi = model.predict(input_data)[0]

    st.session_state.predicted_aqi = predicted_aqi


# ============================================================
# DASHBOARD
# ============================================================

st.header(
    f"📍 Air Quality — {location}"
)


# ============================================================
# SHOW RESULT
# ============================================================

if st.session_state.predicted_aqi is not None:

    current_aqi = st.session_state.predicted_aqi

    category = get_category(current_aqi)

    advisory = get_advisory(current_aqi)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Current AQI",
            f"{current_aqi:.2f}"
        )

    with col2:
        st.metric(
            "AQI Category",
            category
        )

    with col3:
        st.metric(
            "Future AQI",
            f"{future_aqi:.2f}"
        )

    st.info(
        f"💡 Health Advisory: {advisory}"
    )

else:

    st.info(
        "👈 Enter or change the environmental values "
        "and click **🔮 Predict AQI** to get the prediction."
    )


# ============================================================
# POLLUTION TABLE
# ============================================================

st.subheader("🌫️ Environmental Parameters")

pollution_data = pd.DataFrame({

    "Parameter": [
        "PM2.5",
        "PM10",
        "SO2",
        "NO2",
        "CO",
        "O3",
        "Temperature",
        "Pressure",
        "Dew Point",
        "Rainfall",
        "Wind Speed"
    ],

    "Value": [
        pm25,
        pm10,
        so2,
        no2,
        co,
        o3,
        temp,
        pres,
        dewp,
        rain,
        wspm
    ]

})

st.dataframe(
    pollution_data,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# POLLUTION GRAPH
# ============================================================

st.subheader("📊 Pollution Analysis")

pollutants = {
    "PM2.5": pm25,
    "PM10": pm10,
    "SO2": so2,
    "NO2": no2,
    "CO": co,
    "O3": o3
}

fig, ax = plt.subplots(
    figsize=(10, 5)
)

ax.bar(
    pollutants.keys(),
    pollutants.values()
)

ax.set_xlabel("Pollutant")
ax.set_ylabel("Value")
ax.set_title("Pollution Levels")

ax.grid(
    axis="y",
    alpha=0.3
)

st.pyplot(fig)


# ============================================================
# HISTORICAL AQI
# ============================================================

st.subheader("📈 Historical AQI Trend")

recent_aqi = df["AQI"].iloc[-500:]

fig2, ax2 = plt.subplots(
    figsize=(12, 5)
)

ax2.plot(
    recent_aqi.values
)

ax2.set_xlabel("Observation")
ax2.set_ylabel("AQI")
ax2.set_title("Historical AQI Trend")

ax2.grid(
    alpha=0.3
)

st.pyplot(fig2)


# ============================================================
# FUTURE AQI
# ============================================================

st.subheader("🔮 Future AQI Forecast")

col4, col5 = st.columns(2)

with col4:

    st.metric(
        "Predicted Future AQI",
        f"{future_aqi:.2f}"
    )

with col5:

    st.write(
        f"Future AQI Category: **{future_category}**"
    )


# ============================================================
# PROJECT INFORMATION
# ============================================================

st.divider()

st.subheader("🤖 Project Information")

st.write(
    "Machine Learning Algorithm: Random Forest Regression"
)

st.write(
    "Dataset Records: 35,046"
)

st.write(
    "Input Features: 11 environmental parameters"
)

st.write(
    "Target Variable: AQI"
)

st.write(
    "Platform: Streamlit"
)

st.caption(
    "AI-Based Air Quality Prediction and Forecasting System"
)
```
