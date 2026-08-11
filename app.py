
import streamlit as st
import pandas as pd
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

    model = joblib.load(
        "aqi_prediction_model.pkl"
    )

    forecast_model = joblib.load(
        "aqi_forecasting_model.pkl"
    )

    return model, forecast_model


# ============================================================
# LOAD DATASET
# ============================================================

@st.cache_data
def load_data():

    df = pd.read_csv(
        "AQI_ML_Dataset.csv"
    )

    df["AQI"] = pd.to_numeric(
        df["AQI"],
        errors="coerce"
    )

    df["AQI"] = df["AQI"].fillna(
        df["AQI"].median()
    )

    return df


# Load models and dataset

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

        return (
            "Air quality is good. "
            "Normal outdoor activities are generally safe."
        )

    elif aqi <= 100:

        return (
            "Air quality is satisfactory. "
            "Sensitive people should monitor conditions."
        )

    elif aqi <= 200:

        return (
            "Sensitive individuals should "
            "reduce prolonged outdoor exposure."
        )

    elif aqi <= 300:

        return (
            "Reduce prolonged outdoor activities."
        )

    elif aqi <= 400:

        return (
            "Avoid prolonged outdoor exposure."
        )

    else:

        return (
            "Avoid outdoor exposure as much as possible."
        )


# ============================================================
# TITLE
# ============================================================

st.title(
    "🌍 AI Environmental Intelligence System"
)

st.subheader(
    "AI-Based Air Quality Index Prediction and Forecasting"
)

st.write(
    "Enter environmental conditions below and click "
    "**🔮 Predict AQI** to calculate the results."
)

st.divider()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header(
    "📍 Location"
)

location = st.sidebar.text_input(
    "Area / City",
    value="Perundurai"
)


st.sidebar.header(
    "🌫️ Environmental Parameters"
)


# ============================================================
# USER INPUTS
# ============================================================

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
# CREATE MODEL INPUT
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
# PREDICTION
# ============================================================

if predict_button:

    try:

        # ----------------------------------------------------
        # CURRENT AQI
        # ----------------------------------------------------

        current_aqi = float(
            model.predict(
                input_data
            )[0]
        )

        current_aqi = max(
            0.0,
            current_aqi
        )


        # ----------------------------------------------------
        # CURRENT CATEGORY
        # ----------------------------------------------------

        current_category = get_category(
            current_aqi
        )


        # ----------------------------------------------------
        # HEALTH ADVISORY
        # ----------------------------------------------------

        advisory = get_advisory(
            current_aqi
        )


        # ----------------------------------------------------
        # FUTURE FORECAST INPUT
        # ----------------------------------------------------

        aqi = df["AQI"]

        future_input = pd.DataFrame(
            [[
                current_aqi,

                float(aqi.iloc[-1]),

                float(aqi.iloc[-2]),

                float(aqi.iloc[-5]),

                float(aqi.iloc[-11]),

                float(aqi.iloc[-23])
            ]],
            columns=[
                "AQI_Lag_1",
                "AQI_Lag_2",
                "AQI_Lag_3",
                "AQI_Lag_6",
                "AQI_Lag_12",
                "AQI_Lag_24"
            ]
        )


        # ----------------------------------------------------
        # FUTURE AQI
        # ----------------------------------------------------

        future_aqi = float(
            forecast_model.predict(
                future_input
            )[0]
        )

        future_aqi = max(
            0.0,
            future_aqi
        )


        # ----------------------------------------------------
        # FUTURE CATEGORY
        # ----------------------------------------------------

        future_category = get_category(
            future_aqi
        )


        # ----------------------------------------------------
        # SAVE EVERYTHING
        # ----------------------------------------------------

        st.session_state["current_aqi"] = current_aqi

        st.session_state["current_category"] = (
            current_category
        )

        st.session_state["advisory"] = advisory

        st.session_state["future_aqi"] = future_aqi

        st.session_state["future_category"] = (
            future_category
        )


        # Save input values for graph

        st.session_state["pm25"] = pm25
        st.session_state["pm10"] = pm10
        st.session_state["so2"] = so2
        st.session_state["no2"] = no2
        st.session_state["co"] = co
        st.session_state["o3"] = o3


        st.success(
            "✅ Prediction updated successfully!"
        )


    except Exception as e:

        st.error(
            f"Prediction error: {e}"
        )


# ============================================================
# DASHBOARD
# ============================================================

st.header(
    f"📍 Air Quality — {location}"
)


# ============================================================
# SHOW RESULTS
# ============================================================

if "current_aqi" in st.session_state:

    current_aqi = st.session_state[
        "current_aqi"
    ]

    current_category = st.session_state[
        "current_category"
    ]

    advisory = st.session_state[
        "advisory"
    ]

    future_aqi = st.session_state[
        "future_aqi"
    ]

    future_category = st.session_state[
        "future_category"
    ]


    # ========================================================
    # RESULT CARDS
    # ========================================================

    col1, col2, col3 = st.columns(3)


    with col1:

        st.metric(
            "🌫️ Current AQI",
            f"{current_aqi:.2f}"
        )


    with col2:

        st.metric(
            "📊 AQI Category",
            current_category
        )


    with col3:

        st.metric(
            "🔮 Future AQI",
            f"{future_aqi:.2f}"
        )


    # ========================================================
    # CATEGORY
    # ========================================================

    st.success(
        f"✅ Current AQI: **{current_aqi:.2f}**"
    )

    st.info(
        f"📊 AQI Category: **{current_category}**"
    )

    st.warning(
        f"💡 Health Advisory: {advisory}"
    )

    st.info(
        f"🔮 Future AQI: **{future_aqi:.2f}** "
        f"| Future Category: **{future_category}**"
    )


else:

    st.info(
        "👈 Enter your environmental values "
        "and click **🔮 Predict AQI**."
    )


# ============================================================
# CURRENT INPUT TABLE
# ============================================================

st.subheader(
    "🌫️ Current Environmental Parameters"
)


current_values = {

    "PM2.5": pm25,

    "PM10": pm10,

    "SO2": so2,

    "NO2": no2,

    "CO": co,

    "O3": o3,

    "Temperature": temp,

    "Pressure": pres,

    "Dew Point": dewp,

    "Rainfall": rain,

    "Wind Speed": wspm
}


pollution_data = pd.DataFrame({

    "Parameter": list(
        current_values.keys()
    ),

    "Value": list(
        current_values.values()
    )

})


st.dataframe(
    pollution_data,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# POLLUTION GRAPH
# ============================================================

st.subheader(
    "📊 Current Pollution Levels"
)


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


bars = ax.bar(
    list(pollutants.keys()),
    list(pollutants.values())
)


ax.set_xlabel(
    "Pollutant"
)

ax.set_ylabel(
    "Value"
)

ax.set_title(
    "User Entered Pollution Values"
)


ax.grid(
    axis="y",
    alpha=0.3
)


# Display values on bars

for bar in bars:

    height = bar.get_height()

    ax.text(
        bar.get_x()
        + bar.get_width() / 2,

        height,

        f"{height:.1f}",

        ha="center",

        va="bottom"
    )


st.pyplot(
    fig,
    clear_figure=True
)


# ============================================================
# AQI CURRENT VS FUTURE GRAPH
# ============================================================

if "current_aqi" in st.session_state:

    st.subheader(
        "📈 Current AQI vs Future AQI"
    )


    aqi_labels = [
        "Current AQI",
        "Future AQI"
    ]


    aqi_values = [

        st.session_state[
            "current_aqi"
        ],

        st.session_state[
            "future_aqi"
        ]

    ]


    fig3, ax3 = plt.subplots(
        figsize=(10, 5)
    )


    ax3.plot(
        aqi_labels,
        aqi_values,
        marker="o",
        linewidth=3,
        markersize=10
    )


    ax3.set_xlabel(
        "Prediction"
    )

    ax3.set_ylabel(
        "AQI"
    )

    ax3.set_title(
        "Current AQI vs Future AQI"
    )


    ax3.grid(
        axis="y",
        alpha=0.3
    )


    # Display values

    for i, value in enumerate(
        aqi_values
    ):

        ax3.text(
            i,
            value,
            f"{value:.2f}",
            ha="center",
            va="bottom"
        )


    st.pyplot(
        fig3,
        clear_figure=True
    )


# ============================================================
# HISTORICAL AQI GRAPH
# ============================================================

st.subheader(
    "📈 Historical AQI Trend"
)


recent_aqi = df[
    "AQI"
].iloc[-500:]


fig2, ax2 = plt.subplots(
    figsize=(12, 5)
)


ax2.plot(
    recent_aqi.values
)


ax2.set_xlabel(
    "Historical Observation"
)

ax2.set_ylabel(
    "AQI"
)

ax2.set_title(
    "Historical AQI Trend"
)


ax2.grid(
    alpha=0.3
)


st.pyplot(
    fig2,
    clear_figure=True
)


# ============================================================
# FUTURE FORECAST
# ============================================================

st.subheader(
    "🔮 Future AQI Forecast"
)


if "future_aqi" in st.session_state:

    col4, col5 = st.columns(2)


    with col4:

        st.metric(
            "Predicted Future AQI",

            f"{st.session_state['future_aqi']:.2f}"
        )


    with col5:

        st.metric(
            "Future AQI Category",

            st.session_state[
                "future_category"
            ]
        )


else:

    st.info(
        "Future AQI will appear after "
        "clicking Predict AQI."
    )


# ============================================================
# AQI CATEGORY GUIDE
# ============================================================

st.subheader(
    "📋 AQI Category Guide"
)


category_data = pd.DataFrame({

    "AQI Range": [
        "0 – 50",
        "51 – 100",
        "101 – 200",
        "201 – 300",
        "301 – 400",
        "401+"
    ],

    "Category": [
        "Good",
        "Satisfactory",
        "Moderate",
        "Poor",
        "Very Poor",
        "Severe"
    ]

})


st.dataframe(
    category_data,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# PROJECT INFORMATION
# ============================================================

st.divider()

st.subheader(
    "🤖 Project Information"
)

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


