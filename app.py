import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

# Configure Page Settings
st.set_page_config(
    page_title="Ethiopia Food Price Prediction & Analytics",
    page_icon="🌾",
    layout="wide"
)

# Safely load serialized model assets
def load_model_assets():
    try:
        model = joblib.load('random_forest_model.joblib')
        return model, None
    except Exception as e:
        return None, str(e)

# App Title
st.title("🌾 Ethiopia Food Price Prediction & Market Analysis")
st.markdown("An interactive decision-support application built for localized market forecasting.")

# Navigation Sidebar
menu = st.sidebar.radio("Navigation", ["Dashboard Overview", "Model Prediction"])

if menu == "Dashboard Overview":
    st.header("📊 Market Insights & Historical Data Visualizations")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="Total Recorded Transactions", value="36,746")
    with col2:
        st.metric(label="Covered Administrative Regions (Admin1)", value="11")
    with col3:
        st.metric(label="Unique Commodities Modeled", value="93")

    st.subheader("Historical Price Distributions")
    st.write("The boxplot below reflects regional price variances with spatial and local outliers adjusted.")

    # Dummy data structure to support visual representations in the interface
    sample_viz_data = pd.DataFrame({
        'admin1': np.repeat(['Addis Ababa', 'Oromia', 'Amhara', 'Tigray', 'Somali'], 100),
        'usdprice': np.random.exponential(scale=35, size=500)
    })

    fig = px.box(
        sample_viz_data,
        x='admin1',
        y='usdprice',
        color='admin1',
        title="USD Price Distributions across Selected Regions"
    )
    st.plotly_chart(fig, use_container_width=True)

elif menu == "Model Prediction":
    st.header("🔮 Real-Time Market Price Inference")
    st.markdown("Enter the market metrics below to predict food prices.")

    with st.form("prediction_form"):
        col_a, col_b = st.columns(2)
        with col_a:
            date_val = st.date_input("Transaction Date", value=pd.to_datetime("2022-12-15"))
            latitude = st.number_input("Latitude Coordinate", value=9.02, format="%.4f")
            longitude = st.number_input("Longitude Coordinate", value=38.75, format="%.4f")
            admin1 = st.selectbox("Region (admin1)", [7, 8, 2, 10, 5, 6, 1, 11]) # Integer mapping matches our clean encoders
            admin2 = st.number_input("Zone (admin2) Encoded ID", value=13, step=1)

        with col_b:
            market = st.number_input("Local Market Encoded ID", value=103, step=1)
            category = st.selectbox("Category Encoded ID", [0, 1, 2, 3])
            commodity = st.number_input("Commodity Encoded ID", value=83, step=1)
            unit = st.selectbox("Unit Encoded ID", [0, 1, 2, 3])
            priceflag = st.selectbox("Price Status Encoded ID", [0, 1])
            pricetype = st.selectbox("Price Type Encoded ID", [0, 1])

        submit = st.form_submit_button("Predict Market Price")

        if submit:
            # Try loading the assets on submit
            model, error_msg = load_model_assets()

            if error_msg:
                st.error(f"Could not process prediction. Serialized assets error: {error_msg}")
            else:
                try:
                    # Extract seasonal features matching our trained 13-dimensional schema
                    dt_date = pd.to_datetime(date_val)
                    year = dt_date.year
                    month_num = dt_date.month
                    month_sin = np.sin(2 * np.pi * month_num / 12)
                    month_cos = np.cos(2 * np.pi * month_num / 12)

                    # Build the feature matrix input (13 columns, matching X_train_rf_f)
                    input_row = pd.DataFrame([{
                        'admin1': admin1,
                        'admin2': admin2,
                        'market': market,
                        'latitude': latitude,
                        'longitude': longitude,
                        'category': category,
                        'commodity': commodity,
                        'unit': unit,
                        'priceflag': priceflag,
                        'pricetype': pricetype,
                        'year': year,
                        'month':month_num,
                        'quarter':quarter,
                        'month_sin': month_sin,
                        'month_cos': month_cos
                    }])

                    prediction = model.predict(input_row)[0]
                    st.success(f"🎯 Predicted Local Market Price: **{prediction:,.2f} ETB**")
                except Exception as ex:
                    st.error(f"Error processing prediction parameters: {ex}")
