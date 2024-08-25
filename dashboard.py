import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import plotly.express as px
import folium
from streamlit_folium import folium_static

# Load the dataset
@st.cache_data
def load_data():
    return pd.read_csv('data_0.csv')

# Main function to run the dashboard
def main():
    st.title("Rainfall and Flood Prediction Dashboard")

    df = load_and_preprocess_data()

    # Shared inputs for both sections
    st.sidebar.header("Prediction Settings")
    
    # Feature Selector
    features = st.sidebar.multiselect("Select Features", options=df.columns.tolist(), default=['latitude', 'longitude', 't2m', 'swvl1', 'v10', 'sp'])

    # Model Parameters
    n_estimators = st.sidebar.slider("Number of Trees in Random Forest", min_value=10, max_value=200, value=100, step=10, key="n_estimators_slider")

    # Number of Days to Predict
    days_to_predict = st.sidebar.slider("Number of Days to Predict", min_value=1, max_value=365, value=3, step=1, key="days_to_predict_slider")

    # Tabs for different predictions
    tab1, tab2 = st.tabs(["🔍 Rainfall Prediction", "🌊 Flood Prediction"])

    with tab1:
        rainfall_prediction(df, features, n_estimators, days_to_predict)
    
    with tab2:
        flood_prediction(df, features, n_estimators, days_to_predict)

# Load and preprocess the data
def load_and_preprocess_data():
    df = load_data()

    # Handle missing values
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    imputer = SimpleImputer(strategy='mean')
    df[numerical_cols] = imputer.fit_transform(df[numerical_cols])

    return df

# Train the model and predict the next days
def predict_next_days(df_train, df_predict, features, n_estimators):
    target = 'tp'  # Rainfall column

    X_train = df_train[features]
    y_train = df_train[target]

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    X_predict = df_predict[features]
    X_predict_scaled = scaler.transform(X_predict)

    model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_predict_scaled)

    return y_pred

# Rainfall Prediction
def rainfall_prediction(df, features, n_estimators, days_to_predict):

    # Prepare Data for Prediction
    df_train = df.iloc[:-days_to_predict]
    df_predict = df.iloc[-days_to_predict:]

    # Train and predict with the Random Forest model
    y_pred = predict_next_days(df_train, df_predict, features, n_estimators)

    # Assign predictions to the prediction DataFrame
    df_predict['predicted_tp'] = y_pred

    # Create a "days" column (1, 2, 3, ...) for the x-axis
    df_predict['days'] = range(1, days_to_predict + 1)

    # Visualize the results
    visualize_rainfall_predictions(df_predict, days_to_predict)

# Visualize the rainfall predictions
def visualize_rainfall_predictions(df_predict, days_to_predict):
    st.subheader("Rainfall Prediction Over the Next Days")

    # Line plot with "days" as the x-axis
    fig = px.line(df_predict, x='days', y='predicted_tp',
                  title=f"Predicted Rainfall Over the Next {days_to_predict} Days",
                  labels={'predicted_tp': 'Predicted Rainfall', 'days': 'Upcoming Days'},
                  color_discrete_sequence=["orange"])
    st.plotly_chart(fig, use_container_width=True)

# Flood Prediction based on predicted rainfall
def flood_prediction(df, features, n_estimators, days_to_predict):

    # Prepare Data for Prediction
    df_train = df.iloc[:-days_to_predict]
    df_predict = df.iloc[-days_to_predict:]

    # Train and predict rainfall with the Random Forest model
    y_pred = predict_next_days(df_train, df_predict, features, n_estimators)

    # Calculate total predicted rainfall over the period
    total_predicted_rainfall = np.sum(y_pred)
    
    # Simple threshold logic for flood risk prediction
    flood_risk = 'High' if total_predicted_rainfall > 15 else 'Low'  # Example threshold value

    st.subheader(f"Flood Risk Prediction: {flood_risk}")
    st.write(f"Total Predicted Rainfall Over {days_to_predict} Days: {total_predicted_rainfall:.2f} mm")

    # Visualize flood risk on a map
    visualize_flood_risk(df_predict, y_pred, flood_risk)

# Visualize flood risk on a map
def visualize_flood_risk(df_predict, y_pred, flood_risk):
    st.subheader("Flood Risk Visualization")

    # Initialize a folium map centered on the region of interest
    m = folium.Map(location=[df_predict['latitude'].mean(), df_predict['longitude'].mean()], zoom_start=12)

    # Add flood risk markers
    for _, row in df_predict.iterrows():
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=5,
            color='red' if flood_risk == 'High' else 'green',
            fill=True,
            fill_color='red' if flood_risk == 'High' else 'green',
            fill_opacity=0.6
        ).add_to(m)

    # Display the map in Streamlit
    folium_static(m)

# Run the app
if __name__ == "__main__":
    main()
