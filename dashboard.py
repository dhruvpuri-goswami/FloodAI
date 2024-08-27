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
import random
from sklearn.preprocessing import StandardScaler
import ee
import rasterio
from pyproj import Transformer
import osmnx as ox
from shapely.geometry import Point
from sklearn.cluster import KMeans
import datetime
from geopy.distance import geodesic
import requests

# Authenticate and initialize Earth Engine
ee.Authenticate()
ee.Initialize(project='ee-mrdhruvpurigoswami')  # Replace with your actual project ID

# Load the dataset
@st.cache_data
def load_data():
    return pd.read_csv('data_0.csv')

def get_prediction_days(user_input_days):
    """
    Determines the number of days to predict based on the current month.
    
    If the current month is July, August, September, October, November, or December,
    the function will use the user input. Otherwise, it will set the number of days to 0.
    
    Args:
    - user_input_days (int): The number of days to predict provided by the user.
    
    Returns:
    - int: The actual number of days to predict.
    """
    # Get the current date and month
    current_date = datetime.datetime.now()
    current_month = current_date.month

    # Define the months during which predictions should be allowed
    active_months = [7, 8, 9, 10, 11, 12]

    # Check if the current month is within the active months
    if current_month in active_months:
        return user_input_days
    else:
        return 0

# Main function to run the dashboard
def main():
    st.title("FloodAI Dashboard")

    df = load_and_preprocess_data()

    # Shared inputs for both sections
    st.sidebar.header("Prediction Settings")
    
    # Feature Selector
    features = st.sidebar.multiselect("Select Features", options=df.columns.tolist(), default=['latitude', 'longitude', 't2m', 'swvl1', 'v10', 'sp'])

    # Model Parameters
    n_estimators = st.sidebar.slider("Number of Trees in Random Forest", min_value=10, max_value=200, value=100, step=10, key="n_estimators_slider")

    # Number of Days to Predict - User input
    user_input_days = st.sidebar.slider("Number of Days to Predict", min_value=1, max_value=365, value=3, step=1, key="days_to_predict_slider")

    # Determine the actual number of days to predict based on the current month
    actual_prediction_days = get_prediction_days(user_input_days)

    # Tabs for different predictions
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔍 Rainfall Prediction", 
        "🌊 Flood Prediction", 
        "🛰️ Find Safe Zone", 
        "🚑 Route to Safety", 
        "📊 Insurance Planning"
    ])

    with tab1:
        rainfall_prediction(df, features, n_estimators, actual_prediction_days)

    with tab2:
        flood_prediction_model(df, features, n_estimators, actual_prediction_days)

    with tab3:
        elevation_analysis(df, features, n_estimators, actual_prediction_days)

    with tab4:
        route_to_safety()

    with tab5:
        risk_clustering()


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

    # Add random noise between 0 and 1 to each prediction
    y_pred_with_noise = y_pred + np.random.uniform(0, 1, size=y_pred.shape)

    return y_pred_with_noise

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
def flood_prediction_model(df, features, n_estimators, days_to_predict):
    st.subheader("Flood Prediction Model for Surat")

    # Define the Area of Interest (AOI) for Surat, Gujarat
    surat_boundary = ox.geocode_to_gdf("Surat, Gujarat, India")

    # Predict rainfall data for the next specified days
    df_train = df.iloc[:-days_to_predict]
    df_predict = df.iloc[-days_to_predict:]
    
    # Predict rainfall using the existing predict_next_days function
    y_pred = predict_next_days(df_train, df_predict, features, n_estimators)

    # Create a DataFrame to store the latitude and longitude along with predicted rainfall
    df_flood_prediction = pd.DataFrame({
        'Latitude': df_predict['latitude'],
        'Longitude': df_predict['longitude'],
        'Predicted_Rainfall': y_pred
    })

    # Define a threshold for flooding (this can be adjusted based on the data)
    flood_threshold = 15  # Example threshold in mm

    # Filter for flood-predicted areas
    df_flood_prediction = df_flood_prediction[df_flood_prediction['Predicted_Rainfall'] > flood_threshold]

    # Ensure all predicted locations are within the Surat boundary
    df_flood_prediction = df_flood_prediction[df_flood_prediction.apply(
        lambda row: surat_boundary.geometry.iloc[0].contains(Point(row['Longitude'], row['Latitude'])), axis=1)]

    # Create a Folium map centered on Surat using the OSM boundary
    map_surat = folium.Map(location=[21.170240, 72.831062], zoom_start=13)

    # Add the Surat boundary to the map
    folium.GeoJson(surat_boundary.to_json(), name="Surat Boundary").add_to(map_surat)

    # Mark flood-predicted areas with distinct markers
    for _, row in df_flood_prediction.iterrows():
        folium.Marker(
            location=(row['Latitude'], row['Longitude']),
            icon=folium.Icon(color='red', icon='info-sign'),
            popup=f"Predicted Rainfall: {row['Predicted_Rainfall']} mm<br>Potential Flood Area"
        ).add_to(map_surat)

    # Display the map in Streamlit
    folium_static(map_surat)

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

def elevation_analysis(df, features, n_estimators, days_to_predict):
    st.subheader("Elevation and Flood Risk Analysis for Surat")

    # Define the Area of Interest (AOI) for Surat, Gujarat
    surat_boundary = ox.geocode_to_gdf("Surat, Gujarat, India")

    # Predict rainfall data for the next two years (e.g., 365*2 = 730 days)
    df_train = df.iloc[:-730]
    df_predict = df.iloc[-730:]
    
    # Predict rainfall using the existing predict_next_days function
    y_pred = predict_next_days(df_train, df_predict, features, n_estimators)

    # Create a transformer to convert WGS84 to Mercator
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3395", always_xy=True)

    # Create a grid of points within the Surat boundary
    num_points = 5000
    points = []

    minx, miny, maxx, maxy = surat_boundary.total_bounds
    lons = np.linspace(minx, maxx, int(np.sqrt(num_points)))
    lats = np.linspace(miny, maxy, int(np.sqrt(num_points)))

    for lon in lons:
        for lat in lats:
            point = Point(lon, lat)
            if surat_boundary.geometry.iloc[0].contains(point):
                points.append((lon, lat))

    # Open the TIFF file and extract elevation data
    elevations = []
    tiff_file = 'dem_india.tif'

    with rasterio.open(tiff_file) as src:
        image_data = src.read(1)

        for lon, lat in points:
            x, y = transformer.transform(lon, lat)
            row, col = src.index(x, y)
            if 0 <= row < src.height and 0 <= col < src.width:
                elevation = image_data[row, col]
                elevations.append((lon, lat, elevation))
            else:
                elevations.append((lon, lat, None))

    df_elevation = pd.DataFrame(elevations, columns=['Longitude', 'Latitude', 'Elevation'])

    # Define classification thresholds for elevation
    elevation_high = np.percentile(df_elevation['Elevation'].dropna(), 75)
    elevation_low = np.percentile(df_elevation['Elevation'].dropna(), 25)

    colors = {
        'Safe Zone': 'green',
        'Moderate Zone': 'yellow',
        'Danger Zone': 'red'
    }

    # Ensure all predicted locations are within the boundary
    df_elevation = df_elevation[df_elevation.apply(lambda row: surat_boundary.geometry.iloc[0].contains(Point(row['Longitude'], row['Latitude'])), axis=1)]

    # Cluster elevation data points using KMeans
    kmeans = KMeans(n_clusters=3, random_state=42).fit(df_elevation[['Longitude', 'Latitude', 'Elevation']].dropna())
    df_elevation['Cluster'] = kmeans.labels_

    # Mark the zones based on elevation
    flood_prone_set = set((round(lat, 4), round(lon, 4)) for lat, lon in zip(df_predict['latitude'], df_predict['longitude']))

    def classify_risk(elevation, lat, lon):
        point = (round(lat, 4), round(lon, 4))
        if point in flood_prone_set:
            return 'Danger Zone'
        elif elevation >= elevation_high:
            return 'Safe Zone'
        elif elevation <= elevation_low:
            return 'Danger Zone'
        else:
            return 'Moderate Zone'

    df_elevation['Zone'] = df_elevation.apply(lambda row: classify_risk(row['Elevation'], row['Latitude'], row['Longitude']), axis=1)

    # Reduce the number of flood markers by sampling
    df_danger_zone = df_elevation[df_elevation['Zone'] == 'Danger Zone'].sample(frac=0.1)  # Adjust fraction as needed

    # Create a Folium map centered on Surat using the OSM boundary
    map_surat = folium.Map(location=[21.170240, 72.831062], zoom_start=13)

    # Add the Surat boundary to the map
    folium.GeoJson(surat_boundary.to_json(), name="Surat Boundary").add_to(map_surat)

    # Add elevation clusters to the map with colors based on classified zones
    for _, row in df_elevation.iterrows():
        if row['Elevation'] is not None:
            color = colors[row['Zone']]
            folium.CircleMarker(
                location=(row['Latitude'], row['Longitude']),
                radius=3,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.6,
                popup=f"Elevation: {row['Elevation']} meters<br>Zone: {row['Zone']}<br>Cluster: {row['Cluster']}"
            ).add_to(map_surat)

    # Pinpoint flood-prone areas with distinct markers, but reduce the number
    for _, row in df_danger_zone.iterrows():
        folium.Marker(
            location=(row['Latitude'], row['Longitude']),
            icon=folium.Icon(color='red', icon='info-sign'),
            popup="High Probability Flood Area"
        ).add_to(map_surat)

    # Add a layer control panel to the map
    folium.LayerControl().add_to(map_surat)

    # Display the map in Streamlit
    folium_static(map_surat)

def route_to_safety():
    st.subheader("Optimal Route to Safety")

    # Define the Area of Interest (AOI) for Surat, Gujarat
    surat_boundary = ox.geocode_to_gdf("Surat, Gujarat, India")

    # Define the OSRM public server URL
    osrm_url = "http://router.project-osrm.org/route/v1/driving/"

    # Example multiple flooded locations (longitude, latitude)
    flooded_locations = [
        (72.8311, 21.1702),
        (72.8258, 21.1852),
        (72.7805, 21.1475)
    ]

    # Example shelters and hospitals (longitude, latitude)
    shelters = [
        (72.8281, 21.1957),  # Shelter 1
        (72.8395, 21.1739),  # Shelter 2
        (72.7857, 21.1456)   # Shelter 3
    ]

    hospitals = [
        (72.8327, 21.1915),  # Hospital 1
        (72.8512, 21.1631),  # Hospital 2
        (72.7729, 21.1486)   # Hospital 3
    ]

    # Function to check if a point is within a flooded area
    def is_within_flooded_area(location, flooded_locations, radius=500):
        for flood_lat, flood_lon in flooded_locations:
            if geodesic(location[::-1], (flood_lon, flood_lat)).meters <= radius:
                return True
        return False

    # Function to calculate the optimal route using the public OSRM server
    def calculate_optimal_route(start, end):
        coordinates = f"{start[0]},{start[1]};{end[0]},{end[1]}"
        response = requests.get(f"{osrm_url}{coordinates}?overview=full&geometries=geojson")
        data = response.json()
        if 'routes' in data:
            return data['routes'][0]['geometry']['coordinates'], data['routes'][0]['duration'], data['routes'][0]['distance']
        return None, None, None

    # Initialize a Folium map centered on the first flooded location
    map_surat = folium.Map(location=flooded_locations[0][::-1], zoom_start=13)

    # Add the Surat boundary to the map
    folium.GeoJson(surat_boundary.to_json(), name="Surat Boundary").add_to(map_surat)

    # Highlight the flooded areas with circles
    for lat, lon in flooded_locations:
        folium.Circle(
            location=(lon, lat),
            radius=500,  # Radius in meters (adjust as needed)
            color='red',
            fill=True,
            fill_color='red',
            fill_opacity=0.4,
            popup="Flooded Area"
        ).add_to(map_surat)

    # Find and plot the shortest risk-free path to the nearest safe shelter and hospital
    for flooded_location in flooded_locations:
        shortest_shelter_duration = float('inf')
        best_shelter_route_coords = []

        shortest_hospital_duration = float('inf')
        best_hospital_route_coords = []

        best_shelter_coords = None
        best_hospital_coords = None

        # Check routes to all shelters
        for shelter in shelters:
            if not is_within_flooded_area(shelter, flooded_locations):
                route_coords, duration, distance = calculate_optimal_route(flooded_location, shelter)
                if duration and duration < shortest_shelter_duration:
                    shortest_shelter_duration = duration
                    best_shelter_route_coords = route_coords
                    best_shelter_coords = shelter

        # Check routes to all hospitals
        for hospital in hospitals:
            if not is_within_flooded_area(hospital, flooded_locations):
                route_coords, duration, distance = calculate_optimal_route(flooded_location, hospital)
                if duration and duration < shortest_hospital_duration:
                    shortest_hospital_duration = duration
                    best_hospital_route_coords = route_coords
                    best_hospital_coords = hospital

        # Plot the best shelter route
        if best_shelter_route_coords:
            folium.PolyLine(
                locations=[(coord[1], coord[0]) for coord in best_shelter_route_coords],
                color="blue",
                weight=2.5,
                opacity=1,
                popup=f"Route to Nearest Shelter (Duration: {shortest_shelter_duration/60:.2f} mins)"
            ).add_to(map_surat)
            # Mark the best shelter
            folium.Marker(location=best_shelter_coords[::-1], popup="Nearest Safe Shelter", icon=folium.Icon(color='green', icon='home')).add_to(map_surat)

        # Plot the best hospital route
        if best_hospital_route_coords:
            folium.PolyLine(
                locations=[(coord[1], coord[0]) for coord in best_hospital_route_coords],
                color="purple",
                weight=2.5,
                opacity=1,
                popup=f"Route to Nearest Hospital (Duration: {shortest_hospital_duration/60:.2f} mins)"
            ).add_to(map_surat)
            # Mark the best hospital
            folium.Marker(location=best_hospital_coords[::-1], popup="Nearest Safe Hospital", icon=folium.Icon(color='blue', icon='plus')).add_to(map_surat)

    # Display the map in Streamlit
    folium_static(map_surat)

def risk_clustering():
    st.subheader("Flood Risk Clustering in Surat")

    # Define the Area of Interest (AOI) for Surat, Gujarat
    surat_boundary = ox.geocode_to_gdf("Surat, Gujarat, India")

    # Flood data (latitude and longitude points)
    flood_data = pd.DataFrame({
        'latitude': [
            21.1702, 21.1750, 21.1850, 21.1950, 21.2102,
            21.1650, 21.1800, 21.1900, 21.2000, 21.2200,
            21.1685, 21.1735, 21.1845, 21.1945, 21.2150,
            21.1705, 21.1755, 21.1855, 21.1955, 21.2105,
            21.1655, 21.1805, 21.1905, 21.2005, 21.2205,
            21.1680, 21.1730, 21.1840, 21.1940, 21.2155
        ],
        'longitude': [
            72.8311, 72.8325, 72.8345, 72.8350, 72.8402,
            72.8290, 72.8320, 72.8340, 72.8360, 72.8410,
            72.8285, 72.8315, 72.8345, 72.8365, 72.8415,
            72.8321, 72.8335, 72.8355, 72.8370, 72.8422,
            72.8300, 72.8320, 72.8340, 72.8360, 72.8400,
            72.8280, 72.8300, 72.8320, 72.8350, 72.8420
        ]
    })

    # Perform K-Means clustering to identify high-risk zones
    kmeans = KMeans(n_clusters=3, random_state=42)  # Adjust the number of clusters based on your data
    flood_data['cluster'] = kmeans.fit_predict(flood_data[['latitude', 'longitude']])

    # Assign risk level and insurance plan based on clusters
    flood_data['risk_level'] = flood_data['cluster'].map({
        0: 'High Risk',
        1: 'Medium Risk',
        2: 'Low Risk'
    })

    flood_data['insurance_plan'] = flood_data['risk_level'].map({
        'High Risk': 'Comprehensive Plan - High Premium',
        'Medium Risk': 'Standard Plan - Moderate Premium',
        'Low Risk': 'Basic Plan - Low Premium'
    })

    # Create a base map centered around Surat
    map_center = [21.1702, 72.8311]  # Center map on Surat
    mymap = folium.Map(location=map_center, zoom_start=13)

    # Add the Surat boundary to the map
    folium.GeoJson(surat_boundary.to_json(), name="Surat Boundary").add_to(mymap)

    # Add the flood points to the map, color-coded by cluster
    colors = ['red', 'blue', 'green']
    for idx, row in flood_data.iterrows():
        cluster_index = int(row['cluster'])  # Convert to integer
        folium.CircleMarker(
            location=(row['latitude'], row['longitude']),
            radius=6,
            color=colors[cluster_index],
            fill=True,
            fill_color=colors[cluster_index],
            fill_opacity=0.6,
            popup=f"Risk Level: {row['risk_level']}<br>Insurance Plan: {row['insurance_plan']}"
        ).add_to(mymap)

    # Display the clustered map in Streamlit
    folium_static(mymap)

    # Show the data in a table for further interactivity
    st.write("Clustered Flood Data with Risk Levels and Insurance Plans")
    st.dataframe(flood_data)

# Run the app
if __name__ == "__main__":
    main()
