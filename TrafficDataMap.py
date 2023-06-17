import os
import requests
import pandas as pd
import numpy as np
import folium
import geopandas as gpd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

# Step 1: Collect OSM road data
def get_osm_data(center_latitude, center_longitude, radius):
    query = f"""
    [out:json];
    (
        way[highway](around:{radius},{center_latitude},{center_longitude});
    );
    out center;
    """
    response = requests.post('https://overpass-api.de/api/interpreter', data=query)
    try:
        data = response.json()
    except requests.exceptions.JSONDecodeError as e:
        print("Error decoding JSON response:")
        print(response.content)
        raise e
    return data

# Step 2: Preprocess and analyze the data (using simulated historical traffic data)
def preprocess_data(osm_data):
    roads = []

    for element in osm_data['elements']:
        if 'center' in element:
            roads.append({
                'id': element['id'],
                'latitude': element['center']['lat'],
                'longitude': element['center']['lon']
            })

    df = pd.DataFrame(roads)

    # Simulate historical traffic data (replace this with real historical traffic data)
    df['traffic'] = np.random.randint(1, 10, df.shape[0])

    return df

# Step 3: Develop a more robust machine learning model
def train_ml_model(processed_data):
    X = processed_data[['latitude', 'longitude']]
    y = processed_data['traffic']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = XGBRegressor()
    model.fit(X_train, y_train)

    return model

# Step 4: Create an interactive map
def create_map(processed_data, model):
    portland_map = folium.Map(location=[45.523064, -122.676483], zoom_start=12)

    for index, row in processed_data.iterrows():
        # Predict traffic using the model
        traffic_prediction = model.predict(row[['latitude', 'longitude']].values.reshape(1, -1))
        
        # Add traffic data markers or layers (replace this with your actual data)
        folium.Marker(
            [row['latitude'], row['longitude']],
            popup=f'Traffic Prediction: {traffic_prediction[0]:.2f}'
        ).add_to(portland_map)

    portland_map.save('portland_traffic_map.html')

# Main function to run the entire process
def main():
    center_latitude = 45.523064
    center_longitude = -122.676483
    radius = 5000
    osm_data = get_osm_data(center_latitude, center_longitude, radius)
    processed_data = preprocess_data(osm_data)
    ml_model = train_ml_model(processed_data)
    create_map(processed_data, ml_model)

if __name__ == '__main__':
    main()
