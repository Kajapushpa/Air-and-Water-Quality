from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# AQI breakpoints mapping for pollutants
aqi_breakpoints = {
    'PM2.5': [(0, 30, 0, 50), (31, 60, 51, 100), (61, 90, 101, 150), (91, 120, 151, 200), (121, 250, 201, 300), (251, 500, 301, 500)],
    'PM10': [(0, 50, 0, 50), (51, 100, 51, 100), (101, 250, 101, 200), (251, 350, 201, 300), (351, 430, 301, 400), (431, 600, 401, 500)],
    'NO2': [(0, 40, 0, 50), (41, 80, 51, 100), (81, 180, 101, 200), (181, 280, 201, 300), (281, 400, 301, 400), (401, 600, 401, 500)],
    'SO2': [(0, 40, 0, 50), (41, 80, 51, 100), (81, 380, 101, 200), (381, 800, 201, 300), (801, 1600, 301, 400), (1601, 2100, 401, 500)],
    'CO': [(0, 4.4, 0, 50), (4.5, 9.4, 51, 100), (9.5, 12.4, 101, 150), (12.5, 15.4, 151, 200), (15.5, 30.4, 201, 300), (30.5, 50.4, 301, 500)]
}

def calculate_aqi(concentration, pollutant):
    for bp in aqi_breakpoints.get(pollutant, []):
        if concentration <= bp[1]:
            return ((bp[3] - bp[2]) / (bp[1] - bp[0])) * (concentration - bp[0]) + bp[2]
    return 500  # Max AQI

def calculate_overall_aqi(row):
    return max((calculate_aqi(row[p], p) for p in aqi_breakpoints if p in row), default=None)

def calculate_wqi(row):
    weights = {'pH': 0.11, 'HCO3': 0.10, 'Cl': 0.08, 'SO4': 0.10, 'NO3': 0.10, 'Ca': 0.10, 'Mg': 0.10, 'Na': 0.10}
    quality_index = {'pH': (6.5, 8.5), 'HCO3': (200, 600), 'Cl': (250, 1000), 'SO4': (200, 400), 'NO3': (45, 100), 'Ca': (75, 200), 'Mg': (30, 150), 'Na': (200, 600)}
    wqi = sum(((max(0, min(100, ((row[p] - ideal_min) / (ideal_max - ideal_min)) * 100))) * weights[p]) for p, (ideal_min, ideal_max) in quality_index.items() if p in row)
    return round(wqi, 2)

# Load data
air_df = pd.read_excel('Air 2022.xlsx')
water_df = pd.read_excel('Water 2022.xlsx')
air_df['Calculated_AQI'] = air_df.apply(calculate_overall_aqi, axis=1)
water_df['WQI'] = water_df.apply(calculate_wqi, axis=1)

# Features & Model Setup
AIR_FEATURES = ['Temperature', 'Humidity', 'PM2.5', 'PM10', 'NO2', 'SO2', 'CO']
WATER_FEATURES = ['pH', 'HCO3', 'Cl', 'SO4', 'NO3', 'Ca', 'Mg', 'Na']

X_air_train, X_air_test, y_air_train, y_air_test = train_test_split(air_df[AIR_FEATURES], air_df['Calculated_AQI'], test_size=0.2, random_state=42)
X_water_train, X_water_test, y_water_train, y_water_test = train_test_split(water_df[WATER_FEATURES], water_df['WQI'], test_size=0.2, random_state=42)

air_model = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_air_train, y_air_train)
water_model = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_water_train, y_water_train)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        state, district, parameter = request.form['state'].strip().title(), request.form['district'].strip().title(), request.form['parameter']
        if parameter == 'AQI':
            filtered = air_df[(air_df['State'].str.title() == state) & (air_df['District'].str.title() == district)]
            if filtered.empty:
                return jsonify({'error': 'No air quality data found'})
            return jsonify({'result': float(air_model.predict(filtered[AIR_FEATURES])[0])})
        elif parameter == 'WQI':
            filtered = water_df[(water_df['STATE'].str.title() == state) & (water_df['DISTRICT'].str.title() == district)]
            if filtered.empty:
                return jsonify({'error': 'No water quality data found'})
            return jsonify({'result': float(water_model.predict(filtered[WATER_FEATURES])[0])})
        return jsonify({'error': 'Invalid parameter'})
    except Exception as e:
        return jsonify({'error': f'Server error: {str(e)}'})

if __name__ == '__main__':
    app.run(debug=True)
