import os
import joblib
import numpy as np
import pandas as pd
import requests
import folium
from folium.plugins import HeatMap
from flask import Flask, render_template, request, jsonify
from flask_caching import Cache
from dotenv import load_dotenv

# Load environment variables if .env exists
load_dotenv()

app = Flask(__name__)

# Configure in-memory cache with 30-minute default expiration
app.config['CACHE_TYPE'] = 'SimpleCache'
app.config['CACHE_DEFAULT_TIMEOUT'] = 1800
cache = Cache(app)

# OpenWeather API Configuration
OPENWEATHER_API_KEY = os.environ.get("OPENWEATHER_API_KEY", "YOUR_OPENWEATHER_API_KEY")

# Load Machine Learning Model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "dengue_model.pkl")
model = None

try:
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        print(f"[*] Model successfully loaded from {MODEL_PATH}")
    else:
        print(f"[!] Warning: Model file not found at {MODEL_PATH}")
except Exception as e:
    print(f"[!] Error loading model: {e}")

# Major Indian Cities for Real-Time Dengue Risk Map Visualization
CITIES = [
    {"name": "New Delhi", "lat": 28.6139, "lon": 77.2090},
    {"name": "Mumbai", "lat": 19.0760, "lon": 72.8777},
    {"name": "Bengaluru", "lat": 12.9716, "lon": 77.5946},
    {"name": "Kolkata", "lat": 22.5726, "lon": 88.3639},
    {"name": "Chennai", "lat": 13.0827, "lon": 80.2707},
    {"name": "Hyderabad", "lat": 17.3850, "lon": 78.4867},
    {"name": "Ahmedabad", "lat": 23.0225, "lon": 72.5714},
    {"name": "Jaipur", "lat": 26.9124, "lon": 75.7873},
    {"name": "Lucknow", "lat": 26.8467, "lon": 80.9462},
    {"name": "Bhopal", "lat": 23.2599, "lon": 77.4126},
    {"name": "Surat", "lat": 21.1702, "lon": 72.8311},
    {"name": "Patna", "lat": 25.5941, "lon": 85.1376},
    {"name": "Pune", "lat": 18.5204, "lon": 73.8567},
    {"name": "Visakhapatnam", "lat": 17.6868, "lon": 83.2185},
    {"name": "Kochi", "lat": 9.9312, "lon": 76.2673},
    {"name": "Guwahati", "lat": 26.1445, "lon": 91.7362}
]

@cache.memoize(timeout=1800)
def fetch_weather_data(lat, lon):
    """
    Fetch current meteorological data from OpenWeather API with 30-min cache.
    Returns (temp, humidity, rainfall, wind_speed) or fallback defaults if unavailable.
    """
    if OPENWEATHER_API_KEY and OPENWEATHER_API_KEY != "YOUR_OPENWEATHER_API_KEY":
        try:
            url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={OPENWEATHER_API_KEY}&units=metric"
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                data = response.json()
                temp = round(float(data.get("main", {}).get("temp", 28.0)), 1)
                humidity = round(float(data.get("main", {}).get("humidity", 70.0)), 1)
                rainfall = round(float(data.get("rain", {}).get("1h", 0.0)), 1)
                wind = round(float(data.get("wind", {}).get("speed", 3.5)), 1)
                return temp, humidity, rainfall, wind
        except Exception as e:
            print(f"[!] OpenWeather API fetch failed for ({lat}, {lon}): {e}")

    # Fallback simulated weather conditions based on region
    np.random.seed(int((lat + lon) * 100) % 1000)
    sim_temp = round(float(np.random.uniform(23.0, 35.0)), 1)
    sim_humidity = round(float(np.random.uniform(50.0, 92.0)), 1)
    sim_rainfall = round(float(np.random.uniform(0.0, 22.0)), 1)
    sim_wind = round(float(np.random.uniform(2.0, 14.0)), 1)
    return sim_temp, sim_humidity, sim_rainfall, sim_wind

def predict_risk(temp, humidity, rainfall, wind):
    """
    Predict dengue risk using pre-trained ML model or epidemiological heuristic fallback.
    Returns: (is_high_risk: bool, label_text: str, risk_score: float)
    """
    risk_score = 0.0
    if model is not None:
        try:
            features = np.array([[float(temp), float(humidity), float(rainfall), float(wind)]])
            pred = model.predict(features)
            
            if hasattr(model, "predict_proba"):
                try:
                    proba = model.predict_proba(features)[0]
                    risk_score = float(proba[-1])
                except Exception:
                    risk_score = 0.85 if pred[0] > 0 else 0.15
            else:
                risk_score = 0.85 if pred[0] > 0 else 0.15

            if isinstance(pred[0], (int, np.integer)):
                is_high = bool(pred[0] > 0)
            elif isinstance(pred[0], (float, np.floating)):
                is_high = bool(pred[0] >= 0.5)
            elif isinstance(pred[0], str):
                is_high = pred[0].strip().lower() in ["high", "yes", "outbreak", "positive"]
            else:
                is_high = bool(pred[0])

            pct = round(risk_score * 100, 1)
            status = f"High Risk: Dengue Outbreak Likely! (Confidence: {pct}%)" if is_high else f"Low Risk: Dengue Outbreak Unlikely (Confidence: {100 - pct}%)"
            return is_high, status, risk_score
        except Exception as e:
            print(f"[!] Error during model.predict: {e}")

    # Epidemiological heuristic fallback
    is_high = (24.0 <= temp <= 35.0) and (humidity >= 65.0) and (rainfall > 5.0)
    risk_score = 0.85 if is_high else 0.20
    status = "High Risk: Dengue Outbreak Likely!" if is_high else "Low Risk: Dengue Outbreak Unlikely"
    return is_high, status, risk_score

@cache.cached(timeout=1800, key_prefix='cached_india_dengue_map')
def generate_india_map():
    """
    Creates an interactive Folium map centered on India with:
    1. A HeatMap layer visualizing regional outbreak intensity gradients.
    2. City CircleMarkers with detailed weather and risk popups.
    3. LayerControl allowing toggling between HeatMap and Markers.
    """
    india_map = folium.Map(
        location=[21.5, 78.9],
        zoom_start=5,
        tiles="OpenStreetMap",
        control_scale=True
    )

    heat_data = []
    markers_group = folium.FeatureGroup(name="City Risk Markers", show=True)

    for city in CITIES:
        temp, humidity, rainfall, wind = fetch_weather_data(city["lat"], city["lon"])
        is_high_risk, _, risk_score = predict_risk(temp, humidity, rainfall, wind)

        # Populate heat data: [lat, lon, weight/intensity]
        # High risk areas emit a higher intensity weight
        weight = float(max(0.2, min(1.0, risk_score)))
        heat_data.append([city["lat"], city["lon"], weight])

        color = "#e53935" if is_high_risk else "#43a047"
        status = "High Risk (Dengue Prone)" if is_high_risk else "Low Risk"

        popup_html = f"""
        <div style="font-family: Arial, sans-serif; min-width: 170px;">
            <h4 style="margin: 0 0 5px 0; color: #111;">{city['name']}</h4>
            <p style="margin: 3px 0; color: {color}; font-weight: bold;">
                Status: {status}
            </p>
            <hr style="margin: 5px 0;">
            <p style="margin: 2px 0; font-size: 12px;"><b>Temp:</b> {temp} °C</p>
            <p style="margin: 2px 0; font-size: 12px;"><b>Humidity:</b> {humidity} %</p>
            <p style="margin: 2px 0; font-size: 12px;"><b>Rainfall:</b> {rainfall} mm</p>
            <p style="margin: 2px 0; font-size: 12px;"><b>Wind:</b> {wind} km/h</p>
        </div>
        """

        folium.CircleMarker(
            location=[city["lat"], city["lon"]],
            radius=9,
            popup=folium.Popup(popup_html, max_width=300),
            tooltip=f"{city['name']}: {status}",
            color="#222222",
            weight=1.5,
            fill=True,
            fill_color=color,
            fill_opacity=0.85
        ).add_to(markers_group)

    # Add HeatMap Layer
    heatmap_group = folium.FeatureGroup(name="Outbreak HeatMap", show=True)
    HeatMap(
        heat_data,
        radius=30,
        blur=20,
        min_opacity=0.35,
        gradient={0.2: 'blue', 0.4: 'lime', 0.7: 'orange', 1.0: 'red'}
    ).add_to(heatmap_group)

    heatmap_group.add_to(india_map)
    markers_group.add_to(india_map)
    folium.LayerControl(collapsed=False).add_to(india_map)

    return india_map._repr_html_()

@app.route("/", methods=["GET"])
def index():
    map_html = generate_india_map()
    return render_template("index.html", map_html=map_html, cities=CITIES)

@app.route("/api/weather", methods=["GET"])
def api_weather():
    """
    Returns live or regional weather for a given lat/lon or city name.
    """
    lat = request.args.get("lat", type=float)
    lon = request.args.get("lon", type=float)
    city_name = request.args.get("city", type=str)

    if city_name:
        for c in CITIES:
            if c["name"].lower() == city_name.lower():
                lat, lon = c["lat"], c["lon"]
                break

    if lat is None or lon is None:
        return jsonify({"success": False, "error": "Missing coordinates or city"}), 400

    temp, humidity, rainfall, wind = fetch_weather_data(lat, lon)
    return jsonify({
        "success": True,
        "temperature": temp,
        "humidity": humidity,
        "rainfall": rainfall,
        "wind": wind
    })

@app.route("/api/predict", methods=["POST"])
def api_predict():
    """
    Asynchronous JSON endpoint for instant risk evaluation without page reloads.
    """
    try:
        data = request.get_json(silent=True) or request.form
        temp = float(data.get("temperature", 0))
        humidity = float(data.get("humidity", 0))
        rainfall = float(data.get("rainfall", 0))
        wind = float(data.get("wind", 0))

        if not (-20 <= temp <= 65):
            return jsonify({"success": False, "error": "Temperature must be between -20°C and 65°C."}), 400
        if not (0 <= humidity <= 100):
            return jsonify({"success": False, "error": "Humidity must be between 0% and 100%."}), 400
        if rainfall < 0 or wind < 0:
            return jsonify({"success": False, "error": "Rainfall and Wind cannot be negative."}), 400

        is_high_risk, prediction_text, _ = predict_risk(temp, humidity, rainfall, wind)

        return jsonify({
            "success": True,
            "is_high_risk": is_high_risk,
            "prediction_text": prediction_text
        })
    except Exception as e:
        return jsonify({"success": False, "error": f"Invalid input: {str(e)}"}), 400

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
