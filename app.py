import os
import joblib
import numpy as np
import pandas as pd
import requests
import folium
from flask import Flask, render_template, request, jsonify
from flask_caching import Cache

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
    {"name": "Patna", "lat": 25.5941, "lon": 85.1376}
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
                temp = data.get("main", {}).get("temp", 28.0)
                humidity = data.get("main", {}).get("humidity", 70.0)
                rainfall = data.get("rain", {}).get("1h", 0.0)
                wind = data.get("wind", {}).get("speed", 3.5)
                return temp, humidity, rainfall, wind
        except Exception as e:
            print(f"[!] OpenWeather API fetch failed for ({lat}, {lon}): {e}")

    # Fallback simulated weather conditions based on region
    np.random.seed(int((lat + lon) * 100) % 1000)
    sim_temp = float(np.random.uniform(22.0, 36.0))
    sim_humidity = float(np.random.uniform(45.0, 90.0))
    sim_rainfall = float(np.random.uniform(0.0, 25.0))
    sim_wind = float(np.random.uniform(2.0, 15.0))
    return sim_temp, sim_humidity, sim_rainfall, sim_wind

def predict_risk(temp, humidity, rainfall, wind):
    """
    Predict dengue risk using pre-trained ML model or epidemiological heuristic fallback.
    Returns: (is_high_risk: bool, label_text: str, probability_str: str)
    """
    if model is not None:
        try:
            features = np.array([[float(temp), float(humidity), float(rainfall), float(wind)]])
            pred = model.predict(features)
            
            # Predict probabilities if supported by the model
            prob_percent = None
            if hasattr(model, "predict_proba"):
                try:
                    proba = model.predict_proba(features)[0]
                    # Assume positive outbreak class is index 1
                    prob_percent = round(float(proba[-1]) * 100, 1)
                except Exception:
                    pass

            if isinstance(pred[0], (int, np.integer)):
                is_high = bool(pred[0] > 0)
            elif isinstance(pred[0], (float, np.floating)):
                is_high = bool(pred[0] >= 0.5)
            elif isinstance(pred[0], str):
                is_high = pred[0].strip().lower() in ["high", "yes", "outbreak", "positive"]
            else:
                is_high = bool(pred[0])

            status = "High Risk: Dengue Outbreak Likely!" if is_high else "Low Risk: Dengue Outbreak Unlikely"
            if prob_percent is not None:
                status += f" (Confidence: {prob_percent}%)"
            return is_high, status
        except Exception as e:
            print(f"[!] Error during model.predict: {e}")

    # Epidemiological heuristic fallback: Dengue transmission thrives at 24-35°C, >=65% humidity, and rainfall
    is_high = (24.0 <= temp <= 35.0) and (humidity >= 65.0) and (rainfall > 5.0)
    status = "High Risk: Dengue Outbreak Likely!" if is_high else "Low Risk: Dengue Outbreak Unlikely"
    return is_high, status

@cache.cached(timeout=1800, key_prefix='cached_india_dengue_map')
def generate_india_map():
    """
    Creates an interactive Folium map centered on India with color-coded markers for dengue risk zones.
    Result is cached in memory for 30 minutes to eliminate repeated API queries.
    """
    india_map = folium.Map(
        location=[21.5, 78.9],
        zoom_start=5,
        tiles="OpenStreetMap",
        control_scale=True
    )

    for city in CITIES:
        temp, humidity, rainfall, wind = fetch_weather_data(city["lat"], city["lon"])
        is_high_risk, _ = predict_risk(temp, humidity, rainfall, wind)

        color = "red" if is_high_risk else "green"
        status = "High Risk (Dengue Prone)" if is_high_risk else "Low Risk"

        popup_html = f"""
        <div style="font-family: Arial, sans-serif; min-width: 170px;">
            <h4 style="margin: 0 0 5px 0;">{city['name']}</h4>
            <p style="margin: 3px 0; color: {'red' if is_high_risk else 'green'}; font-weight: bold;">
                Status: {status}
            </p>
            <hr style="margin: 5px 0;">
            <p style="margin: 2px 0; font-size: 12px;"><b>Temp:</b> {temp:.1f} °C</p>
            <p style="margin: 2px 0; font-size: 12px;"><b>Humidity:</b> {humidity:.1f} %</p>
            <p style="margin: 2px 0; font-size: 12px;"><b>Rainfall:</b> {rainfall:.1f} mm</p>
            <p style="margin: 2px 0; font-size: 12px;"><b>Wind:</b> {wind:.1f} km/h</p>
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
        ).add_to(india_map)

    return india_map._repr_html_()

@app.route("/", methods=["GET"])
def index():
    map_html = generate_india_map()
    return render_template("index.html", map_html=map_html)

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

        # Basic input sanity validation
        if not (-20 <= temp <= 65):
            return jsonify({"success": False, "error": "Temperature must be between -20°C and 65°C."}), 400
        if not (0 <= humidity <= 100):
            return jsonify({"success": False, "error": "Humidity must be between 0% and 100%."}), 400
        if rainfall < 0 or wind < 0:
            return jsonify({"success": False, "error": "Rainfall and Wind cannot be negative."}), 400

        is_high_risk, prediction_text = predict_risk(temp, humidity, rainfall, wind)

        return jsonify({
            "success": True,
            "is_high_risk": is_high_risk,
            "prediction_text": prediction_text
        })
    except Exception as e:
        return jsonify({"success": False, "error": f"Invalid input: {str(e)}"}), 400

# Backward compatibility for traditional form post fallback
@app.route("/predict", methods=["POST"])
def predict():
    map_html = generate_india_map()
    try:
        temp = float(request.form.get("temperature", 0))
        humidity = float(request.form.get("humidity", 0))
        rainfall = float(request.form.get("rainfall", 0))
        wind = float(request.form.get("wind", 0))

        request_values = {
            "temperature": temp,
            "humidity": humidity,
            "rainfall": rainfall,
            "wind": wind
        }

        _, prediction_text = predict_risk(temp, humidity, rainfall, wind)
    except Exception as e:
        prediction_text = f"Invalid input or prediction error: {e}"
        request_values = None

    return render_template(
        "index.html",
        map_html=map_html,
        prediction_text=prediction_text,
        request_values=request_values
    )

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
