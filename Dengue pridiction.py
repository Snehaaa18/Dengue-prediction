from flask import Flask, request, render_template
import joblib
import folium
import requests


app = Flask(__name__)

# Load your saved model
model = joblib.load(r'C:\Users\sneha\Documents\project\dengue\dengue_model.pkl')
# OpenWeather API key
API_KEY = '9443f00722c1850c03bce88203f13b0b'

# Function to get real-time weather data
def get_weather_data(lat, lon):
    url = f'https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={API_KEY}&units=metric'
    response = requests.get(url)
    data = response.json()
    
    if response.status_code == 200:
        # Extract weather data
        temperature = data['main']['temp']
        humidity = data['main']['humidity']
        wind = data['wind']['speed']
        rainfall = data['rain']['1h'] if 'rain' in data else 0  # Rainfall in the past hour
        return temperature, humidity, rainfall, wind
    else:
        return None


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    temperature = float(request.form['Temperature'])
    humidity = float(request.form['Humidity'])
    rainfall = float(request.form['Rainfall'])
    wind = float(request.form['Wind']) 
    
    prediction = model.predict([[temperature, humidity, rainfall, wind]])
    
    result = 'Prone to Dengue' if prediction[0] == 1 else 'Not Prone to Dengue'
    return render_template('index.html', prediction_text=result)

if __name__ == "__main__":
    app.run(debug=True)
