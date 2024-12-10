# Dengue Outbreak Prediction 🌦️🦟<Flask/Javascript>

This application predicts the likelihood of a dengue outbreak based on weather conditions using a pre-trained model and real-time weather data. The app visualizes predictions on an interactive map of India.

## Features

- **User Inputs**: Temperature, Humidity, Rainfall, and Wind data.
- **Map Integration**: India map visualization with color-coded dengue prediction zones.
- **Real-Time Weather Data**: Uses OpenWeather API to fetch live data.
- **Prediction Model**: Machine learning model trained to predict dengue likelihood.

## Setup Instructions

### Prerequisites

- **Python 3.6+**
- **Joblib, Flask, Requests, Folium** libraries

### Installation

1. **Clone the repository**:
    ```bash
    git clone https://github.com/yourusername/dengue-prediction-app.git
    cd dengue-prediction-app
    ```

2. **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3. **Add OpenWeather API Key**:
   Replace `API_KEY` in `app.py` with your OpenWeather API key.

4. **Place the ML model file**:
   Make sure `dengue_model.pkl` is in the specified directory or update the path in `app.py`.

### Usage

1. **Run the Flask App**:
    ```bash
    python app.py
    ```

2. **Open in Browser**:
   Navigate to `http://127.0.0.1:5000/` to access the app.

## Files

- **app.py**: Main Flask application file.
- **templates/index.html**: HTML template with prediction form and map.
- **static/style.css**: Basic styling for the web app.

## Screenshots
![Dengue Prediction](Dengue.png)



## Contributing

Feel free to open issues, contribute, or suggest improvements!
Dengue Outbreak Prediction 🌦️🦟 <Flask/Javascript>
This application predicts the likelihood of a dengue outbreak based on weather conditions using a pre-trained model and real-time weather data. The app visualizes predictions on an interactive map of India.


🚀 Getting Started
Open Using Daytona
Install Daytona: Follow the Daytona installation guide.

Create the Workspace:

bash
daytona create https://github.com/yourusername/dengue-prediction-app.git
Setup Instructions
Prerequisites
Python 3.6+

Joblib, Flask, Requests, Folium libraries

Installation
Clone the repository:

bash
git clone https://github.com/yourusername/dengue-prediction-app.git
cd dengue-prediction-app
Install dependencies:

bash
pip install -r requirements.txt
Add OpenWeather API Key: Replace API_KEY in app.py with your OpenWeather API key.

Place the ML model file: Make sure dengue_model.pkl is in the specified directory or update the path in app.py.

Start the Application:
bash
python app.py
```




