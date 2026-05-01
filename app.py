from flask import Flask, render_template, request, send_file
from werkzeug.utils import secure_filename
import os
import pandas as pd
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import folium
from folium.plugins import MarkerCluster
import io
import base64
import math

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Updated hotspots with latitude, longitude, and location names
hotspots = {
    'Barangay 2': {'name': 'Ambalatungan', 'lat': 16.717280, 'lon': 121.546510},
    'Barangay 3': {'name': 'Balintocatoc', 'lat': 16.649540, 'lon': 121.558700},
    'Barangay 4': {'name': 'Baluarte', 'lat': 16.669109, 'lon': 121.563553},
    'Barangay 6': {'name': 'Batal', 'lat': 16.706520, 'lon': 121.596062},
    'Barangay 7': {'name': 'Buenavista', 'lat': 16.700120, 'lon': 121.551460},
    'Barangay 14': {'name': 'Divisoria', 'lat': 16.700220, 'lon': 121.605140},
    'Barangay 18': {'name': 'Mabini', 'lat': 16.706659, 'lon': 121.556343},
    'Barangay 22': {'name': 'Patul', 'lat': 16.678699, 'lon': 121.539391},
    'Barangay 23': {'name': 'Rizal', 'lat': 16.719320, 'lon': 121.553740}
}

@app.route('/', methods=['GET', 'POST'])
def index():
    selected_barangay = None
    time_horizon = None
    if request.method == 'POST':
        # Handle file upload
        file = request.files['file']
        time_horizon = request.form['time_horizon']
        selected_barangay = request.form.get('barangay')
        
        # Secure file and save
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Load data and preprocess
        data = pd.read_csv(filepath)
        data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
        
        # Run forecasting and collect plots, metrics, warnings, and recommendations
        plots, metrics, average_people, people_involved_summary, warnings = generate_forecasts(data, time_horizon, selected_barangay)
        
        # Generate GIS map with color-coded markers and hover details
        map_html = generate_map(hotspots, average_people)
        
        return render_template('index.html', plots=plots, metrics=metrics, time_horizon=time_horizon, map_html=map_html, people_involved_summary=people_involved_summary, warnings=warnings, barangay=selected_barangay, hotspots=hotspots)
    
    # Render template for a GET request
    return render_template('index.html', barangay=selected_barangay, hotspots=hotspots)


def generate_forecasts(data, time_horizon, selected_barangay):
    all_metrics = {'Timeframe': [], 'MSE': [], 'MAE': [], 'RMSE': []}
    average_people = {}
    people_involved_summary = {}
    plots = {}
    warnings = []
    timeframes = {
        '3_weeks': 21, '1_month': 30, '3_months': 90,
        '6_months': 180, '9_months': 270, '12_months': 365
    }
    
    barangay_names_order = [
        'Ambalatungan', 'Balintocatoc', 'Baluarte', 'Batal', 
        'Buenavista', 'Divisoria', 'Mabini', 'Patul', 'Rizal'
    ]
    
    for barangay, hotspot_info in hotspots.items():
        if selected_barangay and barangay != selected_barangay:
            continue

        hotspot_data = data[data['Barangays'] == barangay]
        
        # Aggregate data by date for the hotspot
        aggregated_data = hotspot_data.groupby('Date').sum(numeric_only=True).reset_index()
        aggregated_data['Total_Involved'] = aggregated_data['Resident'] + aggregated_data['Non-Resident']
        prophet_data = aggregated_data[['Date', 'Total_Involved']].rename(columns={'Date': 'ds', 'Total_Involved': 'y'})

        # Fit Prophet model
        prophet_model = Prophet(yearly_seasonality=True, weekly_seasonality=True, seasonality_mode='multiplicative')
        prophet_model.add_country_holidays(country_name='PH')
        prophet_model.fit(prophet_data)
        
        # Future forecast
        future_dates = prophet_model.make_future_dataframe(periods=365)
        prophet_forecast = prophet_model.predict(future_dates)

        # ARIMA model
        arima_model = ARIMA(prophet_data['y'], order=(5, 1, 0))
        arima_results = arima_model.fit()
        arima_forecast = arima_results.get_forecast(steps=365).predicted_mean

        # Hybrid model
        forecasted_data = pd.DataFrame({
            'Date': future_dates['ds'],
            'Hybrid': (prophet_forecast['yhat'] + np.concatenate([prophet_data['y'], arima_forecast])[:len(future_dates)]) / 2
        })

        # Extract expected people involved based on the selected time horizon
        end = timeframes[time_horizon]
        if time_horizon == '3_weeks':
            # Use weekly granularity for 3-week horizon
            future_data_filtered = forecasted_data[forecasted_data['Date'] > prophet_data['ds'].max()][:end]
            weeks = future_data_filtered['Date'].dt.to_period('W').unique()
            people_involved_summary = {str(week): future_data_filtered[future_data_filtered['Date'].dt.to_period('W') == week]['Hybrid'].mean() for week in weeks}
        else:
            # Use monthly granularity for other horizons
            future_data_filtered = forecasted_data[forecasted_data['Date'] > prophet_data['ds'].max()][:end]
            months = future_data_filtered['Date'].dt.to_period('M').unique()
            people_involved_summary = {str(month): future_data_filtered[future_data_filtered['Date'].dt.to_period('M') == month]['Hybrid'].mean() for month in months}

        # Additional calculations for metrics
        actual_values = prophet_data['y'][-end:]
        predicted_values = forecasted_data[forecasted_data['Date'] <= prophet_data['ds'].max()]['Hybrid'][-end:]
        
        avg_people_involved = predicted_values.mean()
        average_people[barangay] = avg_people_involved
        
        min_length = min(len(actual_values), len(predicted_values))
        actual_values = actual_values[-min_length:]
        predicted_values = predicted_values[-min_length:]
        
        mse = mean_squared_error(actual_values, predicted_values)
        mae = mean_absolute_error(actual_values, predicted_values)
        rmse = np.sqrt(mse)
        
        all_metrics['Timeframe'].append(time_horizon)
        all_metrics['MSE'].append(mse)
        all_metrics['MAE'].append(mae)
        all_metrics['RMSE'].append(rmse)

        # Generate historical forecast plot and save as base64
        plt.figure(figsize=(10, 5))
        plt.plot(prophet_data['ds'], prophet_data['y'], label='Actual', color='black')
        plt.plot(forecasted_data['Date'], forecasted_data['Hybrid'], label='Hybrid Forecast', color='blue')
        plt.axvline(x=prophet_data['ds'].max(), color='red', linestyle='--')
        plt.title(f"Historical Forecast Comparison for {hotspot_info['name']}")
        plt.xlabel('Date')
        plt.ylabel('Number of People Involved')
        plt.legend()
        
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plots[f"{hotspot_info['name']}_historical"] = base64.b64encode(img.getvalue()).decode()
        plt.close()

        # Generate future forecast plot for selected time horizon and save as base64
        future_data = forecasted_data[forecasted_data['Date'] > prophet_data['ds'].max()][:end]
        plt.figure(figsize=(10, 5))
        plt.plot(future_data['Date'], future_data['Hybrid'], label=f'{time_horizon.replace("_", " ").capitalize()} Forecast', color='blue')
        plt.title(f"Future Forecast for {hotspot_info['name']} ({time_horizon.replace('_', ' ').capitalize()})")
        plt.xlabel('Date')
        plt.ylabel('Forecasted Number of People Involved')
        plt.legend()
        
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plots[f"{hotspot_info['name']}_future"] = base64.b64encode(img.getvalue()).decode()
        plt.close()
    
    # Assess risk levels based on your existing threshold
    WARNING_THRESHOLD = 10  # Replace this with your existing threshold if it's defined elsewhere
    risk_assessment = {
        "high_risk": [],
        "moderate_risk": []
    }

    for i, (barangay, avg_people) in enumerate(average_people.items()):
        if avg_people > WARNING_THRESHOLD:
            risk_assessment["high_risk"].append(hotspots[barangay]['name'])
        elif avg_people > (0.7 * WARNING_THRESHOLD):  # Moderate threshold is 70% of WARNING_THRESHOLD
            risk_assessment["moderate_risk"].append(hotspots[barangay]['name'])

    # Generate warnings with recommendations based on risk levels
    if risk_assessment["high_risk"]:
        warnings.append({
            "message": "High accident risk detected in the following barangays:",
            "barangays": risk_assessment["high_risk"],
            "recommendations": [
                "Increase patrols in high-risk areas.",
                "Install additional warning signs or lights.",
                "Consider road infrastructure improvements in these areas."
            ]
        })

    if risk_assessment["moderate_risk"]:
        warnings.append({
            "message": "Moderate accident risk detected in the following barangays:",
            "barangays": risk_assessment["moderate_risk"],
            "recommendations": [
                "Conduct community awareness programs on road safety.",
                "Introduce speed management measures.",
                "Increase visibility of traffic warnings and signs."
            ]
        })
    
    # If no high or moderate risk, add low risk message for selected barangay
    if not risk_assessment["high_risk"] and not risk_assessment["moderate_risk"] and selected_barangay:
        warnings.append({
            "message": f"Low accident risk detected in {hotspots[selected_barangay]['name']}.",
            "barangays": [hotspots[selected_barangay]['name']],
            "recommendations": [
                "Maintain current safety measures.",
                "Encourage community participation in safety programs."
            ]
        })
    elif not risk_assessment["high_risk"] and not risk_assessment["moderate_risk"]:
        warnings.append({
            "message": "Low accident risk detected in all barangays.",
            "barangays": list(hotspots[barangay]['name'] for barangay in hotspots),
            "recommendations": [
                "Maintain current safety measures.",
                "Encourage community participation in safety programs."
            ]
        })
    
    # Filter out months with NaN values in people involved summary
    people_involved_summary = {month: count for month, count in people_involved_summary.items() if not math.isnan(count)}

    average_metrics = {
        'rmse': np.mean(all_metrics['RMSE']),
        'mse': np.mean(all_metrics['MSE']),
        'mae': np.mean(all_metrics['MAE'])
    }
    
    return plots, average_metrics, average_people, people_involved_summary, warnings


def generate_map(hotspots, average_people):
    # Center map on the region
    m = folium.Map(location=[16.7000, 121.5600], zoom_start=12)
    marker_cluster = MarkerCluster().add_to(m)
    
    for barangay, info in hotspots.items():
        avg_people = average_people.get(barangay, 0)
        if avg_people <= 7:
            color = 'green'
        elif 7 < avg_people <= 10:
            color = 'orange'
        else:
            color = 'red'
        
        popup_text = f"<strong>{barangay}</strong><br>{info['name']}<br>Average People Involved: {avg_people:.2f}"
        tooltip_text = f"{barangay} - {info['name']} (Avg: {avg_people:.2f})"
        
        folium.Marker(
            location=[info['lat'], info['lon']],
            popup=popup_text,
            tooltip=tooltip_text,
            icon=folium.Icon(color=color, icon="info-sign")
        ).add_to(marker_cluster)
    
    map_path = os.path.join('templates', 'map.html')
    m.save(map_path)
    return map_path

@app.route('/map')
def map():
    return render_template('map.html')

if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    # This part helps the app find the right port on a live server
    import os
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
