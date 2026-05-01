from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

import base64
import io
import math
import os
import traceback

import folium
from folium.plugins import MarkerCluster
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA


app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# Hotspot barangays with coordinates
hotspots = {
    "Barangay 2": {"name": "Ambalatungan", "lat": 16.717280, "lon": 121.546510},
    "Barangay 3": {"name": "Balintocatoc", "lat": 16.649540, "lon": 121.558700},
    "Barangay 4": {"name": "Baluarte", "lat": 16.669109, "lon": 121.563553},
    "Barangay 6": {"name": "Batal", "lat": 16.706520, "lon": 121.596062},
    "Barangay 7": {"name": "Buenavista", "lat": 16.700120, "lon": 121.551460},
    "Barangay 14": {"name": "Divisoria", "lat": 16.700220, "lon": 121.605140},
    "Barangay 18": {"name": "Mabini", "lat": 16.706659, "lon": 121.556343},
    "Barangay 22": {"name": "Patul", "lat": 16.678699, "lon": 121.539391},
    "Barangay 23": {"name": "Rizal", "lat": 16.719320, "lon": 121.553740},
}


@app.route("/", methods=["GET", "POST"])
def index():
    selected_barangay = None
    time_horizon = None

    if request.method == "POST":
        try:
            if "file" not in request.files:
                return "No file uploaded.", 400

            file = request.files["file"]

            if file.filename == "":
                return "No selected file.", 400

            time_horizon = request.form.get("time_horizon")
            selected_barangay = request.form.get("barangay")

            if selected_barangay in ["", "all", "All", "All Barangays", None]:
                selected_barangay = None

            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)

            data = pd.read_csv(filepath)
            data.columns = data.columns.str.strip()

            required_columns = ["Date", "Barangays", "Resident", "Non-Resident"]
            missing_columns = [
                column for column in required_columns if column not in data.columns
            ]

            if missing_columns:
                return f"CSV is missing these columns: {missing_columns}", 400

            data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
            data["Barangays"] = data["Barangays"].astype(str).str.strip()
            data["Resident"] = pd.to_numeric(data["Resident"], errors="coerce").fillna(0)
            data["Non-Resident"] = pd.to_numeric(
                data["Non-Resident"], errors="coerce"
            ).fillna(0)

            data = data.dropna(subset=["Date"])

            plots, metrics, average_people, people_involved_summary, warnings = (
                generate_forecasts(data, time_horizon, selected_barangay)
            )

            generate_map(hotspots, average_people)

            return render_template(
                "index.html",
                plots=plots,
                metrics=metrics,
                time_horizon=time_horizon,
                map_html="map.html",
                people_involved_summary=people_involved_summary,
                warnings=warnings,
                barangay=selected_barangay,
                hotspots=hotspots,
            )

        except Exception:
            print("ERROR DURING FORECAST:")
            print(traceback.format_exc())

            return (
                "An error occurred while processing the forecast. "
                "Please check the Render logs for the full traceback."
            ), 500

    return render_template(
        "index.html",
        barangay=selected_barangay,
        hotspots=hotspots,
    )


def generate_forecasts(data, time_horizon, selected_barangay):
    all_metrics = {"Timeframe": [], "MSE": [], "MAE": [], "RMSE": []}
    average_people = {}
    summary_collector = {}
    plots = {}
    warnings = []

    timeframes = {
        "3_weeks": 21,
        "1_month": 30,
        "3_months": 90,
        "6_months": 180,
        "9_months": 270,
        "12_months": 365,
    }

    if time_horizon not in timeframes:
        time_horizon = "1_month"

    forecast_days = timeframes[time_horizon]

    for barangay, hotspot_info in hotspots.items():
        if selected_barangay and barangay != selected_barangay:
            continue

        hotspot_data = data[data["Barangays"] == barangay]

        if hotspot_data.empty:
            warnings.append(
                {
                    "message": f"No data found for {hotspot_info['name']}.",
                    "barangays": [hotspot_info["name"]],
                    "recommendations": [
                        "Check if the Barangays column matches the required format, such as Barangay 2, Barangay 3, etc."
                    ],
                }
            )
            continue

        aggregated_data = (
            hotspot_data.groupby("Date")
            .sum(numeric_only=True)
            .reset_index()
            .sort_values("Date")
        )

        if (
            "Resident" not in aggregated_data.columns
            or "Non-Resident" not in aggregated_data.columns
        ):
            warnings.append(
                {
                    "message": f"Missing Resident or Non-Resident data for {hotspot_info['name']}.",
                    "barangays": [hotspot_info["name"]],
                    "recommendations": [
                        "Make sure the CSV has Resident and Non-Resident columns with numeric values."
                    ],
                }
            )
            continue

        aggregated_data["Total_Involved"] = (
            aggregated_data["Resident"] + aggregated_data["Non-Resident"]
        )

        aggregated_data = aggregated_data.dropna(subset=["Date", "Total_Involved"])

        if len(aggregated_data) < 3:
            warnings.append(
                {
                    "message": f"Not enough data to forecast for {hotspot_info['name']}.",
                    "barangays": [hotspot_info["name"]],
                    "recommendations": [
                        "Upload at least 3 valid date records for this barangay."
                    ],
                }
            )
            continue

        prophet_data = aggregated_data[["Date", "Total_Involved"]].rename(
            columns={"Date": "ds", "Total_Involved": "y"}
        )

        prophet_data["y"] = pd.to_numeric(prophet_data["y"], errors="coerce")
        prophet_data = prophet_data.dropna(subset=["ds", "y"])

        if len(prophet_data) < 3:
            warnings.append(
                {
                    "message": f"Not enough clean data to forecast for {hotspot_info['name']}.",
                    "barangays": [hotspot_info["name"]],
                    "recommendations": [
                        "Make sure the selected barangay has enough valid Date, Resident, and Non-Resident records."
                    ],
                }
            )
            continue

        try:
            prophet_model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=True,
                seasonality_mode="multiplicative",
            )
            prophet_model.add_country_holidays(country_name="PH")
            prophet_model.fit(prophet_data)

            future_dates = prophet_model.make_future_dataframe(periods=365)
            prophet_forecast = prophet_model.predict(future_dates)

            y_values = prophet_data["y"].reset_index(drop=True)
            forecast_steps = 365

            try:
                if len(y_values) >= 8:
                    p_value = min(5, len(y_values) - 2)
                    arima_model = ARIMA(y_values, order=(p_value, 1, 0))
                    arima_results = arima_model.fit()
                    arima_forecast = arima_results.get_forecast(
                        steps=forecast_steps
                    ).predicted_mean
                else:
                    arima_forecast = pd.Series([y_values.mean()] * forecast_steps)
            except Exception:
                arima_forecast = pd.Series([y_values.mean()] * forecast_steps)

            arima_combined = np.concatenate([y_values, arima_forecast])
            arima_combined = arima_combined[: len(future_dates)]

            forecasted_data = pd.DataFrame(
                {
                    "Date": future_dates["ds"],
                    "Hybrid": (prophet_forecast["yhat"].values + arima_combined) / 2,
                }
            )

            forecasted_data["Hybrid"] = forecasted_data["Hybrid"].clip(lower=0)

            future_data_filtered = forecasted_data[
                forecasted_data["Date"] > prophet_data["ds"].max()
            ].head(forecast_days)

            if future_data_filtered.empty:
                warnings.append(
                    {
                        "message": f"No future forecast generated for {hotspot_info['name']}.",
                        "barangays": [hotspot_info["name"]],
                        "recommendations": ["Check the uploaded CSV data and try again."],
                    }
                )
                continue

            if time_horizon == "3_weeks":
                period_values = future_data_filtered.copy()
                period_values["Period"] = (
                    period_values["Date"].dt.to_period("W").astype(str)
                )
            else:
                period_values = future_data_filtered.copy()
                period_values["Period"] = (
                    period_values["Date"].dt.to_period("M").astype(str)
                )

            for period, group in period_values.groupby("Period"):
                summary_collector.setdefault(period, []).append(group["Hybrid"].mean())

            historical_predictions = forecasted_data[
                forecasted_data["Date"] <= prophet_data["ds"].max()
            ]["Hybrid"]

            actual_values = prophet_data["y"].tail(forecast_days)
            predicted_values = historical_predictions.tail(forecast_days)

            min_length = min(len(actual_values), len(predicted_values))

            if min_length > 0:
                actual_values = actual_values.tail(min_length)
                predicted_values = predicted_values.tail(min_length)

                mse = mean_squared_error(actual_values, predicted_values)
                mae = mean_absolute_error(actual_values, predicted_values)
                rmse = np.sqrt(mse)

                all_metrics["Timeframe"].append(time_horizon)
                all_metrics["MSE"].append(mse)
                all_metrics["MAE"].append(mae)
                all_metrics["RMSE"].append(rmse)

            avg_people_involved = future_data_filtered["Hybrid"].mean()
            average_people[barangay] = avg_people_involved

            historical_plot = create_historical_plot(
                prophet_data,
                forecasted_data,
                hotspot_info["name"],
            )
            plots[f"{hotspot_info['name']}_historical"] = historical_plot

            future_plot = create_future_plot(
                future_data_filtered,
                hotspot_info["name"],
                time_horizon,
            )
            plots[f"{hotspot_info['name']}_future"] = future_plot

        except Exception as error:
            warnings.append(
                {
                    "message": f"Forecast could not be generated for {hotspot_info['name']}.",
                    "barangays": [hotspot_info["name"]],
                    "recommendations": [
                        f"Technical detail: {str(error)}",
                        "Check if the uploaded CSV has enough valid records for this barangay.",
                    ],
                }
            )
            continue

    people_involved_summary = {
        period: np.mean(values)
        for period, values in summary_collector.items()
        if values and not math.isnan(np.mean(values))
    }

    if all_metrics["RMSE"]:
        average_metrics = {
            "rmse": round(float(np.mean(all_metrics["RMSE"])), 4),
            "mse": round(float(np.mean(all_metrics["MSE"])), 4),
            "mae": round(float(np.mean(all_metrics["MAE"])), 4),
        }
    else:
        average_metrics = {
            "rmse": 0,
            "mse": 0,
            "mae": 0,
        }

    add_risk_warnings(warnings, average_people, selected_barangay)

    return plots, average_metrics, average_people, people_involved_summary, warnings


def create_historical_plot(prophet_data, forecasted_data, barangay_name):
    plt.figure(figsize=(10, 5))
    plt.plot(prophet_data["ds"], prophet_data["y"], label="Actual", color="black")
    plt.plot(
        forecasted_data["Date"],
        forecasted_data["Hybrid"],
        label="Hybrid Forecast",
        color="blue",
    )
    plt.axvline(x=prophet_data["ds"].max(), color="red", linestyle="--")
    plt.title(f"Historical Forecast Comparison for {barangay_name}")
    plt.xlabel("Date")
    plt.ylabel("Number of People Involved")
    plt.legend()

    return save_plot_to_base64()


def create_future_plot(future_data, barangay_name, time_horizon):
    plt.figure(figsize=(10, 5))
    plt.plot(
        future_data["Date"],
        future_data["Hybrid"],
        label=f'{time_horizon.replace("_", " ").capitalize()} Forecast',
        color="blue",
    )
    plt.title(
        f"Future Forecast for {barangay_name} "
        f"({time_horizon.replace('_', ' ').capitalize()})"
    )
    plt.xlabel("Date")
    plt.ylabel("Forecasted Number of People Involved")
    plt.legend()

    return save_plot_to_base64()


def save_plot_to_base64():
    img = io.BytesIO()
    plt.tight_layout()
    plt.savefig(img, format="png")
    img.seek(0)

    encoded_image = base64.b64encode(img.getvalue()).decode()
    plt.close()

    return encoded_image


def add_risk_warnings(warnings, average_people, selected_barangay):
    warning_threshold = 10

    risk_assessment = {
        "high_risk": [],
        "moderate_risk": [],
    }

    for barangay, avg_people in average_people.items():
        if avg_people > warning_threshold:
            risk_assessment["high_risk"].append(hotspots[barangay]["name"])
        elif avg_people > (0.7 * warning_threshold):
            risk_assessment["moderate_risk"].append(hotspots[barangay]["name"])

    if risk_assessment["high_risk"]:
        warnings.append(
            {
                "message": "High accident risk detected in the following barangays:",
                "barangays": risk_assessment["high_risk"],
                "recommendations": [
                    "Increase patrols in high-risk areas.",
                    "Install additional warning signs or lights.",
                    "Consider road infrastructure improvements in these areas.",
                ],
            }
        )

    if risk_assessment["moderate_risk"]:
        warnings.append(
            {
                "message": "Moderate accident risk detected in the following barangays:",
                "barangays": risk_assessment["moderate_risk"],
                "recommendations": [
                    "Conduct community awareness programs on road safety.",
                    "Introduce speed management measures.",
                    "Increase visibility of traffic warnings and signs.",
                ],
            }
        )

    if not risk_assessment["high_risk"] and not risk_assessment["moderate_risk"]:
        if selected_barangay and selected_barangay in hotspots:
            barangay_names = [hotspots[selected_barangay]["name"]]
            message = f"Low accident risk detected in {hotspots[selected_barangay]['name']}."
        else:
            barangay_names = [info["name"] for info in hotspots.values()]
            message = "Low accident risk detected in all barangays."

        warnings.append(
            {
                "message": message,
                "barangays": barangay_names,
                "recommendations": [
                    "Maintain current safety measures.",
                    "Encourage community participation in safety programs.",
                ],
            }
        )


def generate_map(hotspots_data, average_people):
    map_object = folium.Map(location=[16.7000, 121.5600], zoom_start=12)
    marker_cluster = MarkerCluster().add_to(map_object)

    for barangay, info in hotspots_data.items():
        avg_people = average_people.get(barangay, 0)

        if avg_people <= 7:
            color = "green"
        elif 7 < avg_people <= 10:
            color = "orange"
        else:
            color = "red"

        popup_text = (
            f"{barangay}<br>"
            f"{info['name']}<br>"
            f"Average People Involved: {avg_people:.2f}"
        )

        tooltip_text = f"{barangay} - {info['name']} (Avg: {avg_people:.2f})"

        folium.Marker(
            location=[info["lat"], info["lon"]],
            popup=popup_text,
            tooltip=tooltip_text,
            icon=folium.Icon(color=color, icon="info-sign"),
        ).add_to(marker_cluster)

    map_path = os.path.join("templates", "map.html")
    map_object.save(map_path)

    return map_path


@app.route("/map")
def map_view():
    return render_template("map.html")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)