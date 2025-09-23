from flask import Flask, jsonify, send_from_directory, request
from flask_cors import CORS
import requests
import json
from dotenv import load_dotenv
import os
import csv
from io import StringIO
import re
import logging
from cachetools import TTLCache
from datetime import datetime

app = Flask(__name__, static_folder="../static")
CORS(app)  # Enable CORS to allow frontend requests

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# InfluxDB and Groq API configuration
INFLUXDB_URL = "https://us-east-1-1.aws.cloud2.influxdata.com"
INFLUXDB_TOKEN = "nZ49M1MTGbHtRCrc2OJhx-kVIBWuwvereT-o1mcq2COz3urUNuUuIIMjysObK8oOEHn8352w7LKFyrX8PQpdsA=="
INFLUXDB_ORG = "Agri"
INFLUXDB_BUCKET = "smart_agri"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

# Cache for recommendations (TTL of 300 seconds = 5 minutes)
recommendation_cache = TTLCache(maxsize=100, ttl=300)

# Request counter
request_counter = 0

def get_rain_status(value):
    if not value or value == "null" or isinstance(value, float) and value != value:
        return "Unknown"
    try:
        value = int(float(value))
        if value < 1500:
            return "Heavy Rain"
        elif value < 3000:
            return "Light Rain"
        return "No Rain"
    except (ValueError, TypeError):
        return "Unknown"

def fetch_sensor_data(range="-24h", interval="1h"):
    logger.info(f"Fetching sensor data for range {range} with interval {interval}")
    query = f"""
        from(bucket: "{INFLUXDB_BUCKET}")
          |> range(start: {range})
          |> filter(fn: (r) => r._measurement == "sensor_data" and r.location == "field")
          |> aggregateWindow(every: {interval}, fn: mean, createEmpty: false)
          |> pivot(rowKey: ["_time"], columnKey: ["_field"], valueColumn: "_value")
    """
    url = f"{INFLUXDB_URL}/api/v2/query?org={INFLUXDB_ORG}"

    try:
        response = requests.post(
            url,
            headers={
                "Authorization": f"Token {INFLUXDB_TOKEN}",
                "Content-Type": "application/vnd.flux",
                "Accept": "application/csv",
            },
            data=query,
        )

        if not response.ok:
            logger.error(f"InfluxDB request failed for range {range}: Status {response.status_code} - {response.text}")
            return None

        text = response.text
        lines = text.split("\n")
        lines = [line for line in lines if line and not line.startswith("#")]
        if len(lines) < 2:
            logger.warning(f"No data returned from InfluxDB for range {range}")
            return None

        reader = csv.DictReader(lines)
        data_points = []
        for row in reader:
            data = {}
            for key, value in row.items():
                if key and value and key not in ["result", "table", "location", "_time"] and value != "null":
                    cleaned_key = key.strip()
                    if cleaned_key == "wind_speed\r":
                        cleaned_key = "wind_speed"
                    try:
                        data[cleaned_key] = float(value) if cleaned_key != "motion_detected" else value
                    except (ValueError, TypeError):
                        data[cleaned_key] = value
                elif key == "_time":
                    data[key] = value
            if data:
                data_points.append(data)

        logger.info(f"Successfully fetched {len(data_points)} data points for range {range}")
        return data_points
    except Exception as e:
        logger.error(f"Error fetching sensor data for range {range}: {str(e)}")
        return None

def analyze_weather_trends(data_points):
    if not data_points or len(data_points) < 2:
        logger.warning("Insufficient data points for trend analysis")
        return None

    trends = {
        "temperature": {"avg": None, "trend": None, "min": None, "max": None},
        "humidity": {"avg": None, "trend": None, "min": None, "max": None},
        "soil_moisture": {"avg": None, "trend": None, "min": None, "max": None},
        "rain_intensity": {"heavy_rain_count": 0, "light_rain_count": 0, "no_rain_count": 0},
        "wind_speed": {"avg": None, "trend": None, "min": None, "max": None},
    }

    temp_values = [d["temperature"] for d in data_points if "temperature" in d and isinstance(d["temperature"], (int, float))]
    humidity_values = [d["humidity"] for d in data_points if "humidity" in d and isinstance(d["humidity"], (int, float))]
    soil_moisture_values = [d["soil_moisture"] for d in data_points if "soil_moisture" in d and isinstance(d["soil_moisture"], (int, float))]
    wind_speed_values = [d["wind_speed"] for d in data_points if "wind_speed" in d and isinstance(d["wind_speed"], (int, float))]
    rain_statuses = [get_rain_status(d.get("rain_intensity")) for d in data_points if "rain_intensity" in d]

    if temp_values:
        trends["temperature"]["avg"] = sum(temp_values) / len(temp_values)
        trends["temperature"]["min"] = min(temp_values)
        trends["temperature"]["max"] = max(temp_values)
        trends["temperature"]["trend"] = (
            "increasing" if temp_values[-1] > temp_values[0] else "decreasing" if temp_values[-1] < temp_values[0] else "stable"
        )

    if humidity_values:
        trends["humidity"]["avg"] = sum(humidity_values) / len(humidity_values)
        trends["humidity"]["min"] = min(humidity_values)
        trends["humidity"]["max"] = max(humidity_values)
        trends["humidity"]["trend"] = (
            "increasing" if humidity_values[-1] > humidity_values[0] else "decreasing" if humidity_values[-1] < humidity_values[0] else "stable"
        )

    if soil_moisture_values:
        trends["soil_moisture"]["avg"] = sum(soil_moisture_values) / len(soil_moisture_values)
        trends["soil_moisture"]["min"] = min(soil_moisture_values)
        trends["soil_moisture"]["max"] = max(soil_moisture_values)
        trends["soil_moisture"]["trend"] = (
            "increasing" if soil_moisture_values[-1] > soil_moisture_values[0]
            else "decreasing" if soil_moisture_values[-1] < soil_moisture_values[0]
            else "stable"
        )

    if wind_speed_values:
        trends["wind_speed"]["avg"] = sum(wind_speed_values) / len(wind_speed_values)
        trends["wind_speed"]["min"] = min(wind_speed_values)
        trends["wind_speed"]["max"] = max(wind_speed_values)
        trends["wind_speed"]["trend"] = (
            "increasing" if wind_speed_values[-1] > wind_speed_values[0]
            else "decreasing" if wind_speed_values[-1] < wind_speed_values[0]
            else "stable"
        )

    trends["rain_intensity"]["heavy_rain_count"] = rain_statuses.count("Heavy Rain")
    trends["rain_intensity"]["light_rain_count"] = rain_statuses.count("Light Rain")
    trends["rain_intensity"]["no_rain_count"] = rain_statuses.count("No Rain")

    logger.info(f"Weather trends: {json.dumps(trends, indent=2)}")
    return trends

def get_grok_recommendations(data_points, trends):
    if not data_points:
        logger.warning("No sensor data provided for recommendations")
        return {"pomegranate": "No data available to provide recommendations.", "guava": "No data available to provide recommendations."}

    latest_data = data_points[-1] if data_points else {}
    cache_key = json.dumps({"latest": latest_data, "trends": trends}, sort_keys=True)
    if cache_key in recommendation_cache:
        logger.info(f"Cache hit for sensor data and trends")
        return recommendation_cache[cache_key]

    logger.info(f"Cache miss, fetching new recommendations")

    temp = latest_data.get("temperature", "unknown")
    humidity = latest_data.get("humidity", "unknown")
    soil_moisture = latest_data.get("soil_moisture", "unknown")
    rain_intensity = get_rain_status(latest_data.get("rain_intensity", "unknown"))
    wind_speed = latest_data.get("wind_speed", "unknown")
    motion_detected = "detected" if latest_data.get("motion_detected") == "1" else "not detected"

    trend_summary = ""
    if trends:
        trend_summary = f"""
        24-Hour Weather Trends:
        - Temperature: Avg {trends['temperature']['avg']:.1f}°C ({trends['temperature']['trend']}, Min: {trends['temperature']['min']:.1f}°C, Max: {trends['temperature']['max']:.1f}°C)
        - Humidity: Avg {trends['humidity']['avg']:.1f}% ({trends['humidity']['trend']}, Min: {trends['humidity']['min']:.1f}%, Max: {trends['humidity']['max']:.1f}%)
        - Soil Moisture: Avg {trends['soil_moisture']['avg']:.1f}% ({trends['soil_moisture']['trend']}, Min: {trends['soil_moisture']['min']:.1f}%, Max: {trends['soil_moisture']['max']:.1f}%)
        - Wind Speed: Avg {trends['wind_speed']['avg']:.1f} m/s ({trends['wind_speed']['trend']}, Min: {trends['wind_speed']['min']:.1f} m/s, Max: {trends['wind_speed']['max']:.1f} m/s)
        - Rainfall: {trends['rain_intensity']['heavy_rain_count']} heavy rain, {trends['rain_intensity']['light_rain_count']} light rain, {trends['rain_intensity']['no_rain_count']} no rain periods
        """

    prompt = f"""
    You are an agricultural expert providing recommendations for Pomegranate in Flowering stage and Guava in Fruiting stage, based on the following current sensor data and 24-hour weather trends from a farm:

    Current Conditions:
    - Temperature: {temp} °C
    - Humidity: {humidity} %
    - Soil Moisture: {soil_moisture} %
    - Rainfall: {rain_intensity}
    - Wind Speed: {wind_speed} m/s
    - Motion Detected: {motion_detected}

    {trend_summary}

    Provide specific, concise recommendations for each crop (Pomegranate and Guava) to optimize growth and health based on current conditions and 24-hour trends. Additionally, include a brief interpretation of the 24-hour weather patterns and their potential impact on these crops. Return the response in JSON format with keys 'pomegranate' and 'guava', each containing a string with recommendations and trend interpretation.
    """
    logger.debug(f"Groq API prompt: {prompt}")

    try:
        logger.info("Sending request to Groq API")
        response = requests.post(
            GROQ_API_URL,
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": "llama-3.3-70b-versatile",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 700,
            },
        )

        if not response.ok:
            error_data = response.json().get("error", {})
            error_message = error_data.get("message", "Unknown error")
            error_code = error_data.get("code", "unknown")
            logger.error(f"Groq API request failed: Status {response.status_code} - {error_message} (code: {error_code})")
            return {
                "pomegranate": f"Failed to fetch recommendations: {error_message}",
                "guava": f"Failed to fetch recommendations: {error_message}",
            }

        result = response.json()
        recommendations = result["choices"][0]["message"]["content"]
        logger.info(f"Raw Groq API response: ```json\n{recommendations}\n```")

        cleaned_response = re.sub(r"^```json\n|\n```$", "", recommendations.strip())

        try:
            parsed_recommendations = json.loads(cleaned_response)
            if not isinstance(parsed_recommendations, dict) or "pomegranate" not in parsed_recommendations or "guava" not in parsed_recommendations:
                raise ValueError("Invalid JSON structure: missing 'pomegranate' or 'guava' keys")
            recommendation_cache[cache_key] = parsed_recommendations
            logger.info("Successfully parsed and cached recommendations")
            return parsed_recommendations
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Groq API response as JSON: {str(e)}")
            return {
                "pomegranate": f"Error parsing recommendations: {str(e)}",
                "guava": f"Error parsing recommendations: {str(e)}",
            }
        except ValueError as e:
            logger.error(f"Invalid response structure: {str(e)}")
            return {"pomegranate": f"Invalid response: {str(e)}", "guava": f"Invalid response: {str(e)}"}
    except Exception as e:
        logger.error(f"Error fetching recommendations from Groq API: {str(e)}")
        return {
            "pomegranate": f"Failed to fetch recommendations: {str(e)}",
            "guava": f"Failed to fetch recommendations: {str(e)}",
        }

@app.route("/")
def serve_index():
    global request_counter
    request_counter += 1
    logger.info(f"Serving index.html - Request #{request_counter} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    return send_from_directory(app.static_folder, "index.html")

@app.route("/api/recommendations")
def get_recommendations():
    global request_counter
    request_counter += 1
    logger.info(f"Handling /api/recommendations request #{request_counter} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    time_range = request.args.get("range", "-24h")
    interval = "1h"

    data_points = fetch_sensor_data(time_range, interval)
    if not data_points:
        logger.warning(f"No data in range {time_range}, trying fallback range -7d")
        data_points = fetch_sensor_data("-7d", "1d")
        time_range = "-7d" if data_points else time_range

    trends = analyze_weather_trends(data_points) if data_points else None
    recommendations = get_grok_recommendations(data_points, trends)

    return jsonify({
        "recommendations": recommendations,
        "data": data_points or [],
        "trends": trends or {},
        "time_range": time_range,
    })

@app.route("/api/trends")
def get_trends():
    global request_counter
    request_counter += 1
    logger.info(f"Handling /api/trends request #{request_counter} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    time_range = request.args.get("range", "-24h")
    interval = "1h"

    data_points = fetch_sensor_data(time_range, interval)
    if not data_points:
        logger.warning(f"No data in range {time_range}")
        return jsonify({"error": "No data available for the specified time range", "data": [], "trends": {}}), 404

    trends = analyze_weather_trends(data_points)
    return jsonify({
        "data": data_points,
        "trends": trends or {},
        "time_range": time_range,
    })

@app.route("/api/ask", methods=["POST"])
def ask_crop_question():
    global request_counter
    request_counter += 1
    logger.info(f"Handling /api/ask request #{request_counter} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    try:
        request_data = request.get_json()
        if not request_data or "prompt" not in request_data:
            logger.error("Invalid request: Missing 'prompt' in request body")
            return jsonify({"error": "Missing prompt in request body"}), 400

        prompt = request_data["prompt"]
        logger.debug(f"Received prompt: {prompt}")

        response = requests.post(
            GROQ_API_URL,
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": "llama-3.3-70b-versatile",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 500,
            },
        )

        if not response.ok:
            error_data = response.json().get("error", {})
            error_message = error_data.get("message", "Unknown error")
            error_code = error_data.get("code", "unknown")
            logger.error(f"Groq API request failed: Status {response.status_code} - {error_message} (code: {error_code})")
            return jsonify({"error": f"Groq API error: {error_message}"}), response.status_code

        result = response.json()
        response_text = result["choices"][0]["message"]["content"]
        logger.info(f"Groq API response: {response_text}")
        return jsonify({"response": response_text})

    except Exception as e:
        logger.error(f"Error processing /api/ask request: {str(e)}")
        return jsonify({"error": f"Failed to process request: {str(e)}"}), 500

if __name__ == "__main__":
    logger.info("Starting Flask application")
    app.run(debug=True, host="0.0.0.0", port=5000)