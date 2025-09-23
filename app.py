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
    handlers=[logging.StreamHandler()]  # Output logs to terminal
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
    if not value or value == "null" or isinstance(value, float) and value != value:  # Check for None, 'null', or NaN
        return "Unknown"
    try:
        value = int(float(value))  # Convert to float first to handle string numbers, then to int
        if value < 1500:
            return "Heavy Rain"
        elif value < 3000:
            return "Light Rain"
        return "No Rain"
    except (ValueError, TypeError):
        return "Unknown"

def fetch_sensor_data(range="-10m"):
    logger.info(f"Fetching sensor data for range {range}")
    query = f"""
        from(bucket: "{INFLUXDB_BUCKET}")
          |> range(start: {range})
          |> filter(fn: (r) => r._measurement == "sensor_data" and r.location == "field")
          |> last()
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
        data = next(reader, None)
        if not data:
            logger.warning(f"No valid data rows in InfluxDB response for range {range}")
            return None

        latest_data = {}
        for key, value in data.items():
            if key and value and key not in ["result", "table", "location", "_time"] and value != "null":
                cleaned_key = key.strip()
                if cleaned_key == "wind_speed\r":
                    cleaned_key = "wind_speed"
                try:
                    latest_data[cleaned_key] = float(value) if cleaned_key != "motion_detected" else value
                except (ValueError, TypeError):
                    latest_data[cleaned_key] = value
            elif key == "_time":
                latest_data[key] = value

        logger.info(f"Successfully fetched sensor data for range {range}: {json.dumps(latest_data, indent=2)}")
        return latest_data
    except Exception as e:
        logger.error(f"Error fetching sensor data for range {range}: {str(e)}")
        return None

def fetch_historical_data(range="-24h", aggregate_window="1h"):
    logger.info(f"Fetching historical sensor data for range {range} with aggregation window {aggregate_window}")
    query = f"""
        from(bucket: "{INFLUXDB_BUCKET}")
          |> range(start: {range})
          |> filter(fn: (r) => r._measurement == "sensor_data" and r.location == "field")
          |> aggregateWindow(every: {aggregate_window}, fn: mean, createEmpty: false)
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
            logger.error(f"InfluxDB historical request failed for range {range}: Status {response.status_code} - {response.text}")
            return []

        text = response.text
        lines = text.split("\n")
        lines = [line for line in lines if line and not line.startswith("#")]
        if len(lines) < 2:
            logger.warning(f"No historical data returned from InfluxDB for range {range}")
            return []

        reader = csv.DictReader(lines)
        historical_data = []
        for row in reader:
            data_point = {}
            for key, value in row.items():
                if key and value and key not in ["result", "table", "location"] and value != "null":
                    cleaned_key = key.strip()
                    if cleaned_key == "wind_speed\r":
                        cleaned_key = "wind_speed"
                    try:
                        data_point[cleaned_key] = float(value) if cleaned_key != "motion_detected" else value
                    except (ValueError, TypeError):
                        data_point[cleaned_key] = value
                elif key == "_time":
                    data_point[key] = value
            historical_data.append(data_point)

        logger.info(f"Successfully fetched {len(historical_data)} historical data points for range {range}")
        return historical_data
    except Exception as e:
        logger.error(f"Error fetching historical data for range {range}: {str(e)}")
        return []

def get_grok_recommendations(data_list):
    if not data_list or not isinstance(data_list, list) or len(data_list) == 0:
        logger.warning("No historical sensor data provided for recommendations")
        return {"pomegranate": "No data available to provide recommendations.", "guava": "No data available to provide recommendations."}

    # Check cache
    cache_key = json.dumps(data_list, sort_keys=True)
    if cache_key in recommendation_cache:
        logger.info(f"Cache hit for historical sensor data")
        return recommendation_cache[cache_key]

    logger.info(f"Cache miss, fetching new recommendations for historical sensor data")

    # Aggregate data for the prompt
    temp_avg = sum(d.get("temperature", 0) for d in data_list if isinstance(d.get("temperature"), (int, float))) / max(
        1, sum(1 for d in data_list if isinstance(d.get("temperature"), (int, float)))
    )
    humidity_avg = sum(d.get("humidity", 0) for d in data_list if isinstance(d.get("humidity"), (int, float))) / max(
        1, sum(1 for d in data_list if isinstance(d.get("humidity"), (int, float)))
    )
    soil_moisture_avg = sum(d.get("soil_moisture", 0) for d in data_list if isinstance(d.get("soil_moisture"), (int, float))) / max(
        1, sum(1 for d in data_list if isinstance(d.get("soil_moisture"), (int, float)))
    )
    rain_values = [d.get("rain_intensity") for d in data_list if d.get("rain_intensity") and d.get("rain_intensity") != "null"]
    rain_status = (
        get_rain_status(max(rain_values, key=lambda x: int(float(x)) if x and x != "null" else 0))
        if rain_values
        else "Unknown"
    )
    wind_speed_avg = sum(d.get("wind_speed", 0) for d in data_list if isinstance(d.get("wind_speed"), (int, float))) / max(
        1, sum(1 for d in data_list if isinstance(d.get("wind_speed"), (int, float)))
    )
    motion_detected = "detected" if any(d.get("motion_detected") == "1" for d in data_list) else "not detected"

    prompt = f"""
    You are an agricultural expert providing recommendations for Pomegranate in Flowering stage and Guava plants in Fruiting stage, based on the following average sensor data from a farm over the last 24 hours:
    - Temperature: {temp_avg:.1f} °C
    - Humidity: {humidity_avg:.1f} %
    - Soil Moisture: {soil_moisture_avg:.1f} %
    - Rainfall: {rain_status}
    - Wind Speed: {wind_speed_avg:.2f} m/s
    - Motion Detected: {motion_detected}

    Provide specific, concise recommendations for each crop (Pomegranate and Guava) to optimize growth and health based on these conditions. Return the response in JSON format with keys 'pomegranate' and 'guava', each containing a string with recommendations.
    """
    logger.debug(f"Groq API prompt: {prompt}")

    try:
        logger.info("Sending request to Groq API")
        response = requests.post(
            GROQ_API_URL,
            headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
            json={"model": "llama-3.3-70b-versatile", "messages": [{"role": "user", "content": prompt}], "max_tokens": 500},
        )

        if not response.ok:
            error_data = response.json().get("error", {})
            error_message = error_data.get("message", "Unknown error")
            error_code = error_data.get("code", "unknown")
            logger.error(f"Groq API request failed: Status {response.status_code} - {error_message} (code: {error_code})")
            return {"pomegranate": f"Failed to fetch recommendations: {error_message}", "guava": f"Failed to fetch recommendations: {error_message}"}

        result = response.json()
        recommendations = result["choices"][0]["message"]["content"]
        logger.info(f"Raw Groq API response: ```json\n{recommendations}\n```")

        # Strip Markdown code block markers if present
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
            return {"pomegranate": f"Error parsing recommendations: {str(e)}", "guava": f"Error parsing recommendations: {str(e)}"}
        except ValueError as e:
            logger.error(f"Invalid response structure: {str(e)}")
            return {"pomegranate": f"Invalid response: {str(e)}", "guava": f"Invalid response: {str(e)}"}
    except Exception as e:
        logger.error(f"Error fetching recommendations from Groq API: {str(e)}")
        return {"pomegranate": f"Failed to fetch recommendations: {str(e)}", "guava": f"Failed to fetch recommendations: {str(e)}"}

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

    # Fetch historical data for the last 24 hours
    historical_data = fetch_historical_data("-24h", "1h")
    if not historical_data:
        logger.warning("No historical data available for last 24 hours, falling back to recent data")
        data = fetch_sensor_data("-10m") or fetch_sensor_data("-1d") or fetch_sensor_data("-7d")
        recommendations = get_grok_recommendations([data] if data else [])
        return jsonify({"recommendations": recommendations, "data": data or {}})

    recommendations = get_grok_recommendations(historical_data)
    return jsonify({"recommendations": recommendations, "data": historical_data})

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

        # Fetch historical data for the last 7 days
        historical_data = fetch_historical_data("-7d", "6h")
        if not historical_data:
            logger.warning("No historical data available for last 7 days")
            data = fetch_sensor_data("-10m") or fetch_sensor_data("-1d") or fetch_sensor_data("-7d")
            historical_data = [data] if data else []

        # Aggregate data for the prompt
        temp_avg = sum(d.get("temperature", 0) for d in historical_data if isinstance(d.get("temperature"), (int, float))) / max(
            1, sum(1 for d in historical_data if isinstance(d.get("temperature"), (int, float)))
        )
        humidity_avg = sum(d.get("humidity", 0) for d in historical_data if isinstance(d.get("humidity"), (int, float))) / max(
            1, sum(1 for d in historical_data if isinstance(d.get("humidity"), (int, float)))
        )
        soil_moisture_avg = sum(d.get("soil_moisture", 0) for d in historical_data if isinstance(d.get("soil_moisture"), (int, float))) / max(
            1, sum(1 for d in historical_data if isinstance(d.get("soil_moisture"), (int, float)))
        )
        rain_values = [d.get("rain_intensity") for d in historical_data if d.get("rain_intensity") and d.get("rain_intensity") != "null"]
        rain_status = (
            get_rain_status(max(rain_values, key=lambda x: int(float(x)) if x and x != "null" else 0))
            if rain_values
            else "Unknown"
        )
        wind_speed_avg = sum(d.get("wind_speed", 0) for d in historical_data if isinstance(d.get("wind_speed"), (int, float))) / max(
            1, sum(1 for d in historical_data if isinstance(d.get("wind_speed"), (int, float)))
        )
        motion_detected = "detected" if any(d.get("motion_detected") == "1" for d in historical_data) else "not detected"

        enhanced_prompt = f"""
        You are an agricultural expert providing advice based on the following average sensor data from a farm over the last 7 days:
        - Temperature: {temp_avg:.1f} °C
        - Humidity: {humidity_avg:.1f} %
        - Soil Moisture: {soil_moisture_avg:.1f} %
        - Rainfall: {rain_status}
        - Wind Speed: {wind_speed_avg:.2f} m/s
        - Motion Detected: {motion_detected}

        The user has asked: "{prompt}"

        Provide a concise, specific response to the user's question, considering the provided sensor data and its impact on crop growth and health.
        """
        logger.debug(f"Received prompt: {prompt}")
        logger.debug(f"Enhanced prompt: {enhanced_prompt}")

        response = requests.post(
            GROQ_API_URL,
            headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
            json={"model": "llama-3.3-70b-versatile", "messages": [{"role": "user", "content": enhanced_prompt}], "max_tokens": 500},
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