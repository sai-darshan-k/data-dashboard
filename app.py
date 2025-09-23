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
import pandas as pd
import numpy as np

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

def fetch_24hr_aggregated_data():
    """Fetch aggregated sensor data from the last 24 hours"""
    logger.info("Fetching aggregated sensor data for last 24 hours")
    
    query = f"""
        from(bucket: "{INFLUXDB_BUCKET}")
          |> range(start: -24h)
          |> filter(fn: (r) => r._measurement == "sensor_data" and r.location == "field")
          |> group(columns: ["_field"])
          |> mean()
          |> pivot(rowKey: ["_time"], columnKey: ["_field"], valueColumn: "_value")
    """
    
    url = f"{INFLUXDB_URL}/api/v2/query?org={INFLUXDB_ORG}"
    
    try:
        response = requests.post(url, headers={
            'Authorization': f'Token {INFLUXDB_TOKEN}',
            'Content-Type': 'application/vnd.flux',
            'Accept': 'application/csv'
        }, data=query)

        if not response.ok:
            logger.error(f"InfluxDB request failed for 24hr data: Status {response.status_code} - {response.text}")
            return None

        text = response.text
        lines = text.split("\n")
        lines = [line for line in lines if line and not line.startswith("#")]
        if len(lines) < 2:
            logger.warning("No 24hr aggregated data returned from InfluxDB")
            return None

        reader = csv.DictReader(lines)
        data = next(reader, None)
        if not data:
            logger.warning("No valid 24hr data rows in InfluxDB response")
            return None

        aggregated_data = {}
        for key, value in data.items():
            if key and value and key not in ['result', 'table', 'location', '_time'] and value != 'null':
                cleaned_key = key.strip()
                if cleaned_key == 'wind_speed\r':
                    cleaned_key = 'wind_speed'
                try:
                    aggregated_data[cleaned_key] = round(float(value), 2) if cleaned_key != 'motion_detected' else value
                except (ValueError, TypeError):
                    aggregated_data[cleaned_key] = value

        logger.info(f"Successfully fetched 24hr aggregated data: {json.dumps(aggregated_data, indent=2)}")
        return aggregated_data
        
    except Exception as e:
        logger.error(f"Error fetching 24hr aggregated sensor data: {str(e)}")
        return None

def fetch_current_sensor_data():
    """Fetch the most recent sensor reading"""
    logger.info("Fetching current sensor data")
    return fetch_sensor_data('-10m')

def fetch_historical_data(range="-7d", aggregate_window="6h"):
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

def analyze_historical_data(data, metric, operation):
    if not data:
        return None
    df = pd.DataFrame(data)
    df = df[df[metric].notnull() & (df[metric] != "null")]
    if df.empty:
        return None
    try:
        df[metric] = df[metric].astype(float)
        if operation == "min":
            return df[metric].min()
        elif operation == "max":
            return df[metric].max()
        elif operation == "avg":
            return df[metric].mean()
        else:
            return df[metric].mean()  # Default to average
    except (ValueError, TypeError) as e:
        logger.error(f"Error analyzing metric {metric}: {str(e)}")
        return None

def get_grok_recommendations_24hr(aggregated_data, current_data=None):
    if not aggregated_data:
        logger.warning("No 24hr sensor data provided for recommendations")
        return {"pomegranate": "No 24-hour data available to provide recommendations.", 
                "guava": "No 24-hour data available to provide recommendations."}

    cache_key = json.dumps({
        'aggregated': aggregated_data,
        'current': current_data or {}
    }, sort_keys=True)
    
    if cache_key in recommendation_cache:
        logger.info("Cache hit for 24hr recommendations")
        return recommendation_cache[cache_key]

    logger.info("Cache miss, fetching new 24hr recommendations")

    temp_avg = aggregated_data.get('temperature', 'unknown')
    humidity_avg = aggregated_data.get('humidity', 'unknown')
    soil_moisture_avg = aggregated_data.get('soil_moisture', 'unknown')
    rain_intensity_avg = aggregated_data.get('rain_intensity', 'unknown')
    wind_speed_avg = aggregated_data.get('wind_speed', 'unknown')
    
    current_info = ""
    if current_data:
        current_temp = current_data.get('temperature', 'unknown')
        current_humidity = current_data.get('humidity', 'unknown')
        current_soil = current_data.get('soil_moisture', 'unknown')
        current_motion = 'detected' if current_data.get('motion_detected') == '1' else 'not detected'
        current_rain = get_rain_status(current_data.get('rain_intensity', 'unknown'))
        
        current_info = f"""
    
    Current conditions (last 10 minutes):
    - Current Temperature: {current_temp} °C
    - Current Humidity: {current_humidity} %
    - Current Soil Moisture: {current_soil} %
    - Current Rainfall: {current_rain}
    - Motion Detected: {current_motion}"""

    if isinstance(rain_intensity_avg, (int, float)):
        rain_status = get_rain_status(rain_intensity_avg)
    else:
        rain_status = rain_intensity_avg

    prompt = f"""
    You are an agricultural expert providing recommendations for Pomegranate (Flowering stage) and Guava (Fruiting stage) based on sensor data from a farm.
    
    24-hour average conditions:
    - Average Temperature: {temp_avg} °C
    - Average Humidity: {humidity_avg} %
    - Average Soil Moisture: {soil_moisture_avg} %
    - Average Rainfall Status: {rain_status}
    - Average Wind Speed: {wind_speed_avg} m/s{current_info}

    Based on these 24-hour patterns and current conditions, provide specific, actionable recommendations for each crop to optimize growth and health. Consider:
    - Irrigation needs based on soil moisture trends
    - Disease prevention based on humidity patterns
    - Weather protection measures
    - Optimal timing for agricultural activities
    
    Return the response in JSON format with keys 'pomegranate' and 'guava', each containing a string with detailed recommendations.
    """
    
    logger.debug(f"24hr Groq API prompt: {prompt}")

    try:
        logger.info("Sending 24hr recommendation request to Groq API")
        response = requests.post(GROQ_API_URL, headers={
            'Authorization': f'Bearer {GROQ_API_KEY}',
            'Content-Type': 'application/json'
        }, json={
            'model': 'llama-3.3-70b-versatile',
            'messages': [{'role': 'user', 'content': prompt}],
            'max_tokens': 600
        })

        if not response.ok:
            error_data = response.json().get('error', {})
            error_message = error_data.get('message', 'Unknown error')
            logger.error(f"Groq API request failed: {error_message}")
            return {"pomegranate": f"Failed to fetch recommendations: {error_message}", 
                    "guava": f"Failed to fetch recommendations: {error_message}"}

        result = response.json()
        recommendations = result['choices'][0]['message']['content']
        
        cleaned_response = re.sub(r'^```json\n|\n```$', '', recommendations.strip())
        
        try:
            parsed_recommendations = json.loads(cleaned_response)
            if not isinstance(parsed_recommendations, dict) or 'pomegranate' not in parsed_recommendations or 'guava' not in parsed_recommendations:
                raise ValueError("Invalid JSON structure: missing 'pomegranate' or 'guava' keys")
            
            recommendation_cache[cache_key] = parsed_recommendations
            logger.info("Successfully parsed and cached 24hr recommendations")
            return parsed_recommendations
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Groq API response as JSON: {str(e)}")
            return {"pomegranate": f"Error parsing recommendations: {str(e)}", 
                    "guava": f"Error parsing recommendations: {str(e)}"}
                    
    except Exception as e:
        logger.error(f"Error fetching 24hr recommendations from Groq API: {str(e)}")
        return {"pomegranate": f"Failed to fetch recommendations: {str(e)}", 
                "guava": f"Failed to fetch recommendations: {str(e)}"}

@app.route("/")
def serve_index():
    global request_counter
    request_counter += 1
    logger.info(f"Serving index.html - Request #{request_counter} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    return send_from_directory(app.static_folder, "index.html")

@app.route('/api/recommendations')
def get_recommendations():
    global request_counter
    request_counter += 1
    logger.info(f"Handling /api/recommendations request #{request_counter} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    aggregated_data = fetch_24hr_aggregated_data()
    current_data = fetch_current_sensor_data()
    
    if not aggregated_data:
        logger.warning("No 24hr data available, falling back to recent data")
        if current_data:
            recommendations = get_grok_recommendations_24hr(current_data)
        else:
            fallback_data = fetch_sensor_data('-1d')
            if not fallback_data:
                fallback_data = fetch_sensor_data('-7d')
            recommendations = get_grok_recommendations_24hr(fallback_data) if fallback_data else {
                "pomegranate": "No sensor data available for recommendations.",
                "guava": "No sensor data available for recommendations."
            }
    else:
        recommendations = get_grok_recommendations_24hr(aggregated_data, current_data)
    
    return jsonify({
        'recommendations': recommendations,
        'aggregated_data': aggregated_data or {},
        'current_data': current_data or {},
        'data_source': '24hr_average' if aggregated_data else 'fallback'
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

        prompt = request_data["prompt"].lower()
        logger.debug(f"Received prompt: {prompt}")

        # Extract time range from prompt
        time_range = "-7d"  # Default to 7 days
        aggregate_window = "6h"
        if "last 24 hours" in prompt or "past 24 hours" in prompt:
            time_range = "-24h"
            aggregate_window = "1h"
        elif "last 3 days" in prompt or "past 3 days" in prompt:
            time_range = "-3d"
            aggregate_window = "3h"
        elif "last month" in prompt or "past month" in prompt:
            time_range = "-30d"
            aggregate_window = "12h"

        # Fetch historical data for the specified time range
        historical_data = fetch_historical_data(time_range, aggregate_window)
        if not historical_data:
            logger.warning(f"No historical data available for {time_range}, falling back to recent data")
            current_data = fetch_current_sensor_data()
            historical_data = [current_data] if current_data else []

        # Analyze data for specific metrics
        metric = None
        operation = None
        metric_value = None
        if "temperature" in prompt:
            metric = "temperature"
            if "minimum" in prompt or "min" in prompt:
                operation = "min"
                metric_value = analyze_historical_data(historical_data, "temperature", "min")
            elif "maximum" in prompt or "max" in prompt:
                operation = "max"
                metric_value = analyze_historical_data(historical_data, "temperature", "max")
            elif "average" in prompt or "avg" in prompt:
                operation = "avg"
                metric_value = analyze_historical_data(historical_data, "temperature", "avg")
        elif "humidity" in prompt:
            metric = "humidity"
            if "minimum" in prompt or "min" in prompt:
                operation = "min"
                metric_value = analyze_historical_data(historical_data, "humidity", "min")
            elif "maximum" in prompt or "max" in prompt:
                operation = "max"
                metric_value = analyze_historical_data(historical_data, "humidity", "max")
            elif "average" in prompt or "avg" in prompt:
                operation = "avg"
                metric_value = analyze_historical_data(historical_data, "humidity", "avg")
        elif "soil moisture" in prompt:
            metric = "soil_moisture"
            if "minimum" in prompt or "min" in prompt:
                operation = "min"
                metric_value = analyze_historical_data(historical_data, "soil_moisture", "min")
            elif "maximum" in prompt or "max" in prompt:
                operation = "max"
                metric_value = analyze_historical_data(historical_data, "soil_moisture", "max")
            elif "average" in prompt or "avg" in prompt:
                operation = "avg"
                metric_value = analyze_historical_data(historical_data, "soil_moisture", "avg")

        # Aggregate data for context
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

        # Fetch current data for additional context
        current_data = fetch_current_sensor_data()
        current_info = ""
        if current_data:
            current_temp = current_data.get('temperature', 'unknown')
            current_humidity = current_data.get('humidity', 'unknown')
            current_soil = current_data.get('soil_moisture', 'unknown')
            current_motion = 'detected' if current_data.get('motion_detected') == '1' else 'not detected'
            current_rain = get_rain_status(current_data.get('rain_intensity', 'unknown'))
            current_info = f"""
    
    Current conditions (last 10 minutes):
    - Current Temperature: {current_temp} °C
    - Current Humidity: {current_humidity} %
    - Current Soil Moisture: {current_soil} %
    - Current Rainfall: {current_rain}
    - Motion Detected: {current_motion}"""

        # Construct RAG-style prompt
        data_context = f"""
        Sensor data summary for the last {time_range.replace('-', '')}:
        - Average Temperature: {temp_avg:.1f} °C
        - Average Humidity: {humidity_avg:.1f} %
        - Average Soil Moisture: {soil_moisture_avg:.1f} %
        - Rainfall: {rain_status}
        - Average Wind Speed: {wind_speed_avg:.2f} m/s
        - Motion Detected: {motion_detected}{current_info}
        """
        if metric and metric_value is not None:
            unit = '°C' if metric == 'temperature' else '%' if metric in ['humidity', 'soil_moisture'] else ''
            data_context += f"- {metric.replace('_', ' ').title()} {operation.capitalize()}: {metric_value:.1f} {unit}\n"

        enhanced_prompt = f"""
        You are an agricultural expert providing advice for crop management based on historical sensor data. The user has asked: "{request_data['prompt']}"

        Below is the relevant sensor data for the specified time range:
        {data_context}

        Provide a concise, specific response to the user's question, using the provided sensor data to ground your answer. Focus on the crop mentioned in the question (Pomegranate in Flowering stage or Guava in Fruiting stage) and ensure the response is accurate and relevant to the data.
        """
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