from flask import Flask, jsonify, send_from_directory
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

app = Flask(__name__, static_folder='static')  # Adjusted for root-level app.py
CORS(app)  # Enable CORS to allow frontend requests

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] - %(message)s',
    handlers=[
        logging.StreamHandler()  # Output logs to terminal
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# InfluxDB and Groq API configuration
INFLUXDB_URL = 'https://us-east-1-1.aws.cloud2.influxdata.com'
INFLUXDB_TOKEN = os.getenv('INFLUXDB_TOKEN')  # Moved to env variable
INFLUXDB_ORG = 'Agri'
INFLUXDB_BUCKET = 'smart_agri'
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
GROQ_API_URL = 'https://api.groq.com/openai/v1/chat/completions'

# Cache for recommendations (TTL of 300 seconds = 5 minutes)
recommendation_cache = TTLCache(maxsize=100, ttl=300)

# Request counter
request_counter = 0

def get_rain_status(value):
    if not value or value == 'null' or isinstance(value, float) and value != value:  # Check for None, 'null', or NaN
        return 'Unknown'
    try:
        value = int(float(value))  # Convert to float first to handle string numbers, then to int
        if value < 1500:
            return 'Heavy Rain'
        elif value < 3000:
            return 'Light Rain'
        return 'No Rain'
    except (ValueError, TypeError):
        return 'Unknown'

def fetch_sensor_data(range='-10m'):
    logger.info(f"Fetching sensor data for range {range}")
    if not INFLUXDB_TOKEN:
        logger.error("INFLUXDB_TOKEN is not set")
        return None

    query = f"""
        from(bucket: "{INFLUXDB_BUCKET}")
          |> range(start: {range})
          |> filter(fn: (r) => r._measurement == "sensor_data" and r.location == "field")
          |> last()
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
            logger.error(f"InfluxDB request failed for range {range}: Status {response.status_code} - {response.text}")
            return None

        text = response.text
        logger.debug(f"InfluxDB raw response: {text}")
        lines = text.split('\n')
        lines = [line for line in lines if line and not line.startswith('#')]
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
            if key and value and key not in ['result', 'table', 'location', '_time'] and value != 'null':
                cleaned_key = key.strip()
                if cleaned_key == 'wind_speed\r':
                    cleaned_key = 'wind_speed'
                try:
                    latest_data[cleaned_key] = float(value) if cleaned_key != 'motion_detected' else value
                except (ValueError, TypeError):
                    latest_data[cleaned_key] = value
            elif key == '_time':
                latest_data[key] = value

        logger.info(f"Successfully fetched sensor data for range {range}: {json.dumps(latest_data, indent=2)}")
        return latest_data
    except Exception as e:
        logger.error(f"Error fetching sensor data for range {range}: {str(e)}")
        return None

def get_grok_recommendations(data):
    if not data:
        logger.warning("No sensor data provided for recommendations")
        return {"pomegranate": "No data available to provide recommendations.", "guava": "No data available to provide recommendations."}

    if not GROQ_API_KEY:
        logger.error("GROQ_API_KEY is not set")
        return {"pomegranate": "GROQ_API_KEY is not set", "guava": "GROQ_API_KEY is not set"}

    # Check cache
    cache_key = json.dumps(data, sort_keys=True)
    if cache_key in recommendation_cache:
        logger.info(f"Cache hit for sensor data: {json.dumps(data, indent=2)}")
        return recommendation_cache[cache_key]

    logger.info(f"Cache miss, fetching new recommendations for sensor data: {json.dumps(data, indent=2)}")

    # Prepare sensor data for the prompt
    temp = data.get('temperature', 'unknown')
    humidity = data.get('humidity', 'unknown')
    soil_moisture = data.get('soil_moisture', 'unknown')
    rain_intensity = get_rain_status(data.get('rain_intensity', 'unknown'))
    wind_speed = data.get('wind_speed', 'unknown')
    motion_detected = 'detected' if data.get('motion_detected') == '1' else 'not detected'

    prompt = f"""
    You are an agricultural expert providing recommendations for Pomegranate and Guava plants based on the following sensor data from a farm:
    - Temperature: {temp} °C
    - Humidity: {humidity} %
    - Soil Moisture: {soil_moisture} %
    - Rainfall: {rain_intensity}
    - Wind Speed: {wind_speed} m/s
    - Motion Detected: {motion_detected}

    Provide specific, concise recommendations for each crop (Pomegranate and Guava) to optimize growth and health based on these conditions. Return the response in JSON format with keys 'pomegranate' and 'guava', each containing a string with recommendations.
    """
    logger.debug(f"Groq API prompt: {prompt}")

    try:
        logger.info("Sending request to Groq API")
        response = requests.post(GROQ_API_URL, headers={
            'Authorization': f'Bearer {GROQ_API_KEY}',
            'Content-Type': 'application/json'
        }, json={
            'model': 'llama-3.3-70b-versatile',
            'messages': [{'role': 'user', 'content': prompt}],
            'max_tokens': 500
        })

        if not response.ok:
            error_data = response.json().get('error', {})
            error_message = error_data.get('message', 'Unknown error')
            error_code = error_data.get('code', 'unknown')
            logger.error(f"Groq API request failed: Status {response.status_code} - {error_message} (code: {error_code})")
            return {"pomegranate": f"Failed to fetch recommendations: {error_message}", "guava": f"Failed to fetch recommendations: {error_message}"}

        result = response.json()
        recommendations = result['choices'][0]['message']['content']
        logger.info(f"Raw Groq API response: ```json\n{recommendations}\n```")
        
        # Strip Markdown code block markers if present
        cleaned_response = re.sub(r'^```json\n|\n```$', '', recommendations).strip()
        logger.info(f"Cleaned Groq API response: {cleaned_response}")
        
        try:
            parsed_recommendations = json.loads(cleaned_response)
            if not isinstance(parsed_recommendations, dict) or 'pomegranate' not in parsed_recommendations or 'guava' not in parsed_recommendations:
                raise ValueError("Invalid recommendation format: missing pomegranate or guava keys")
            recommendation_cache[cache_key] = parsed_recommendations
            logger.info(f"Cached recommendations for sensor data: {json.dumps(parsed_recommendations, indent=2)}")
            return parsed_recommendations
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Groq response as JSON: {str(e)}")
            return {"pomegranate": f"Error parsing recommendations: {str(e)}", "guava": f"Error parsing recommendations: {str(e)}"}
        except ValueError as e:
            logger.error(f"Invalid recommendation format: {str(e)}")
            return {"pomegranate": f"Error in recommendation format: {str(e)}", "guava": f"Error in recommendation format: {str(e)}"}
    except Exception as e:
        logger.error(f"Error fetching Groq recommendations: {str(e)}")
        return {"pomegranate": f"Error fetching recommendations: {str(e)}", "guava": f"Error fetching recommendations: {str(e)}"}

@app.route('/')
def serve_index():
    logger.info("Serving index.html")
    return send_from_directory('static', 'index.html')

@app.route('/api/recommendations', methods=['GET'])
def get_recommendations():
    global request_counter
    request_counter += 1
    logger.info(f"Handling /api/recommendations request #{request_counter} at {datetime.now().isoformat()}")
    
    try:
        ranges = ['-10m', '-1d', '-7d']
        latest_data = None
        range_used = None

        for range_time in ranges:
            logger.info(f"Attempting to fetch data for range {range_time}")
            latest_data = fetch_sensor_data(range_time)
            if latest_data:
                range_used = range_time
                logger.info(f"Data found for range {range_time}")
                break
            logger.warning(f"No data found for range {range_time}")
        
        recommendations = get_grok_recommendations(latest_data)
        response = {
            'data': latest_data if latest_data else {},
            'recommendations': recommendations
        }
        logger.info(f"Response for request #{request_counter}: data={json.dumps(latest_data, indent=2) if latest_data else {}}, recommendations={json.dumps(recommendations, indent=2)}")
        return jsonify(response)
    except Exception as e:
        logger.error(f"Error in /api/recommendations: {str(e)}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    logger.info("Health check requested")
    return jsonify({"status": "healthy", "request_count": request_counter})

application = app  # For Vercel WSGI compatibility

if __name__ == '__main__':
    logger.info("Starting Flask application on port 5000")
    app.run(debug=False, port=5000)