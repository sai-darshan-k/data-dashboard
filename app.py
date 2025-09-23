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
from datetime import datetime, timedelta
import statistics

app = Flask(__name__, static_folder='../static')
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
INFLUXDB_TOKEN = 'nZ49M1MTGbHtRCrc2OJhx-kVIBWuwvereT-o1mcq2COz3urUNuUuIIMjysObK8oOEHn8352w7LKFyrX8PQpdsA=='
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

def fetch_historical_24h_data():
    """Fetch 24-hour historical data for analysis"""
    logger.info("Fetching 24-hour historical data for analysis")
    
    # Calculate 24 hours ago
    end_time = datetime.now()
    start_time = end_time - timedelta(hours=24)
    
    query = f"""
        from(bucket: "{INFLUXDB_BUCKET}")
          |> range(start: {start_time.isoformat()}Z, stop: {end_time.isoformat()}Z)
          |> filter(fn: (r) => r._measurement == "sensor_data" and r.location == "field")
          |> aggregateWindow(every: 1h, fn: mean, createEmpty: false)
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
            logger.error(f"InfluxDB historical request failed: Status {response.status_code} - {response.text}")
            return []

        text = response.text
        lines = text.split('\n')
        lines = [line for line in lines if line and not line.startswith('#')]
        
        if len(lines) < 2:
            logger.warning("No historical data returned from InfluxDB")
            return []

        # Parse CSV data
        historical_data = []
        header = lines[0].split(',')
        
        for line in lines[1:]:
            if not line.strip():
                continue
            values = line.split(',')
            data_point = {}
            for i, value in enumerate(values):
                if i < len(header):
                    key = header[i].strip()
                    if key == 'wind_speed\r':
                        key = 'wind_speed'
                    
                    if value and value.strip() and value != 'null':
                        try:
                            if key == '_time':
                                data_point[key] = value.strip()
                            elif key in ['temperature', 'humidity', 'soil_moisture', 'wind_speed', 'rain_intensity']:
                                data_point[key] = float(value.strip())
                            elif key == 'motion_detected':
                                data_point[key] = value.strip()
                            elif key not in ['result', 'table', 'location']:
                                data_point[key] = value.strip()
                        except (ValueError, TypeError):
                            continue
            
            if data_point and '_time' in data_point:
                historical_data.append(data_point)
        
        logger.info(f"Successfully fetched {len(historical_data)} historical data points")
        return historical_data
        
    except Exception as e:
        logger.error(f"Error fetching 24-hour historical data: {str(e)}")
        return []

def analyze_historical_trends(historical_data):
    """Analyze 24-hour trends and patterns"""
    if not historical_data:
        return "No historical data available for trend analysis."
    
    try:
        # Extract values for analysis
        temps = [d['temperature'] for d in historical_data if 'temperature' in d and d['temperature'] is not None]
        humidities = [d['humidity'] for d in historical_data if 'humidity' in d and d['humidity'] is not None]
        soil_moistures = [d['soil_moisture'] for d in historical_data if 'soil_moisture' in d and d['soil_moisture'] is not None]
        wind_speeds = [d['wind_speed'] for d in historical_data if 'wind_speed' in d and d['wind_speed'] is not None]
        rain_intensities = [d['rain_intensity'] for d in historical_data if 'rain_intensity' in d and d['rain_intensity'] is not None]
        
        trends = []
        
        # Temperature trends
        if len(temps) >= 2:
            temp_trend = "increasing" if temps[-1] > temps[0] else "decreasing" if temps[-1] < temps[0] else "stable"
            avg_temp = statistics.mean(temps)
            max_temp = max(temps)
            min_temp = min(temps)
            trends.append(f"Temperature: {temp_trend} trend, avg {avg_temp:.1f}°C, range {min_temp:.1f}-{max_temp:.1f}°C")
        
        # Humidity trends
        if len(humidities) >= 2:
            humidity_trend = "increasing" if humidities[-1] > humidities[0] else "decreasing" if humidities[-1] < humidities[0] else "stable"
            avg_humidity = statistics.mean(humidities)
            trends.append(f"Humidity: {humidity_trend} trend, avg {avg_humidity:.1f}%")
        
        # Soil moisture trends
        if len(soil_moistures) >= 2:
            soil_trend = "increasing" if soil_moistures[-1] > soil_moistures[0] else "decreasing" if soil_moistures[-1] < soil_moistures[0] else "stable"
            avg_soil = statistics.mean(soil_moistures)
            trends.append(f"Soil moisture: {soil_trend} trend, avg {avg_soil:.1f}%")
        
        # Wind patterns
        if len(wind_speeds) >= 2:
            avg_wind = statistics.mean(wind_speeds)
            max_wind = max(wind_speeds)
            trends.append(f"Wind: avg {avg_wind:.1f} m/s, max {max_wind:.1f} m/s")
        
        # Rain patterns
        rain_events = [get_rain_status(r) for r in rain_intensities if r is not None]
        if rain_events:
            heavy_rain_hours = rain_events.count('Heavy Rain')
            light_rain_hours = rain_events.count('Light Rain')
            no_rain_hours = rain_events.count('No Rain')
            trends.append(f"Rainfall: {heavy_rain_hours}h heavy, {light_rain_hours}h light, {no_rain_hours}h dry")
        
        return " | ".join(trends) if trends else "Limited historical data available for analysis."
        
    except Exception as e:
        logger.error(f"Error analyzing historical trends: {str(e)}")
        return "Error analyzing historical trends."

def get_grok_recommendations(data):
    if not data:
        logger.warning("No sensor data provided for recommendations")
        return {"pomegranate": "No data available to provide recommendations.", "guava": "No data available to provide recommendations.", "historical_summary": "No historical data available."}

    # Check cache
    cache_key = json.dumps(data, sort_keys=True)
    if cache_key in recommendation_cache:
        logger.info(f"Cache hit for sensor data: {json.dumps(data, indent=2)}")
        return recommendation_cache[cache_key]

    logger.info(f"Cache miss, fetching new recommendations for sensor data: {json.dumps(data, indent=2)}")

    # Fetch 24-hour historical data
    historical_data = fetch_historical_24h_data()
    historical_summary = analyze_historical_trends(historical_data)

    # Prepare current sensor data for the prompt
    temp = data.get('temperature', 'unknown')
    humidity = data.get('humidity', 'unknown')
    soil_moisture = data.get('soil_moisture', 'unknown')
    rain_intensity = get_rain_status(data.get('rain_intensity', 'unknown'))
    wind_speed = data.get('wind_speed', 'unknown')
    motion_detected = 'detected' if data.get('motion_detected') == '1' else 'not detected'

    prompt = f"""
    You are an agricultural expert providing recommendations for Pomegranate in Flowering stage and Guava plants in Fruiting stage.

    CURRENT CONDITIONS:
    - Temperature: {temp} °C
    - Humidity: {humidity} %
    - Soil Moisture: {soil_moisture} %
    - Rainfall: {rain_intensity}
    - Wind Speed: {wind_speed} m/s
    - Motion Detected: {motion_detected}

    24-HOUR HISTORICAL TRENDS:
    {historical_summary}

    Based on both current conditions AND the 24-hour historical patterns, provide specific, actionable recommendations for each crop to optimize growth and health. Consider how the recent weather patterns and trends should influence immediate care decisions.

    Return the response in JSON format with keys 'pomegranate', 'guava', and 'historical_summary', each containing a string with recommendations.
    """
    logger.debug(f"Groq API prompt: {prompt}")

    try:
        logger.info("Sending request to Groq API with historical context")
        response = requests.post(GROQ_API_URL, headers={
            'Authorization': f'Bearer {GROQ_API_KEY}',
            'Content-Type': 'application/json'
        }, json={
            'model': 'llama-3.3-70b-versatile',
            'messages': [{'role': 'user', 'content': prompt}],
            'max_tokens': 800
        })

        if not response.ok:
            error_data = response.json().get('error', {})
            error_message = error_data.get('message', 'Unknown error')
            error_code = error_data.get('code', 'unknown')
            logger.error(f"Groq API request failed: Status {response.status_code} - {error_message} (code: {error_code})")
            return {
                "pomegranate": f"Failed to fetch recommendations: {error_message}", 
                "guava": f"Failed to fetch recommendations: {error_message}",
                "historical_summary": historical_summary
            }

        result = response.json()
        recommendations = result['choices'][0]['message']['content']
        logger.info(f"Raw Groq API response: ```json\n{recommendations}\n```")
        
        # Strip Markdown code block markers if present
        cleaned_response = re.sub(r'^```json\n|\n```$', '', recommendations.strip())
        
        try:
            parsed_recommendations = json.loads(cleaned_response)
            if not isinstance(parsed_recommendations, dict) or 'pomegranate' not in parsed_recommendations or 'guava' not in parsed_recommendations:
                raise ValueError("Invalid JSON structure: missing 'pomegranate' or 'guava' keys")
            
            # Ensure historical_summary is included
            if 'historical_summary' not in parsed_recommendations:
                parsed_recommendations['historical_summary'] = historical_summary
            
            recommendation_cache[cache_key] = parsed_recommendations
            logger.info("Successfully parsed and cached recommendations with historical context")
            return parsed_recommendations
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Groq API response as JSON: {str(e)}")
            return {
                "pomegranate": f"Error parsing recommendations: {str(e)}", 
                "guava": f"Error parsing recommendations: {str(e)}",
                "historical_summary": historical_summary
            }
        except ValueError as e:
            logger.error(f"Invalid response structure: {str(e)}")
            return {
                "pomegranate": f"Invalid response: {str(e)}", 
                "guava": f"Invalid response: {str(e)}",
                "historical_summary": historical_summary
            }
    except Exception as e:
        logger.error(f"Error fetching recommendations from Groq API: {str(e)}")
        return {
            "pomegranate": f"Failed to fetch recommendations: {str(e)}", 
            "guava": f"Failed to fetch recommendations: {str(e)}",
            "historical_summary": historical_summary
        }

@app.route('/')
def serve_index():
    global request_counter
    request_counter += 1
    logger.info(f"Serving index.html - Request #{request_counter} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/api/recommendations')
def get_recommendations():
    global request_counter
    request_counter += 1
    logger.info(f"Handling /api/recommendations request #{request_counter} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Try fetching data from progressively larger time ranges
    data = fetch_sensor_data('-10m')
    if not data:
        logger.warning("No data in last 10 minutes, trying last 1 day")
        data = fetch_sensor_data('-1d')
    if not data:
        logger.warning("No data in last 1 day, trying last 7 days")
        data = fetch_sensor_data('-7d')

    recommendations = get_grok_recommendations(data)
    return jsonify({
        'recommendations': recommendations,
        'data': data or {}
    })

@app.route('/api/ask', methods=['POST'])
def ask_crop_question():
    global request_counter
    request_counter += 1
    logger.info(f"Handling /api/ask request #{request_counter} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    try:
        request_data = request.get_json()
        if not request_data or 'prompt' not in request_data:
            logger.error("Invalid request: Missing 'prompt' in request body")
            return jsonify({'error': 'Missing prompt in request body'}), 400

        prompt = request_data['prompt']
        logger.debug(f"Received prompt: {prompt}")

        # Get current data
        data = fetch_sensor_data('-10m')
        if not data:
            data = fetch_sensor_data('-1d')
        if not data:
            data = fetch_sensor_data('-7d')

        # Get historical data
        historical_data = fetch_historical_24h_data()
        historical_summary = analyze_historical_trends(historical_data)

        # Prepare current sensor data
        temp = data.get('temperature', 'unknown') if data else 'unknown'
        humidity = data.get('humidity', 'unknown') if data else 'unknown'
        soil_moisture = data.get('soil_moisture', 'unknown') if data else 'unknown'
        rain_intensity = get_rain_status(data.get('rain_intensity', 'unknown')) if data else 'unknown'
        wind_speed = data.get('wind_speed', 'unknown') if data else 'unknown'
        motion_detected = 'detected' if data and data.get('motion_detected') == '1' else 'not detected'

        enhanced_prompt = f"""
        You are an agricultural expert. Here's the context:

        CURRENT CONDITIONS:
        - Temperature: {temp} °C
        - Humidity: {humidity} %
        - Soil Moisture: {soil_moisture} %
        - Rainfall: {rain_intensity}
        - Wind Speed: {wind_speed} m/s
        - Motion Detected: {motion_detected}

        24-HOUR HISTORICAL TRENDS:
        {historical_summary}

        USER QUESTION: {prompt}

        Provide a comprehensive response considering both current conditions and recent historical patterns.
        """

        response = requests.post(GROQ_API_URL, headers={
            'Authorization': f'Bearer {GROQ_API_KEY}',
            'Content-Type': 'application/json'
        }, json={
            'model': 'llama-3.3-70b-versatile',
            'messages': [{'role': 'user', 'content': enhanced_prompt}],
            'max_tokens': 600
        })

        if not response.ok:
            error_data = response.json().get('error', {})
            error_message = error_data.get('message', 'Unknown error')
            error_code = error_data.get('code', 'unknown')
            logger.error(f"Groq API request failed: Status {response.status_code} - {error_message} (code: {error_code})")
            return jsonify({'error': f"Groq API error: {error_message}"}), response.status_code

        result = response.json()
        response_text = result['choices'][0]['message']['content']
        logger.info(f"Groq API response: {response_text}")
        return jsonify({'response': response_text})

    except Exception as e:
        logger.error(f"Error processing /api/ask request: {str(e)}")
        return jsonify({'error': f"Failed to process request: {str(e)}"}), 500

if __name__ == '__main__':
    logger.info("Starting Flask application")
    app.run(debug=True, host='0.0.0.0', port=5000)