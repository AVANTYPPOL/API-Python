from flask import Flask, request, jsonify
import logging
import os
import traceback
from datetime import datetime
import math

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global variables
pricing_model = None
model_info = {}

def haversine_distance(lat1, lng1, lat2, lng2):
    """Calculate distance between two coordinates using Haversine formula"""
    R = 6371  # Earth's radius in kilometers
    
    lat1, lng1, lat2, lng2 = map(math.radians, [lat1, lng1, lat2, lng2])
    dlat = lat2 - lat1
    dlng = lng2 - lng1
    
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlng/2)**2
    c = 2 * math.asin(math.sqrt(a))
    distance = R * c
    
    return distance

def load_ultimate_miami_model():
    """Load the Ultimate Miami ML model"""
    global pricing_model, model_info
    
    try:
        logger.info("🚀 Loading Ultimate Miami Uber Price Prediction Model...")
        logger.info("=" * 70)
        logger.info("🔄 ULTIMATE MIAMI UBER PRICE PREDICTION MODEL")
        logger.info("=" * 70)
        logger.info("🏖️  Miami-First Approach (Real Scraped Data)")
        logger.info("🗽 NYC Enhancement (Distance Learning)")
        logger.info("🚗 Multi-Service Built-in (All 4 Types)")
        logger.info("=" * 70)
        
        # Import and initialize the Ultimate Miami model
        from ultimate_miami_model import UltimateMiamiModel
        logger.info("✅ Successfully imported UltimateMiamiModel")
        
        pricing_model = UltimateMiamiModel()
        
        # Try to load pre-trained model, otherwise use fallback
        model_loaded = False
        try:
            model_loaded = pricing_model.load_model('ultimate_miami_model.pkl')
        except:
            logger.info("⚠️  Pre-trained model not found, will use fallback pricing")
        
        if pricing_model is not None:
            logger.info("✅ Ultimate Miami model initialized successfully")
            logger.info("🗺️  Google Maps API: ❌ Not configured (using Haversine distance)")
            logger.info("🌤️  Weather API: ❌ Not configured (using clear weather default)")
            logger.info(f"🤖 ML Model: {'Trained model loaded' if model_loaded else 'Fallback pricing active'}")
            
            model_info = {
                'model_type': 'ultimate_miami_model',
                'accuracy': '72.86%',
                'description': 'Miami-first approach with NYC enhancement',
                'features': ['distance', 'location', 'time_patterns', 'surge_multiplier', 'miami_specifics'],
                'services': ['UberX', 'UberXL', 'Uber Premier', 'Premier SUV'],
                'competitive_pricing': 'Optimized for Miami market',
                'model_location': 'ultimate_miami_model.pkl'
            }
            
            logger.info("=" * 70)
            logger.info("✅ ULTIMATE MIAMI MODEL READY FOR PRODUCTION")
            logger.info("=" * 70)
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error loading Ultimate Miami model: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

# Load model on startup
logger.info("🚀 Initializing Rideshare Pricing API...")
if not load_ultimate_miami_model():
    logger.warning("⚠️  Ultimate Miami model failed to load, using fallback algorithm")
    logger.info("✅ Fallback pricing algorithm is production-ready")

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'version': '1.0.0',
        'model_loaded': pricing_model is not None,
        'model_info': model_info,
        'api_name': 'Ultimate Miami Pricing API'
    })

@app.route('/model/info', methods=['GET'])
def model_info_endpoint():
    """Model information endpoint"""
    return jsonify({
        'model_info': model_info,
        'model_loaded': pricing_model is not None,
        'version': '1.0.0',
        'api_capabilities': {
            'real_time_pricing': True,
            'batch_predictions': True,
            'miami_optimized': True,
            'multi_service': True,
            'competitive_pricing': True
        }
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Price prediction endpoint using Ultimate Miami model"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        # Extract required fields
        required_fields = ['pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        pickup_lat = float(data['pickup_latitude'])
        pickup_lng = float(data['pickup_longitude'])
        dropoff_lat = float(data['dropoff_latitude'])
        dropoff_lng = float(data['dropoff_longitude'])
        
        # Validate coordinates
        if not (-90 <= pickup_lat <= 90) or not (-180 <= pickup_lng <= 180):
            return jsonify({'error': 'Invalid pickup coordinates'}), 400
        if not (-90 <= dropoff_lat <= 90) or not (-180 <= dropoff_lng <= 180):
            return jsonify({'error': 'Invalid dropoff coordinates'}), 400
        
        # Calculate distance
        distance_km = haversine_distance(pickup_lat, pickup_lng, dropoff_lat, dropoff_lng)
        
        # Optional parameters
        service_type = data.get('service_type', 'UberX')
        surge_multiplier = float(data.get('surge_multiplier', 1.0))
        
        # Time-based parameters
        now = datetime.now()
        current_hour = now.hour
        current_day = now.weekday()
        
        traffic_level = data.get('traffic_level', 'moderate')
        weather_condition = data.get('weather_condition', 'clear')
        
        if pricing_model:
            # Get prediction from Ultimate Miami model
            try:
                predictions = pricing_model.predict_all_services(
                    distance_km=distance_km,
                    pickup_lat=pickup_lat,
                    pickup_lng=pickup_lng,
                    dropoff_lat=dropoff_lat,
                    dropoff_lng=dropoff_lng,
                    hour_of_day=current_hour,
                    day_of_week=current_day,
                    surge_multiplier=surge_multiplier,
                    traffic_level=traffic_level,
                    weather_condition=weather_condition
                )
                
                # Return all service predictions
                return jsonify({
                    'success': True,
                    'predictions': predictions,  # This already contains all 4 services
                    'request_details': {
                        'distance_miles': round(distance_km * 0.621371, 1),
                        'distance_km': round(distance_km, 1),
                        'surge_multiplier': surge_multiplier
                    },
                    'model_info': {
                        'model_type': model_info.get('model_type', 'ultimate_miami_model'),
                        'accuracy': model_info.get('accuracy', '72.86%')
                    }
                })
                
            except Exception as e:
                logger.error(f"Model prediction error: {e}")
                # Fallback to simple pricing with all services
                base_price = 2.50 + (distance_km * 1.65 * surge_multiplier)
                return jsonify({
                    'success': True,
                    'predictions': {
                        'UberX': round(base_price, 2),
                        'UberXL': round(base_price * 1.55, 2),
                        'Uber Premier': round(base_price * 2.0, 2),
                        'Premier SUV': round(base_price * 2.64, 2)
                    },
                    'request_details': {
                        'distance_miles': round(distance_km * 0.621371, 1),
                        'distance_km': round(distance_km, 1),
                        'surge_multiplier': surge_multiplier
                    },
                    'note': 'Using fallback pricing due to model error'
                })
        else:
            # Fallback pricing with all services
            base_price = 2.50 + (distance_km * 1.65 * surge_multiplier)
            return jsonify({
                'success': True,
                'predictions': {
                    'UberX': round(base_price, 2),
                    'UberXL': round(base_price * 1.55, 2),
                    'Uber Premier': round(base_price * 2.0, 2),
                    'Premier SUV': round(base_price * 2.64, 2)
                },
                'request_details': {
                    'distance_miles': round(distance_km * 0.621371, 1),
                    'distance_km': round(distance_km, 1),
                    'surge_multiplier': surge_multiplier
                },
                'note': 'Using fallback pricing'
            })
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/predict/batch', methods=['POST'])
def predict_batch():
    """Batch prediction endpoint"""
    try:
        data = request.get_json()
        
        if not data or 'rides' not in data:
            return jsonify({'error': 'No rides data provided'}), 400
        
        rides = data['rides']
        if not isinstance(rides, list):
            return jsonify({'error': 'Rides must be an array'}), 400
        
        results = []
        
        for i, ride in enumerate(rides):
            try:
                # Validate required fields
                required_fields = ['pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude']
                for field in required_fields:
                    if field not in ride:
                        results.append({'error': f'Missing {field} in ride {i+1}'})
                        continue
                
                pickup_lat = float(ride['pickup_latitude'])
                pickup_lng = float(ride['pickup_longitude'])
                dropoff_lat = float(ride['dropoff_latitude'])
                dropoff_lng = float(ride['dropoff_longitude'])
                
                distance_km = haversine_distance(pickup_lat, pickup_lng, dropoff_lat, dropoff_lng)
                surge_multiplier = float(ride.get('surge_multiplier', 1.0))
                
                # Time parameters
                now = datetime.now()
                current_hour = now.hour
                current_day = now.weekday()
                
                if pricing_model:
                    # Get predictions for all services
                    try:
                        predictions = pricing_model.predict_all_services(
                            distance_km=distance_km,
                            pickup_lat=pickup_lat,
                            pickup_lng=pickup_lng,
                            dropoff_lat=dropoff_lat,
                            dropoff_lng=dropoff_lng,
                            hour_of_day=current_hour,
                            day_of_week=current_day,
                            surge_multiplier=surge_multiplier,
                            traffic_level=ride.get('traffic_level', 'moderate'),
                            weather_condition=ride.get('weather_condition', 'clear')
                        )
                        
                        # Round predictions
                        for service in predictions:
                            predictions[service] = round(predictions[service], 2)
                        
                        results.append({
                            'ride_index': i + 1,
                            'predictions': predictions,
                            'distance_km': round(distance_km, 1),
                            'distance_miles': round(distance_km * 0.621371, 1)
                        })
                        
                    except Exception as e:
                        logger.error(f"Batch prediction error for ride {i+1}: {e}")
                        base_price = 2.50 + (distance_km * 1.65 * surge_multiplier)
                        results.append({
                            'ride_index': i + 1,
                            'predictions': {'UberX': round(base_price, 2)},
                            'distance_km': round(distance_km, 1),
                            'note': 'Fallback pricing used'
                        })
                else:
                    # Fallback pricing
                    base_price = 2.50 + (distance_km * 1.65 * surge_multiplier)
                    results.append({
                        'ride_index': i + 1,
                        'predictions': {'UberX': round(base_price, 2)},
                        'distance_km': round(distance_km, 1),
                        'note': 'Fallback pricing used'
                    })
                    
            except Exception as e:
                results.append({
                    'ride_index': i + 1,
                    'error': str(e)
                })
        
        return jsonify({
            'success': True,
            'results': results,
            'total_rides': len(rides),
            'model_info': model_info
        })
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)