from flask import Flask, request, jsonify
import logging
import os
import traceback
from datetime import datetime
import math

# Import XGBoost API at top level to catch import errors early
try:
    from xgboost_pricing_api import XGBoostPricingAPI
    XGBOOST_AVAILABLE = True
    print("✅ XGBoost API imported successfully")
except ImportError as e:
    print(f"❌ XGBoost import failed: {e}")
    XGBOOST_AVAILABLE = False

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

def load_xgboost_model():
    """Load the XGBoost ML model"""
    global pricing_model, model_info
    
    try:
        if not XGBOOST_AVAILABLE:
            logger.error("❌ XGBoost not available - cannot load model")
            return False
            
        logger.info("🚀 Loading XGBoost Miami Uber Price Prediction Model...")
        logger.info("=" * 70)
        logger.info("🔄 XGBOOST MIAMI UBER PRICE PREDICTION MODEL")
        logger.info("=" * 70)
        logger.info("🏖️  Miami-Specific Features (Airport, Beach, Downtown)")
        logger.info("📊 High Accuracy: R² = 0.8822 (88.22%)")
        logger.info("🚗 Multi-Service Built-in (All 4 Types)")
        logger.info("💰 RMSE: $13.31")
        logger.info("=" * 70)
        
        # Initialize the XGBoost model
        logger.info("✅ XGBoost API available, initializing model...")
        pricing_model = XGBoostPricingAPI('xgboost_miami_model.pkl')
        
        if pricing_model is not None and pricing_model.is_loaded:
            logger.info("✅ XGBoost model loaded successfully")
            
            # Check API status
            gmaps_status = "✅ Configured" if pricing_model.gmaps else "❌ Not configured (using Haversine distance)"
            weather_status = "✅ Configured" if pricing_model.weather_api_key else "❌ Not configured (using clear weather default)"
            
            logger.info(f"🗺️  Google Maps API: {gmaps_status}")
            logger.info(f"🌤️  Weather API: {weather_status}")
            logger.info("🤖 ML Model: Trained XGBoost model loaded")
            logger.info("📊 Training Data: 28,531 Miami rides")
            
            model_info = {
                'model_type': 'xgboost_miami_model',
                'accuracy': '88.22%',
                'r2_score': 0.8822,
                'rmse': '$13.31',
                'description': 'XGBoost model with Miami-specific feature engineering',
                'features': ['distance', 'location', 'time_patterns', 'traffic_level', 'weather', 
                           'miami_airports', 'miami_beaches', 'miami_downtown'],
                'services': ['UberX', 'UberXL', 'Uber Premier', 'Premier SUV'],
                'training_samples': 28531,
                'model_location': 'xgboost_miami_model.pkl'
            }
            
            logger.info("=" * 70)
            logger.info("✅ XGBOOST MODEL READY FOR PRODUCTION")
            logger.info("=" * 70)
        else:
            logger.warning("⚠️  XGBoost model not loaded properly, using fallback pricing")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error loading XGBoost model: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

# Load model on startup
logger.info("🚀 Initializing Rideshare Pricing API...")
if not load_xgboost_model():
    logger.warning("⚠️  XGBoost model failed to load, using fallback algorithm")
    logger.info("✅ Fallback pricing algorithm is production-ready")

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'version': '1.0.0',
        'model_loaded': pricing_model is not None,
        'model_info': model_info,
        'api_name': 'XGBoost Miami Pricing API'
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
        
        if pricing_model:
            # Get prediction from XGBoost model - only coordinates needed!
            try:
                predictions = pricing_model.predict_all_services(
                    pickup_lat=pickup_lat,
                    pickup_lng=pickup_lng,
                    dropoff_lat=dropoff_lat,
                    dropoff_lng=dropoff_lng
                )
                
                # Always return all service predictions
                return jsonify({
                    'success': True,
                    'predictions': predictions,  # All 4 services
                    'request_details': {
                        'distance_miles': round(distance_km * 0.621371, 1),
                        'distance_km': round(distance_km, 1)
                    },
                    'model_info': {
                        'model_type': model_info.get('model_type', 'ultimate_miami_model'),
                        'accuracy': model_info.get('accuracy', '72.86%')
                    }
                })
                
            except Exception as e:
                logger.error(f"Model prediction error: {e}")
                # Fallback to simple pricing
                base_price = 2.50 + (distance_km * 1.65)
                
                # Always return all services with fallback pricing
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
                        'distance_km': round(distance_km, 1)
                    },
                    'note': 'Using fallback pricing due to model error'
                })
        else:
            # Fallback pricing when model not loaded
            base_price = 2.50 + (distance_km * 1.65)
            
            # Always return all services
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
                    'distance_km': round(distance_km, 1)
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
                
                if pricing_model:
                    # Get predictions for all services - only coordinates needed!
                    try:
                        predictions = pricing_model.predict_all_services(
                            pickup_lat=pickup_lat,
                            pickup_lng=pickup_lng,
                            dropoff_lat=dropoff_lat,
                            dropoff_lng=dropoff_lng
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
                        base_price = 2.50 + (distance_km * 1.65)
                        results.append({
                            'ride_index': i + 1,
                            'predictions': {
                                'UberX': round(base_price, 2),
                                'UberXL': round(base_price * 1.55, 2),
                                'Uber Premier': round(base_price * 2.0, 2),
                                'Premier SUV': round(base_price * 2.64, 2)
                            },
                            'distance_km': round(distance_km, 1),
                            'note': 'Fallback pricing used'
                        })
                else:
                    # Fallback pricing
                    base_price = 2.50 + (distance_km * 1.65)
                    results.append({
                        'ride_index': i + 1,
                        'predictions': {
                            'UberX': round(base_price, 2),
                            'UberXL': round(base_price * 1.55, 2),
                            'Uber Premier': round(base_price * 2.0, 2),
                            'Premier SUV': round(base_price * 2.64, 2)
                        },
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