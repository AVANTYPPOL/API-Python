from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
import os
import traceback
from datetime import datetime
import math
import psutil  # For memory monitoring
import time

# Import Hybrid Pricing API (new) and XGBoost API (backup) at top level
try:
    from hybrid_pricing_api import HybridPricingAPI
    HYBRID_AVAILABLE = True
    print("[OK] Hybrid Pricing API imported successfully - v3.0")
except ImportError as e:
    print(f"[WARNING] Hybrid API import failed: {e}")
    HYBRID_AVAILABLE = False

try:
    from xgboost_pricing_api import XGBoostPricingAPI
    from model_manager import model_manager
    XGBOOST_AVAILABLE = True
    print("[OK] XGBoost API and ModelManager imported successfully (backup)")
except ImportError as e:
    print(f"[WARNING] XGBoost or ModelManager import failed: {e}")
    XGBOOST_AVAILABLE = False
    model_manager = None

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Global variables (legacy compatibility)
pricing_model = None
model_info = {}
model_loaded = False
model_loading = False

# Enhanced model management
USE_MODEL_MANAGER = model_manager is not None

# Performance optimization: Cache for distance calculations
distance_cache = {}

# Price adjustment: Base 15% discount + service-specific additional discounts
BASE_DISCOUNT = 0.15  # 15% base discount for all services

# Service-specific additional discounts (on top of base discount)
SERVICE_DISCOUNTS = {
    'UBERX': 0.0625,       # +6.25% additional = 21.25% total
    'PREMIER': 0.0625,     # +6.25% additional = 21.25% total
    'UBERXL': 0.10,        # +10% additional = 25% total
    'SUV_PREMIER': 0.1262  # +12.62% additional = 27.62% total
}

def get_total_discount(service):
    """Get total discount for a service (base + service-specific)"""
    return BASE_DISCOUNT + SERVICE_DISCOUNTS.get(service, 0)

def apply_price_discount(predictions):
    """Apply service-specific discounts to all predictions"""
    discounted = {}
    for service, price in predictions.items():
        total_discount = get_total_discount(service)
        discounted[service] = round(float(price) * (1 - total_discount), 2)
    return discounted

def haversine_distance(lat1, lng1, lat2, lng2):
    """Calculate distance between two coordinates with caching for performance"""
    # Create cache key with rounded coordinates for caching
    cache_key = (round(lat1, 4), round(lng1, 4), round(lat2, 4), round(lng2, 4))
    
    if cache_key in distance_cache:
        return distance_cache[cache_key]
    
    R = 6371  # Earth's radius in kilometers
    
    lat1, lng1, lat2, lng2 = map(math.radians, [lat1, lng1, lat2, lng2])
    dlat = lat2 - lat1
    dlng = lng2 - lng1
    
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlng/2)**2
    c = 2 * math.asin(math.sqrt(a))
    distance = R * c
    
    # Cache the result (limit cache size to prevent memory issues)
    if len(distance_cache) < 1000:
        distance_cache[cache_key] = distance
    
    return distance

def load_pricing_model():
    """Load pricing model - tries Hybrid first, falls back to XGBoost"""
    global pricing_model, model_info, model_loaded, model_loading

    # Prevent multiple simultaneous loading attempts
    if model_loading:
        logger.info("[INFO] Model is already being loaded by another process...")
        return model_loaded

    if model_loaded:
        logger.info("[OK] Model already loaded, skipping...")
        return True

    model_loading = True

    try:
        # Try loading Hybrid model first (new v3.0)
        if HYBRID_AVAILABLE:
            logger.info("=" * 70)
            logger.info("LOADING HYBRID PRICING MODEL V3.0")
            logger.info("=" * 70)
            logger.info("Architecture: Static Rules + ML Booking Fee Prediction")
            logger.info("Booking Fee Model: XGBoost (R² = 0.97, MAE = $0.34)")
            logger.info("=" * 70)

            start_time = time.time()
            pricing_model = HybridPricingAPI('booking_fee_model.pkl')
            load_time = time.time() - start_time
            logger.info(f"[TIME] Model initialization took {load_time:.2f} seconds")

            if pricing_model is not None and pricing_model.is_loaded:
                logger.info("[OK] Hybrid pricing model loaded successfully")

                model_info = {
                    'model_type': 'hybrid_pricing_v3',
                    'accuracy': '97.11%',
                    'description': 'Static Uber rules + ML-predicted booking fees',
                    'formula': 'base_fare + (miles × per_mile_rate) + (minutes × rate_per_minute) + ML_booking_fee',
                    'services': ['PREMIER', 'SUV_PREMIER', 'UBERX', 'UBERXL'],
                    'booking_fee_model': {
                        'type': 'XGBoost',
                        'r2_score': 0.9711,
                        'mae': '$0.34',
                        'rmse': '$0.49'
                    }
                }

                logger.info("=" * 70)
                logger.info("[OK] HYBRID MODEL READY FOR PRODUCTION")
                logger.info("=" * 70)
                model_loaded = True
                model_loading = False
                return True

        # Fallback to old XGBoost model
        logger.info("[INFO] Falling back to XGBoost model...")

        # Use ModelManager if available
        if USE_MODEL_MANAGER:
            success = model_manager.load_model()
            if success:
                pricing_model = model_manager.model
                model_info = model_manager.get_api_model_info()
                model_loaded = True
                model_loading = False
            return success

        if not XGBOOST_AVAILABLE:
            logger.error("❌ XGBoost not available - cannot load model")
            return False
            
        logger.info("🚀 Loading XGBoost Miami Uber Price Prediction Model...")
        logger.info("=" * 70)
        logger.info("🔄 XGBOOST MIAMI UBER PRICE PREDICTION MODEL")
        logger.info("=" * 70)
        logger.info("🏖️  Miami-Specific Features (Airport, Beach, Downtown)")
        logger.info("📊 High Accuracy: R² = 0.9093 (90.93%)")
        logger.info("🚗 Multi-Service Built-in (All 4 Types)")
        logger.info("💰 RMSE: $13.31")
        logger.info("=" * 70)
        
        # Monitor memory before loading
        try:
            process = psutil.Process()
            mem_before = process.memory_info().rss / 1024 / 1024  # MB
            logger.info(f"💾 Memory before model load: {mem_before:.2f} MB")
        except:
            mem_before = 0
        
        # Initialize the XGBoost model
        logger.info("✅ XGBoost API available, initializing model...")
        start_time = time.time()
        pricing_model = XGBoostPricingAPI('xgboost_miami_model.pkl')
        load_time = time.time() - start_time
        logger.info(f"⏱️ Model initialization took {load_time:.2f} seconds")
        
        # Monitor memory after loading
        try:
            mem_after = process.memory_info().rss / 1024 / 1024  # MB
            mem_used = mem_after - mem_before
            logger.info(f"💾 Memory after model load: {mem_after:.2f} MB (used: {mem_used:.2f} MB)")
        except:
            pass
        
        if pricing_model is not None and pricing_model.is_loaded:
            logger.info("✅ XGBoost model loaded successfully")
            
            # Check API status
            gmaps_status = "✅ Configured" if pricing_model.gmaps else "❌ Not configured (using Haversine distance)"
            weather_status = "✅ Configured" if pricing_model.weather_api_key else "❌ Not configured (using clear weather default)"
            
            logger.info(f"🗺️  Google Maps API: {gmaps_status}")
            logger.info(f"🌤️  Weather API: {weather_status}")
            logger.info("🤖 ML Model: Trained XGBoost model loaded")
            logger.info("📊 Training Data: 157,446 Miami rides")
            
            model_info = {
                'model_type': 'xgboost_miami_model',
                'accuracy': '90.93%',
                'r2_score': 0.9093,
                'rmse': '$11.38',
                'description': 'XGBoost model with Miami-specific feature engineering',
                'features': ['distance', 'location', 'time_patterns', 'traffic_level', 'weather', 
                           'miami_airports', 'miami_beaches', 'miami_downtown'],
                'services': ['PREMIER', 'SUV_PREMIER', 'UBERX', 'UBERXL'],
                'training_samples': 157446,
                'model_location': 'xgboost_miami_model.pkl'
            }
            
            logger.info("=" * 70)
            logger.info("✅ XGBOOST MODEL READY FOR PRODUCTION")
            logger.info("=" * 70)
            model_loaded = True
            model_loading = False
            return True
        else:
            logger.warning("⚠️  XGBoost model not loaded properly, using fallback pricing")
            model_loading = False
            return False
        
    except Exception as e:
        logger.error(f"❌ Error loading XGBoost model: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        model_loading = False
        return False
    finally:
        model_loading = False

# Lazy loading - model will be loaded on first request
logger.info("🚀 Initializing Rideshare Pricing API...")
logger.info(f"💰 Base discount configured: {BASE_DISCOUNT * 100:.0f}%")
logger.info("💰 Service-specific total discounts:")
for service, additional in SERVICE_DISCOUNTS.items():
    total = (BASE_DISCOUNT + additional) * 100
    logger.info(f"   - {service}: {total:.2f}% ({BASE_DISCOUNT*100:.0f}% base + {additional*100:.2f}% additional)")
logger.info("📦 Model will be loaded on first request (lazy loading enabled)")

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint - triggers lazy model loading"""
    global model_loaded

    # Try to load model if not already loaded (lazy loading)
    if not model_loaded:
        logger.info("[INFO] Health check triggered - attempting to load model...")
        load_pricing_model()

    # Enhanced health info with version tracking
    health_response = {
        'status': 'healthy',
        'version': '3.0.0',
        'model_loaded': model_loaded,
        'model_info': model_info,
        'api_name': 'Hybrid Pricing API (Static Rules + ML Booking Fee)'
    }
    
    # Add version tracking info if available (for monitoring)
    if USE_MODEL_MANAGER and model_manager.is_loaded:
        deployment_info = model_manager.get_deployment_summary()
        health_response['deployment_info'] = deployment_info
    
    return jsonify(health_response)

@app.route('/model/info', methods=['GET'])
def model_info_endpoint():
    """Model information endpoint with enhanced version tracking"""
    base_response = {
        'model_info': model_info,
        'model_loaded': pricing_model is not None,
        'version': '3.0.0',
        'architecture': 'Hybrid: Static Pricing Rules + ML Booking Fee Prediction',
        'api_capabilities': {
            'real_time_pricing': True,
            'batch_predictions': True,
            'miami_optimized': True,
            'multi_service': True,
            'uses_google_maps': True,
            'ml_booking_fee': True
        }
    }
    
    # Add enhanced tracking info if available (for monitoring)
    if USE_MODEL_MANAGER:
        enhanced_info = model_manager.get_model_info()
        base_response['version_tracking'] = enhanced_info['internal_info']
        base_response['deployment_summary'] = model_manager.get_deployment_summary()
    
    return jsonify(base_response)

@app.route('/predict', methods=['POST'])
def predict():
    """Price prediction endpoint using Hybrid Pricing Model v3.0

    Optional query parameters:
    ?debug=true - Includes detailed price breakdown for testing (does NOT break API contract)
    ?details=true - Includes service details with pricing metrics (production-ready, backward compatible)
    """
    global model_loaded, pricing_model

    # Ensure model is loaded (lazy loading)
    if not model_loaded:
        logger.info("[INFO] Prediction request triggered - attempting to load model...")
        load_pricing_model()

    # Check for optional query parameters
    debug_mode = request.args.get('debug', 'false').lower() == 'true'
    details_mode = request.args.get('details', 'false').lower() == 'true'

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

        # Try using pricing_model first (Hybrid or XGBoost)
        if pricing_model and hasattr(pricing_model, 'predict_all_services'):
            # Legacy model prediction path
            try:
                predictions = pricing_model.predict_all_services(
                    pickup_lat=pickup_lat,
                    pickup_lng=pickup_lng,
                    dropoff_lat=dropoff_lat,
                    dropoff_lng=dropoff_lng
                )

                # Apply 20% discount and return all service predictions
                discounted_predictions = apply_price_discount(predictions)

                # Build base response
                response = {
                    'success': True,
                    'predictions': discounted_predictions,  # All 4 services with 20% discount
                    'request_details': {
                        'distance_miles': round(distance_km * 0.621371, 1),
                        'distance_km': round(distance_km, 1)
                    },
                    'model_info': {
                        'model_type': model_info.get('model_type', 'xgboost_miami_model'),
                        'accuracy': model_info.get('accuracy', '88.22%')
                    }
                }

                # Add debug breakdown if requested (OPTIONAL - doesn't break API contract)
                if debug_mode and HYBRID_AVAILABLE and hasattr(pricing_model, 'get_distance_and_duration'):
                    try:
                        # Get Google Maps distance and duration
                        distance_km_gmaps, distance_miles_gmaps, duration_minutes = pricing_model.get_distance_and_duration(
                            pickup_lat, pickup_lng, dropoff_lat, dropoff_lng
                        )

                        # Calculate breakdown for each service
                        breakdown = {}
                        for service_type in ['UBERX', 'UBERXL', 'PREMIER', 'SUV_PREMIER']:
                            rules = pricing_model.PRICING_RULES[service_type]

                            base_fare = rules['base_fare']
                            distance_cost = distance_miles_gmaps * rules['per_mile_rate']
                            time_cost = duration_minutes * rules['per_minute_rate']

                            # Get booking fee prediction
                            booking_fee = 0
                            if hasattr(pricing_model, 'booking_fee_model') and pricing_model.booking_fee_model:
                                # Get current Miami time for prediction
                                from hybrid_pricing_api import get_miami_time
                                miami_time = get_miami_time()

                                booking_fee = pricing_model.booking_fee_model.predict_booking_fee(
                                    pickup_lat=pickup_lat,
                                    pickup_lng=pickup_lng,
                                    dropoff_lat=dropoff_lat,
                                    dropoff_lng=dropoff_lng,
                                    service_type=service_type,
                                    hour_of_day=miami_time.hour,
                                    day_of_week=miami_time.weekday(),
                                    traffic_level='moderate',
                                    weather_condition='clear'
                                )

                            subtotal = base_fare + distance_cost + time_cost + booking_fee
                            final_price = max(subtotal, rules['minimum_fare'])

                            # Calculate discounted price with service-specific discount
                            total_discount = get_total_discount(service_type)
                            discounted_price = final_price * (1 - total_discount)
                            logger.info(f"[DEBUG] {service_type}: final=${final_price:.2f}, discount={total_discount*100:.2f}%, discounted=${discounted_price:.2f}")

                            breakdown[service_type] = {
                                'base_fare': round(base_fare, 2),
                                'distance_cost': round(distance_cost, 2),
                                'time_cost': round(time_cost, 2),
                                'booking_fee': round(booking_fee, 2),
                                'subtotal': round(subtotal, 2),
                                'minimum_fare': round(rules['minimum_fare'], 2),
                                'final_price': round(final_price, 2),
                                'discounted_price': round(discounted_price, 2)
                            }

                        response['debug_breakdown'] = {
                            'google_maps': {
                                'distance_miles': round(distance_miles_gmaps, 2),
                                'distance_km': round(distance_km_gmaps, 2),
                                'duration_minutes': round(duration_minutes, 1)
                            },
                            'price_components': breakdown,
                            'note': 'Debug mode - this field only appears when ?debug=true is used'
                        }
                    except Exception as e:
                        logger.warning(f"Debug breakdown failed: {e}")

                # Add service details if requested (OPTIONAL - doesn't break API contract)
                logger.info(f"[DEBUG] details_mode={details_mode}, HYBRID_AVAILABLE={HYBRID_AVAILABLE}, has_method={hasattr(pricing_model, 'get_distance_and_duration') if pricing_model else False}")
                if details_mode and HYBRID_AVAILABLE and hasattr(pricing_model, 'get_distance_and_duration'):
                    try:
                        # Get Google Maps distance and duration
                        distance_km_gmaps, distance_miles_gmaps, duration_minutes = pricing_model.get_distance_and_duration(
                            pickup_lat, pickup_lng, dropoff_lat, dropoff_lng
                        )

                        # Add Google Maps data to request_details for transparency
                        response['request_details']['google_maps'] = {
                            'distance_miles': round(distance_miles_gmaps, 2),
                            'distance_km': round(distance_km_gmaps, 2),
                            'duration_minutes': round(duration_minutes, 1),
                            'source': 'Google Maps Distance Matrix API with real-time traffic'
                        }

                        # Also show straight-line distance for comparison
                        response['request_details']['straight_line'] = {
                            'distance_miles': round(distance_km * 0.621371, 2),
                            'distance_km': round(distance_km, 2),
                            'source': 'Haversine formula (direct line)'
                        }

                        # Show the difference
                        distance_difference_pct = ((distance_miles_gmaps - (distance_km * 0.621371)) / (distance_km * 0.621371)) * 100
                        response['request_details']['road_vs_straight_line'] = {
                            'difference_miles': round(distance_miles_gmaps - (distance_km * 0.621371), 2),
                            'difference_percent': round(distance_difference_pct, 1),
                            'note': 'Road distance is typically 20-60% longer than straight-line due to actual street routing'
                        }

                        # Clarify which data is used for pricing
                        response['request_details']['pricing_calculation_source'] = {
                            'distance_used': f"{round(distance_miles_gmaps, 2)} miles (Google Maps road distance)",
                            'duration_used': f"{round(duration_minutes, 1)} minutes (Google Maps with real-time traffic)",
                            'note': 'All service prices are calculated using Google Maps road routing, NOT straight-line distance'
                        }

                        # Keep duration for backward compatibility
                        response['request_details']['duration_minutes'] = round(duration_minutes, 1)

                        # Build service details for each service type
                        service_details = {}
                        for service_type in ['UBERX', 'UBERXL', 'PREMIER', 'SUV_PREMIER']:
                            rules = pricing_model.PRICING_RULES[service_type]

                            # Calculate components
                            base_fare = rules['base_fare']
                            distance_cost = distance_miles_gmaps * rules['per_mile_rate']
                            time_cost = duration_minutes * rules['per_minute_rate']

                            # Get booking fee prediction
                            booking_fee = 0
                            if hasattr(pricing_model, 'booking_fee_model') and pricing_model.booking_fee_model:
                                from hybrid_pricing_api import get_miami_time
                                miami_time = get_miami_time()

                                booking_fee = pricing_model.booking_fee_model.predict_booking_fee(
                                    pickup_lat=pickup_lat,
                                    pickup_lng=pickup_lng,
                                    dropoff_lat=dropoff_lat,
                                    dropoff_lng=dropoff_lng,
                                    service_type=service_type,
                                    hour_of_day=miami_time.hour,
                                    day_of_week=miami_time.weekday(),
                                    traffic_level='moderate',
                                    weather_condition='clear'
                                )

                            subtotal = base_fare + distance_cost + time_cost + booking_fee
                            final_price = max(subtotal, rules['minimum_fare'])

                            # Get total discount for this service type
                            total_discount = get_total_discount(service_type)

                            service_details[service_type] = {
                                'base_fare': round(base_fare, 2),
                                'per_mile_rate': round(rules['per_mile_rate'], 2),
                                'per_minute_rate': round(rules['per_minute_rate'], 2),
                                'minimum_fare': round(rules['minimum_fare'], 2),
                                'distance_cost': round(distance_cost, 2),
                                'time_cost': round(time_cost, 2),
                                'booking_fee': round(booking_fee, 2),
                                'subtotal': round(subtotal, 2),
                                'final_price_before_discount': round(final_price, 2),
                                'discount_percentage': round(total_discount * 100, 2),
                                'final_price': round(final_price * (1 - total_discount), 2)
                            }

                        response['service_details'] = service_details

                    except Exception as e:
                        logger.warning(f"Service details generation failed: {e}")

                return jsonify(response)
                
            except Exception as e:
                logger.error(f"Model prediction error: {e}")
                # Fallback to simple pricing
                base_price = 2.50 + (distance_km * 1.65)
                fallback_predictions = {
                    'PREMIER': round(base_price * 2.0, 2),
                    'SUV_PREMIER': round(base_price * 2.64, 2),
                    'UBERX': round(base_price, 2),
                    'UBERXL': round(base_price * 1.55, 2)
                }
                
                # Apply 20% discount to fallback pricing
                discounted_fallback = apply_price_discount(fallback_predictions)
                return jsonify({
                    'success': True,
                    'predictions': discounted_fallback,
                    'request_details': {
                        'distance_miles': round(distance_km * 0.621371, 1),
                        'distance_km': round(distance_km, 1)
                    },
                    'model_info': {
                        'model_type': model_info.get('model_type', 'hybrid_pricing_v3'),
                        'accuracy': model_info.get('accuracy', '97.11%')
                    }
                })
        else:
            # Fallback pricing when model not loaded
            base_price = 2.50 + (distance_km * 1.65)
            fallback_predictions = {
                'PREMIER': round(base_price * 2.0, 2),
                'SUV_PREMIER': round(base_price * 2.64, 2),
                'UBERX': round(base_price, 2),
                'UBERXL': round(base_price * 1.55, 2)
            }
            
            # Apply 20% discount to fallback pricing
            discounted_fallback = apply_price_discount(fallback_predictions)
            return jsonify({
                'success': True,
                'predictions': discounted_fallback,
                'request_details': {
                    'distance_miles': round(distance_km * 0.621371, 1),
                    'distance_km': round(distance_km, 1)
                },
                'model_info': {
                    'model_type': 'xgboost_miami_model',
                    'accuracy': '88.22%'
                }
            })
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/predict/batch', methods=['POST'])
def predict_batch():
    """Batch prediction endpoint"""
    global pricing_model, model_loaded

    # Ensure model is loaded (lazy loading)
    if not model_loaded:
        logger.info("[INFO] Batch prediction request triggered - attempting to load model...")
        load_pricing_model()

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
                
                # Use ModelManager for batch predictions if available
                if USE_MODEL_MANAGER and model_manager.is_model_loaded():
                    try:
                        predictions = model_manager.predict_all_services(
                            pickup_lat=pickup_lat,
                            pickup_lng=pickup_lng,
                            dropoff_lat=dropoff_lat,
                            dropoff_lng=dropoff_lng
                        )
                    except Exception as e:
                        logger.error(f"ModelManager batch prediction error for ride {i+1}: {e}")
                        # Fallback to simple pricing
                        predictions = None
                elif pricing_model:
                    # Legacy model prediction path
                    try:
                        predictions = pricing_model.predict_all_services(
                            pickup_lat=pickup_lat,
                            pickup_lng=pickup_lng,
                            dropoff_lat=dropoff_lat,
                            dropoff_lng=dropoff_lng
                        )
                        
                        # Round predictions and apply 20% discount
                        for service in predictions:
                            predictions[service] = round(predictions[service], 2)
                        discounted_predictions = apply_price_discount(predictions)
                        
                        results.append({
                            'ride_index': i + 1,
                            'predictions': discounted_predictions,
                            'distance_km': round(distance_km, 1),
                            'distance_miles': round(distance_km * 0.621371, 1)
                        })
                        
                    except Exception as e:
                        logger.error(f"Batch prediction error for ride {i+1}: {e}")
                        base_price = 2.50 + (distance_km * 1.65)
                        fallback_predictions = {
                            'PREMIER': round(base_price * 2.0, 2),
                            'SUV_PREMIER': round(base_price * 2.64, 2),
                            'UBERX': round(base_price, 2),
                            'UBERXL': round(base_price * 1.55, 2)
                        }
                        discounted_fallback = apply_price_discount(fallback_predictions)
                        results.append({
                            'ride_index': i + 1,
                            'predictions': discounted_fallback,
                            'distance_km': round(distance_km, 1)
                        })
                else:
                    # Fallback pricing when no model available
                    base_price = 2.50 + (distance_km * 1.65)
                    fallback_predictions = {
                        'PREMIER': round(base_price * 2.0, 2),
                        'SUV_PREMIER': round(base_price * 2.64, 2),
                        'UBERX': round(base_price, 2),
                        'UBERXL': round(base_price * 1.55, 2)
                    }
                    discounted_fallback = apply_price_discount(fallback_predictions)
                    results.append({
                        'ride_index': i + 1,
                        'predictions': discounted_fallback,
                        'distance_km': round(distance_km, 1)
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