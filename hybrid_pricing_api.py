"""
Hybrid Pricing API
==================

Combines static Uber pricing rules with ML-predicted booking fees.

Pricing Formula:
    price = base_fare + (miles * per_mile_rate) + (minutes * rate_per_minute) + predicted_booking_fee
    price = max(price, minimum_fare)
"""

import os
from dotenv import load_dotenv
from datetime import datetime
from zoneinfo import ZoneInfo
import math

# Try to import googlemaps, but make it optional
try:
    import googlemaps
    GOOGLEMAPS_AVAILABLE = True
except ImportError:
    GOOGLEMAPS_AVAILABLE = False
    print("[WARNING] googlemaps module not available - will use Haversine distance/ETA estimates")

# Load environment variables
load_dotenv()

# Import the booking fee model
from booking_fee_model import BookingFeeModel

# Miami timezone (Eastern Time - handles EDT/EST automatically)
MIAMI_TZ = ZoneInfo("America/New_York")


def get_miami_time():
    """Get current time in Miami timezone (Eastern Time)"""
    return datetime.now(MIAMI_TZ)


class HybridPricingAPI:
    """
    Hybrid pricing API that uses:
    1. Google Maps API for distance and duration
    2. Static pricing rules (base fare, per-mile, per-minute rates)
    3. ML model for booking fee prediction
    """

    # Static pricing rules (these don't change)
    PRICING_RULES = {
        'UBERX': {
            'base_fare': 2.25,
            'per_mile_rate': 0.79,
            'per_minute_rate': 0.20,
            'minimum_fare': 5.70
        },
        'UBERXL': {
            'base_fare': 4.22,
            'per_mile_rate': 1.70,
            'per_minute_rate': 0.30,
            'minimum_fare': 8.58
        },
        'PREMIER': {
            'base_fare': 2.98,
            'per_mile_rate': 1.99,
            'per_minute_rate': 0.57,
            'minimum_fare': 15.97
        },
        'SUV_PREMIER': {
            'base_fare': 3.76,
            'per_mile_rate': 2.46,
            'per_minute_rate': 0.71,
            'minimum_fare': 21.95
        }
    }

    def __init__(self, model_path='booking_fee_model.pkl'):
        self.model_path = model_path
        self.booking_fee_model = None
        self.is_loaded = False

        # Initialize API clients
        self.gmaps = None

        # Initialize Google Maps if API key is available
        google_api_key = os.getenv('GOOGLE_MAPS_API_KEY')
        if google_api_key and GOOGLEMAPS_AVAILABLE:
            try:
                self.gmaps = googlemaps.Client(key=google_api_key)
                print("[OK] Google Maps API initialized")
            except Exception as e:
                print(f"[WARNING] Google Maps API initialization failed: {e}")

        # Load booking fee model
        self.load_model()

    def load_model(self):
        """Load the booking fee prediction model"""
        try:
            print(f"[SEARCH] Looking for model file: {self.model_path}")

            if os.path.exists(self.model_path):
                print(f"[OK] Model file found: {self.model_path}")

                self.booking_fee_model = BookingFeeModel()
                self.booking_fee_model.load_model(self.model_path)
                self.is_loaded = True
                print("[OK] Booking fee model loaded successfully")
            else:
                print(f"[ERROR] Model file not found: {self.model_path}")
                self.is_loaded = False
        except Exception as e:
            print(f"[ERROR] Failed to load booking fee model: {e}")
            import traceback
            traceback.print_exc()
            self.is_loaded = False

    def haversine_distance(self, lat1, lon1, lat2, lon2):
        """Calculate Haversine distance between two points (in km and miles)"""
        R = 6371  # Earth's radius in kilometers
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        distance_km = 2 * R * math.asin(math.sqrt(a))
        distance_miles = distance_km * 0.621371
        return distance_km, distance_miles

    def estimate_duration(self, distance_km):
        """
        Estimate duration based on distance
        Uses average Miami speeds: 25 mph = 40 km/h
        """
        avg_speed_kmh = 40
        duration_hours = distance_km / avg_speed_kmh
        duration_minutes = duration_hours * 60
        return duration_minutes

    def get_distance_and_duration(self, pickup_lat, pickup_lng, dropoff_lat, dropoff_lng):
        """
        Get real distance and duration from Google Maps API
        Falls back to Haversine distance + estimated duration if API unavailable
        """
        if self.gmaps:
            try:
                # Request with traffic model for better accuracy
                result = self.gmaps.distance_matrix(
                    origins=[(pickup_lat, pickup_lng)],
                    destinations=[(dropoff_lat, dropoff_lng)],
                    mode="driving",
                    units="metric",
                    departure_time="now",
                    traffic_model="best_guess"
                )

                if result['status'] == 'OK' and result['rows'][0]['elements'][0]['status'] == 'OK':
                    element = result['rows'][0]['elements'][0]

                    # Get distance
                    distance_m = element['distance']['value']
                    distance_km = distance_m / 1000
                    distance_miles = distance_km * 0.621371

                    # Get duration (use duration_in_traffic if available)
                    if 'duration_in_traffic' in element:
                        duration_s = element['duration_in_traffic']['value']
                    else:
                        duration_s = element['duration']['value']

                    duration_minutes = duration_s / 60

                    print(f"[GOOGLE] Distance: {distance_miles:.2f} mi, Duration: {duration_minutes:.1f} min")
                    return distance_km, distance_miles, duration_minutes

            except Exception as e:
                print(f"[WARNING] Google Maps API error: {e}")

        # Fallback to Haversine + estimate
        distance_km, distance_miles = self.haversine_distance(
            pickup_lat, pickup_lng, dropoff_lat, dropoff_lng
        )
        duration_minutes = self.estimate_duration(distance_km)

        print(f"[FALLBACK] Distance: {distance_miles:.2f} mi (Haversine), Duration: {duration_minutes:.1f} min (estimated)")
        return distance_km, distance_miles, duration_minutes

    def calculate_price(self, service_type, distance_miles, duration_minutes, booking_fee):
        """
        Calculate price using static pricing rules + predicted booking fee

        Formula: price = base_fare + (miles * per_mile_rate) + (minutes * rate_per_minute) + booking_fee
        """
        rules = self.PRICING_RULES[service_type]

        base_fare = rules['base_fare']
        distance_cost = distance_miles * rules['per_mile_rate']
        time_cost = duration_minutes * rules['per_minute_rate']
        minimum_fare = rules['minimum_fare']

        # Calculate total price
        price = base_fare + distance_cost + time_cost + booking_fee

        # Apply minimum fare
        price = max(price, minimum_fare)

        return price

    def predict_all_services(self, pickup_lat, pickup_lng, dropoff_lat, dropoff_lng,
                            hour_of_day=None, day_of_week=None,
                            traffic_level=None, weather_condition=None):
        """
        Predict prices for all service types using hybrid model

        Takes: Just coordinates (everything else auto-filled)
        Returns: Same JSON structure as old API
        """
        if not self.is_loaded:
            # Return fallback prices if model not loaded
            return {
                'PREMIER': 45.20,
                'SUV_PREMIER': 55.90,
                'UBERX': 25.50,
                'UBERXL': 35.75
            }

        # STEP 1: Get current time if not provided (MIAMI LOCAL TIME)
        miami_time = get_miami_time()

        if hour_of_day is None:
            hour_of_day = miami_time.hour
            print(f"[TIME] Using current Miami time: {miami_time.strftime('%I:%M %p')} (hour={hour_of_day})")

        if day_of_week is None:
            day_of_week = miami_time.weekday()
            days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            print(f"[DATE] Using current day: {days[day_of_week]}")

        # STEP 2: Get distance and duration from Google Maps
        distance_km, distance_miles, duration_minutes = self.get_distance_and_duration(
            pickup_lat, pickup_lng, dropoff_lat, dropoff_lng
        )

        # STEP 3: Use default traffic/weather if not provided
        if traffic_level is None:
            traffic_level = 'moderate'
            print(f"[TRAFFIC] Using default: {traffic_level}")

        if weather_condition is None:
            weather_condition = 'clear'
            print(f"[WEATHER] Using default: {weather_condition}")

        # STEP 4: Predict booking fees and calculate prices for all services
        results = {}

        for service_type in ['UBERX', 'UBERXL', 'PREMIER', 'SUV_PREMIER']:
            try:
                # Predict booking fee using ML model
                booking_fee = self.booking_fee_model.predict_booking_fee(
                    pickup_lat=pickup_lat,
                    pickup_lng=pickup_lng,
                    dropoff_lat=dropoff_lat,
                    dropoff_lng=dropoff_lng,
                    service_type=service_type,
                    hour_of_day=hour_of_day,
                    day_of_week=day_of_week,
                    traffic_level=traffic_level,
                    weather_condition=weather_condition
                )

                # Calculate total price using formula
                price = self.calculate_price(
                    service_type=service_type,
                    distance_miles=distance_miles,
                    duration_minutes=duration_minutes,
                    booking_fee=booking_fee
                )

                results[service_type] = float(price)

                print(f"[{service_type}] Booking fee: ${booking_fee:.2f}, Total: ${price:.2f}")

            except Exception as e:
                print(f"[ERROR] Failed to predict {service_type}: {e}")
                # Fallback to minimum fare
                results[service_type] = float(self.PRICING_RULES[service_type]['minimum_fare'])

        # Return in exact same format as old API
        return results

    def get_model_info(self):
        """Get information about the loaded model"""
        if not self.is_loaded:
            return {
                'status': 'Not loaded',
                'model_type': 'Hybrid (Static Rules + ML Booking Fee)',
                'features': 'Unknown'
            }

        return {
            'status': 'Loaded',
            'model_type': 'Hybrid Pricing: Static Rules + ML Booking Fee',
            'booking_fee_model': {
                'type': 'XGBoost',
                'r2_score': self.booking_fee_model.model_metadata.get('test_r2', 0),
                'mae': self.booking_fee_model.model_metadata.get('test_mae', 0)
            },
            'services': ['UBERX', 'UBERXL', 'PREMIER', 'SUV_PREMIER'],
            'pricing_formula': 'base_fare + (miles * per_mile_rate) + (minutes * rate_per_minute) + ML_predicted_booking_fee'
        }


def test_api():
    """Test the hybrid pricing API"""
    print("="*70)
    print("TESTING HYBRID PRICING API")
    print("="*70)

    api = HybridPricingAPI()

    # Test model info
    info = api.get_model_info()
    print(f"\nModel Status: {info['status']}")
    print(f"Model Type: {info['model_type']}")

    if api.is_loaded:
        # Test predictions
        print("\n" + "-"*70)
        print("Test Route: Miami Airport -> South Beach")
        print("-"*70)

        prices = api.predict_all_services(
            pickup_lat=25.7959, pickup_lng=-80.2870,  # Airport
            dropoff_lat=25.7907, dropoff_lng=-80.1300,  # South Beach
            hour_of_day=14,
            day_of_week=2,
            traffic_level='moderate',
            weather_condition='clear'
        )

        print("\n" + "-"*70)
        print("PREDICTED PRICES:")
        print("-"*70)
        for service, price in prices.items():
            print(f"   {service:15} ${price:.2f}")

    print("\n" + "="*70)
    print("TEST COMPLETE")
    print("="*70)


if __name__ == "__main__":
    test_api()
