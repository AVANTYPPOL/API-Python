"""
Multi-Service Pricing API
========================

Simple API interface for predicting prices across all 4 Uber service types.
Ready for integration into your ride-sharing app.

Author: AI Assistant  
Date: 2024
"""

import os
from models.multi_service_uber_model import MultiServiceUberModel
from api.hybrid_address_pricing_interface import HybridAddressPricingInterface

# Load API keys from environment variables
if 'GOOGLE_MAPS_API_KEY' not in os.environ:
    raise ValueError("GOOGLE_MAPS_API_KEY environment variable not set")
if 'WEATHER_API_KEY' not in os.environ:
    raise ValueError("WEATHER_API_KEY environment variable not set")

class MultiServicePricingAPI:
    """
    Simple API for multi-service price predictions
    """
    
    def __init__(self):
        # Load the trained model
        self.model = MultiServiceUberModel()
        self.model.load_model('multi_service_uber_model.pkl')
        
        # Address converter for real-world usage
        self.address_interface = HybridAddressPricingInterface()
        
        # Service display names
        self.service_display_names = {
            'uberx': 'UberX',
            'uber_xl': 'UberXL', 
            'uber_premier': 'Uber Premier',
            'premier_suv': 'Premier SUV'
        }
        
        # Service descriptions
        self.service_descriptions = {
            'uberx': 'Affordable rides for up to 4 riders',
            'uber_xl': 'Affordable rides for groups up to 6',
            'uber_premier': 'Premium rides with professional drivers', 
            'premier_suv': 'Premium SUVs for up to 6 riders'
        }
        
        print("✅ Multi-Service Pricing API initialized")
    
    def get_prices_by_coordinates(self, pickup_lat, pickup_lng, dropoff_lat, dropoff_lng,
                                  surge_multiplier=1.0, include_details=True):
        """
        Get prices for all services using coordinates
        
        Args:
            pickup_lat: Pickup latitude
            pickup_lng: Pickup longitude
            dropoff_lat: Dropoff latitude
            dropoff_lng: Dropoff longitude
            surge_multiplier: Current surge multiplier (default 1.0)
            include_details: Include service descriptions and metadata
            
        Returns:
            dict: Service prices and details
        """
        try:
            # Get real-time data
            realtime_data = self.address_interface._get_realtime_data(
                pickup_lat, pickup_lng, dropoff_lat, dropoff_lng
            )
            
            # Calculate distance
            distance_km = self.address_interface._calculate_distance(
                pickup_lat, pickup_lng, dropoff_lat, dropoff_lng
            )
            
            # Get current time info
            from datetime import datetime
            now = datetime.now()
            hour = now.hour
            day_of_week = now.weekday()
            
            # Get price predictions
            prices = self.model.predict_all_services(
                distance_km=distance_km,
                pickup_lat=pickup_lat,
                pickup_lng=pickup_lng,
                dropoff_lat=dropoff_lat,
                dropoff_lng=dropoff_lng,
                hour_of_day=hour,
                day_of_week=day_of_week,
                surge_multiplier=surge_multiplier,
                traffic_level=realtime_data.get('traffic_level', 'light'),
                weather_condition=realtime_data.get('weather_condition', 'clear')
            )
            
            # Build response
            response = {
                'success': True,
                'distance_km': round(distance_km, 2),
                'distance_miles': round(distance_km * 0.621371, 2),
                'estimated_duration_min': realtime_data.get('duration_minutes', distance_km * 3),
                'surge_multiplier': surge_multiplier,
                'services': []
            }
            
            # Add service details
            for service_key, price in prices.items():
                service_info = {
                    'service_id': service_key,
                    'display_name': self.service_display_names.get(service_key, service_key),
                    'price_usd': price,
                    'price_per_km': round(price / distance_km, 2) if distance_km > 0 else 0,
                    'price_per_mile': round(price / (distance_km * 0.621371), 2) if distance_km > 0 else 0
                }
                
                if include_details:
                    service_info['description'] = self.service_descriptions.get(service_key, '')
                    service_info['capacity'] = 4 if 'xl' not in service_key and 'suv' not in service_key else 6
                    service_info['is_premium'] = 'premier' in service_key
                
                response['services'].append(service_info)
            
            # Add metadata
            if include_details:
                response['metadata'] = {
                    'traffic_level': realtime_data.get('traffic_level', 'unknown'),
                    'weather_condition': realtime_data.get('weather_condition', 'unknown'),
                    'is_airport_trip': self._is_airport_trip(pickup_lat, pickup_lng, dropoff_lat, dropoff_lng),
                    'is_peak_hour': hour in [7, 8, 17, 18, 19],
                    'is_late_night': hour >= 22 or hour <= 5,
                    'timestamp': datetime.now().isoformat()
                }
            
            return response
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'services': []
            }
    
    def get_prices_by_address(self, pickup_address, dropoff_address, 
                             surge_multiplier=1.0, include_details=True):
        """
        Get prices for all services using addresses
        
        Args:
            pickup_address: Pickup address string
            dropoff_address: Dropoff address string
            surge_multiplier: Current surge multiplier
            include_details: Include service descriptions
            
        Returns:
            dict: Service prices and details
        """
        try:
            # Convert addresses to coordinates
            pickup_coords = self.address_interface._geocode_address(pickup_address)
            if not pickup_coords:
                return {'success': False, 'error': 'Could not geocode pickup address'}
            
            dropoff_coords = self.address_interface._geocode_address(dropoff_address)
            if not dropoff_coords:
                return {'success': False, 'error': 'Could not geocode dropoff address'}
            
            # Get prices using coordinates
            result = self.get_prices_by_coordinates(
                pickup_lat=pickup_coords['lat'],
                pickup_lng=pickup_coords['lng'],
                dropoff_lat=dropoff_coords['lat'],
                dropoff_lng=dropoff_coords['lng'],
                surge_multiplier=surge_multiplier,
                include_details=include_details
            )
            
            # Add address info
            if result['success']:
                result['addresses'] = {
                    'pickup': pickup_address,
                    'dropoff': dropoff_address
                }
            
            return result
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'services': []
            }
    
    def get_competitive_prices(self, pickup_lat, pickup_lng, dropoff_lat, dropoff_lng,
                              surge_multiplier=1.0):
        """
        Get competitive pricing recommendations for all services
        
        Returns prices with competitive adjustments based on market analysis
        """
        # Get base predictions
        base_result = self.get_prices_by_coordinates(
            pickup_lat, pickup_lng,
            dropoff_lat, dropoff_lng,
            surge_multiplier, include_details=False
        )
        
        if not base_result['success']:
            return base_result
        
        # Apply competitive adjustments
        competitive_result = base_result.copy()
        
        for service in competitive_result['services']:
            base_price = service['price_usd']
            
            # Competitive pricing strategy
            if service['service_id'] == 'uberx':
                # UberX: Slightly undercut for market share
                competitive_price = base_price * 0.95
            elif service['service_id'] == 'uber_xl':
                # XL: Match market for group rides
                competitive_price = base_price * 0.98
            elif service['service_id'] == 'uber_premier':
                # Premier: Premium but competitive
                competitive_price = base_price * 1.02
            else:  # premier_suv
                # Premier SUV: Maintain premium positioning
                competitive_price = base_price * 1.05
            
            service['ml_price'] = base_price
            service['competitive_price'] = round(competitive_price, 2)
            service['price_usd'] = round(competitive_price, 2)
            service['adjustment_factor'] = round(competitive_price / base_price, 3)
        
        competitive_result['pricing_strategy'] = 'competitive'
        
        return competitive_result
    
    def _is_airport_trip(self, pickup_lat, pickup_lng, dropoff_lat, dropoff_lng):
        """Check if trip involves airport"""
        airport_lat, airport_lng = 25.7959, -80.2870
        threshold = 0.02
        
        pickup_dist = ((pickup_lat - airport_lat) ** 2 + (pickup_lng - airport_lng) ** 2) ** 0.5
        dropoff_dist = ((dropoff_lat - airport_lat) ** 2 + (dropoff_lng - airport_lng) ** 2) ** 0.5
        
        return pickup_dist < threshold or dropoff_dist < threshold
    
    def format_price_display(self, result):
        """
        Format prices for UI display
        
        Returns formatted string for each service
        """
        if not result.get('success'):
            return "Price unavailable"
        
        formatted = []
        for service in result['services']:
            name = service['display_name']
            price = service['price_usd']
            per_mile = service.get('price_per_mile', 0)
            
            # Format like Uber app
            formatted.append({
                'service': name,
                'price_range': f"${price:.2f}",
                'price_detail': f"${per_mile:.2f}/mile",
                'eta': f"{result.get('estimated_duration_min', 'N/A')} min"
            })
        
        return formatted

def main():
    """Demo the API"""
    print("\n🚀 Multi-Service Pricing API Demo")
    print("=" * 60)
    
    # Initialize API
    api = MultiServicePricingAPI()
    
    # Test cases
    test_trips = [
        {
            'pickup': "Miami International Airport",
            'dropoff': "South Beach, Miami",
            'surge': 1.0
        },
        {
            'pickup': "Brickell City Centre, Miami",
            'dropoff': "Wynwood Walls, Miami", 
            'surge': 1.5
        }
    ]
    
    for trip in test_trips:
        print(f"\n📍 {trip['pickup']} → {trip['dropoff']}")
        print(f"⚡ Surge: {trip['surge']}x")
        print("-" * 40)
        
        # Get prices
        result = api.get_prices_by_address(
            trip['pickup'], 
            trip['dropoff'],
            trip['surge']
        )
        
        if result['success']:
            # Display formatted prices
            formatted = api.format_price_display(result)
            for service in formatted:
                print(f"{service['service']:15} {service['price_range']:>10} ({service['price_detail']})")
        else:
            print(f"Error: {result['error']}")
    
    # Show JSON response example
    print("\n📋 Example API Response:")
    print("-" * 40)
    
    example = api.get_prices_by_coordinates(
        25.7617, -80.1918,  # Downtown Miami
        25.7907, -80.1300,  # South Beach
        surge_multiplier=1.2
    )
    
    if example['success']:
        import json
        print(json.dumps(example, indent=2))

if __name__ == "__main__":
    main() 