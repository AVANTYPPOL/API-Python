"""
Simple Calibrated Model for GUI
===============================

A lightweight model that provides calibrated predictions.
Falls back to simple calculations if ML models aren't available.
"""

import numpy as np

class SimpleCalibratedModel:
    """Simple model with calibration factor"""
    
    def __init__(self):
        self.calibration_factor = 0.779
        self.base_rate = 2.50
        self.per_km_rate = 1.35
        self.per_minute_rate = 0.25
        
    def predict(self, distance_km, surge_multiplier=1.0, traffic_level='moderate'):
        """Make a simple calibrated prediction"""
        # Base calculation
        base_price = self.base_rate + (distance_km * self.per_km_rate)
        
        # Traffic adjustment
        traffic_multipliers = {
            'light': 1.0,
            'moderate': 1.2,
            'heavy': 1.5
        }
        traffic_mult = traffic_multipliers.get(traffic_level, 1.2)
        
        # Apply calibration and surge
        price = base_price * traffic_mult * surge_multiplier * self.calibration_factor
        
        return max(price, 2.50)  # Minimum fare
    
    def predict_price(self, distance_km, pickup_lat, pickup_lng, dropoff_lat, dropoff_lng,
                     hour_of_day, day_of_week, surge_multiplier=1.0, 
                     traffic_level='moderate', weather_condition='clear'):
        """Predict price using simple formula - matches GUI interface"""
        # Just use distance-based prediction
        return self.predict(distance_km, surge_multiplier, traffic_level)
    
    def predict_all_services(self, distance_km, surge_multiplier=1.0, traffic_level='moderate'):
        """Predict prices for all service types"""
        base_price = self.predict(distance_km, surge_multiplier, traffic_level)
        
        return {
            'uberx': base_price,
            'uber_xl': base_price * 1.4,
            'uber_premier': base_price * 1.8,
            'premier_suv': base_price * 2.2
        } 