"""
Calibrated Miami Uber Model
===========================

Miami-specific model calibrated to real Uber prices.
Uses a price adjustment factor to match actual market rates.
"""

import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
import joblib

class CalibratedMiamiModel:
    """Miami-specific model with price calibration"""
    
    def __init__(self, db_path='uber_ml_data.db'):
        self.db_path = db_path
        self.model = None
        self.label_encoders = {}
        self.is_trained = False
        
        # Miami-specific pricing calibration
        # Based on your observation: Model=$14.10, Real Uber=$10.98
        self.calibration_factor = 10.98 / 14.10  # ≈ 0.78
        
        # Service multipliers (from your cleaned data)
        self.service_multipliers = {
            'uberx': 1.00,
            'uber_xl': 1.55,
            'uber_premier': 2.00,
            'premier_suv': 2.64
        }
        
        print(f"🎯 Miami Model initialized with calibration factor: {self.calibration_factor:.3f}")
    
    def load_data(self):
        """Load Miami data"""
        conn = sqlite3.connect(self.db_path)
        
        # Load only complete records
        query = """
        SELECT distance_km, pickup_lat, pickup_lng, dropoff_lat, dropoff_lng,
               hour_of_day, day_of_week, surge_multiplier, traffic_level, weather_condition,
               uberx_price
        FROM rides 
        WHERE uberx_price > 0 AND distance_km > 0
        """
        
        df = pd.read_sql(query, conn)
        conn.close()
        
        print(f"✅ Loaded {len(df)} Miami records")
        print(f"   Avg UberX price: ${df['uberx_price'].mean():.2f}")
        print(f"   Price per km: ${(df['uberx_price'] / df['distance_km']).mean():.2f}")
        
        return df
    
    def engineer_features(self, df):
        """Create Miami-specific features"""
        df_eng = df.copy()
        
        # Distance features
        df_eng['distance_km_squared'] = df_eng['distance_km'] ** 2
        df_eng['distance_km_log'] = np.log1p(df_eng['distance_km'])
        
        # Miami locations
        airport_lat, airport_lng = 25.7959, -80.2870
        downtown_lat, downtown_lng = 25.7617, -80.1918
        beach_lat, beach_lng = 25.7907, -80.1300
        
        # Location features
        df_eng['is_airport_trip'] = (
            (np.abs(df_eng['pickup_lat'] - airport_lat) < 0.02) |
            (np.abs(df_eng['dropoff_lat'] - airport_lat) < 0.02)
        ).astype(int)
        
        df_eng['is_beach_trip'] = (
            (np.abs(df_eng['pickup_lat'] - beach_lat) < 0.02) |
            (np.abs(df_eng['dropoff_lat'] - beach_lat) < 0.02)
        ).astype(int)
        
        # Time features
        df_eng['is_rush_hour'] = df_eng['hour_of_day'].apply(
            lambda x: 1 if x in [7, 8, 17, 18, 19] else 0
        )
        
        df_eng['is_weekend'] = df_eng['day_of_week'].apply(
            lambda x: 1 if x in [5, 6] else 0
        )
        
        return df_eng
    
    def train_model(self):
        """Train calibrated Miami model"""
        print("\n🚀 Training Calibrated Miami Model...")
        
        # Load data
        df = self.load_data()
        if df is None or len(df) < 50:
            print("❌ Insufficient data for training")
            return False
        
        # Apply calibration to target prices
        df['calibrated_price'] = df['uberx_price'] * self.calibration_factor
        
        print(f"📊 Price Calibration Applied:")
        print(f"   Original avg: ${df['uberx_price'].mean():.2f}")
        print(f"   Calibrated avg: ${df['calibrated_price'].mean():.2f}")
        
        # Engineer features
        df_processed = self.engineer_features(df)
        
        # Encode categorical features
        for feature in ['traffic_level', 'weather_condition']:
            if feature in df_processed.columns:
                self.label_encoders[feature] = LabelEncoder()
                df_processed[feature] = self.label_encoders[feature].fit_transform(
                    df_processed[feature].fillna('unknown').astype(str)
                )
        
        # Prepare features and target
        feature_cols = ['distance_km', 'distance_km_squared', 'distance_km_log',
                       'pickup_lat', 'pickup_lng', 'dropoff_lat', 'dropoff_lng',
                       'hour_of_day', 'day_of_week', 'surge_multiplier',
                       'is_airport_trip', 'is_beach_trip', 'is_rush_hour', 'is_weekend']
        
        # Add encoded categorical features
        for feature in ['traffic_level', 'weather_condition']:
            if feature in df_processed.columns:
                feature_cols.append(feature)
        
        X = df_processed[feature_cols].fillna(0)
        y = df_processed['calibrated_price']
        
        # Train model
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=8,
            min_samples_split=3,
            min_samples_leaf=2,
            random_state=42
        )
        
        self.model.fit(X, y)
        
        # Evaluate
        y_pred = self.model.predict(X)
        r2 = r2_score(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        
        print(f"\n📈 Model Performance:")
        print(f"   R²: {r2:.4f}")
        print(f"   RMSE: ${rmse:.2f}")
        
        # Feature importance
        importance = pd.DataFrame({
            'feature': X.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\n🏆 Top Features:")
        for _, row in importance.head(5).iterrows():
            print(f"   {row['feature']:20s}: {row['importance']*100:.1f}%")
        
        self.feature_names = X.columns.tolist()
        self.is_trained = True
        
        return True
    
    def predict_all_services(self, distance_km, pickup_lat, pickup_lng, 
                           dropoff_lat, dropoff_lng, hour_of_day=12, 
                           day_of_week=0, surge_multiplier=1.0,
                           traffic_level='light', weather_condition='clear'):
        """Predict prices for all services with calibration"""
        
        if not self.is_trained:
            # Simple fallback
            base_price = (2.50 + distance_km * 1.50) * self.calibration_factor
            return {
                'uberx': base_price * surge_multiplier,
                'uber_xl': base_price * self.service_multipliers['uber_xl'] * surge_multiplier,
                'uber_premier': base_price * self.service_multipliers['uber_premier'] * surge_multiplier,
                'premier_suv': base_price * self.service_multipliers['premier_suv'] * surge_multiplier
            }
        
        try:
            # Create input data
            input_data = pd.DataFrame({
                'distance_km': [distance_km],
                'pickup_lat': [pickup_lat],
                'pickup_lng': [pickup_lng],
                'dropoff_lat': [dropoff_lat],
                'dropoff_lng': [dropoff_lng],
                'hour_of_day': [hour_of_day],
                'day_of_week': [day_of_week],
                'surge_multiplier': [surge_multiplier]
            })
            
            # Engineer features
            input_processed = self.engineer_features(input_data)
            
            # Encode categorical features
            for feature in ['traffic_level', 'weather_condition']:
                if feature in self.label_encoders:
                    if feature == 'traffic_level':
                        input_processed[feature] = traffic_level
                    else:
                        input_processed[feature] = weather_condition
                    
                    # Transform using trained encoder
                    try:
                        input_processed[feature] = self.label_encoders[feature].transform([input_processed[feature].iloc[0]])[0]
                    except:
                        input_processed[feature] = 0  # Unknown category
            
            # Prepare final features
            input_final = input_processed.reindex(columns=self.feature_names, fill_value=0)
            
            # Predict UberX price
            uberx_price = self.model.predict(input_final)[0]
            uberx_price = max(uberx_price, 2.50)  # Minimum fare
            
            # Calculate other service prices
            return {
                'uberx': round(uberx_price, 2),
                'uber_xl': round(uberx_price * self.service_multipliers['uber_xl'], 2),
                'uber_premier': round(uberx_price * self.service_multipliers['uber_premier'], 2),
                'premier_suv': round(uberx_price * self.service_multipliers['premier_suv'], 2)
            }
            
        except Exception as e:
            print(f"⚠️ Prediction error: {e}")
            # Fallback
            base_price = (2.50 + distance_km * 1.50) * self.calibration_factor
            return {
                'uberx': base_price * surge_multiplier,
                'uber_xl': base_price * self.service_multipliers['uber_xl'] * surge_multiplier,
                'uber_premier': base_price * self.service_multipliers['uber_premier'] * surge_multiplier,
                'premier_suv': base_price * self.service_multipliers['premier_suv'] * surge_multiplier
            }
    
    def save_model(self, filepath='calibrated_miami_model.pkl'):
        """Save the calibrated model"""
        if not self.is_trained:
            print("❌ Model not trained yet!")
            return
        
        model_data = {
            'model': self.model,
            'label_encoders': self.label_encoders,
            'feature_names': self.feature_names,
            'service_multipliers': self.service_multipliers,
            'calibration_factor': self.calibration_factor
        }
        
        joblib.dump(model_data, filepath)
        print(f"✅ Calibrated model saved to {filepath}")
    
    def load_model(self, filepath='calibrated_miami_model.pkl'):
        """Load the calibrated model"""
        try:
            model_data = joblib.load(filepath)
            self.model = model_data['model']
            self.label_encoders = model_data['label_encoders']
            self.feature_names = model_data['feature_names']
            self.service_multipliers = model_data['service_multipliers']
            self.calibration_factor = model_data['calibration_factor']
            self.is_trained = True
            print(f"✅ Calibrated model loaded from {filepath}")
            return True
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False

def main():
    """Train the calibrated Miami model"""
    print("🎯 CALIBRATED MIAMI UBER MODEL")
    print("="*50)
    
    # Initialize model
    model = CalibratedMiamiModel()
    
    # Train model
    success = model.train_model()
    
    if success:
        # Save model
        model.save_model()
        
        # Test with your example
        print(f"\n🧪 Testing with your example:")
        print("-"*30)
        
        # Test the trip that gave $14.10 vs real $10.98
        test_distance = 8.0  # Estimate based on your trip
        prices = model.predict_all_services(
            distance_km=test_distance,
            pickup_lat=25.7617,  # Downtown Miami
            pickup_lng=-80.1918,
            dropoff_lat=25.7907,  # South Beach
            dropoff_lng=-80.1300,
            hour_of_day=datetime.now().hour
        )
        
        print(f"Calibrated predictions:")
        for service, price in prices.items():
            print(f"  {service:15s}: ${price:.2f}")
        
        print(f"\n💡 The UberX price should now be closer to $10.98!")

if __name__ == "__main__":
    main() 