"""
Multi-Service Uber Price Prediction Model
========================================

Predicts prices for all 4 Uber service types:
- UberX (standard)
- UberXL (larger vehicles) 
- Uber Premier (premium vehicles)
- Premier SUV (luxury SUVs)

Uses multi-output regression with service-specific adjustments.

Author: AI Assistant
Date: 2024
"""

import pandas as pd
import numpy as np
import sqlite3
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("🚗 MULTI-SERVICE UBER PRICE PREDICTION MODEL")
print("=" * 70)
print("📱 Predicting prices for: UberX, UberXL, Premier, Premier SUV")
print("🎯 Optimized for your ride-sharing app")
print("=" * 70)

class MultiServiceUberModel:
    """
    Multi-output model for predicting all Uber service prices
    """
    
    def __init__(self, db_path='uber_ml_data.db'):
        self.db_path = db_path
        
        # Models
        self.base_model = None  # Predicts UberX
        self.multi_model = None  # Predicts all services
        
        # Service multipliers from data analysis
        self.service_multipliers = {
            'uberx': 1.00,
            'uber_xl': 1.55,
            'uber_premier': 2.00,
            'premier_suv': 2.64
        }
        
        # Adjustable multipliers based on features
        self.dynamic_multipliers = {}
        
        # Encoders and scalers
        self.label_encoders = {}
        self.scaler = StandardScaler()
        
        # Feature names
        self.feature_names = None
        self.service_names = ['uberx', 'uber_xl', 'uber_premier', 'premier_suv']
        
        # Training status
        self.is_trained = False
        
    def load_data(self):
        """Load and prepare multi-service data"""
        print("\n📥 Loading Multi-Service Data...")
        
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Load data with all service prices
            query = """
            SELECT distance_km, pickup_lat, pickup_lng, dropoff_lat, dropoff_lng,
                   hour_of_day, day_of_week, surge_multiplier, traffic_level, weather_condition,
                   uberx_price, uber_xl_price, uber_premier_price, premier_suv_price
            FROM rides 
            WHERE uberx_price > 0 AND uber_xl_price > 0 
                  AND uber_premier_price > 0 AND premier_suv_price > 0
            """
            
            df = pd.read_sql(query, conn)
            conn.close()
            
            print(f"✅ Loaded {len(df)} complete multi-service records")
            
            # Show service price statistics
            print("\n📊 Service Price Summary:")
            for service in self.service_names:
                price_col = f"{service}_price"
                mean_price = df[price_col].mean()
                print(f"  {service}: ${mean_price:.2f} avg")
            
            return df
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return None
    
    def preprocess_features(self, df):
        """Prepare features for multi-output prediction"""
        df_processed = df.copy()
        
        # Handle missing values
        df_processed['traffic_level'].fillna('light', inplace=True)
        df_processed['weather_condition'].fillna('clear', inplace=True)
        df_processed['surge_multiplier'].fillna(1.0, inplace=True)
        
        # Encode categorical features
        for feature in ['traffic_level', 'weather_condition']:
            if feature not in self.label_encoders:
                self.label_encoders[feature] = LabelEncoder()
                df_processed[feature] = self.label_encoders[feature].fit_transform(df_processed[feature])
            else:
                # Handle new categories
                df_processed[feature] = df_processed[feature].map(
                    lambda x: self.label_encoders[feature].transform([x])[0] 
                    if x in self.label_encoders[feature].classes_ else 0
                )
        
        # Engineer features
        df_processed = self._engineer_features(df_processed)
        
        # Separate features and targets
        feature_cols = [col for col in df_processed.columns if not col.endswith('_price')]
        target_cols = [f"{service}_price" for service in self.service_names]
        
        X = df_processed[feature_cols]
        y = df_processed[target_cols]
        
        self.feature_names = feature_cols
        
        return X, y
    
    def _engineer_features(self, df):
        """Engineer features with service-specific considerations"""
        df_eng = df.copy()
        
        # Distance features (most important)
        df_eng['distance_km_squared'] = df_eng['distance_km'] ** 2
        df_eng['distance_km_log'] = np.log1p(df_eng['distance_km'])
        df_eng['distance_km_sqrt'] = np.sqrt(df_eng['distance_km'])
        
        # Distance categories
        df_eng['is_short_trip'] = (df_eng['distance_km'] < 5).astype(int)
        df_eng['is_medium_trip'] = ((df_eng['distance_km'] >= 5) & (df_eng['distance_km'] < 15)).astype(int)
        df_eng['is_long_trip'] = (df_eng['distance_km'] >= 15).astype(int)
        
        # Location features
        # Miami key locations
        airport_lat, airport_lng = 25.7959, -80.2870
        downtown_lat, downtown_lng = 25.7617, -80.1918
        beach_lat, beach_lng = 25.7907, -80.1300
        
        # Airport proximity (premium services often preferred)
        df_eng['pickup_airport_dist'] = np.sqrt(
            (df_eng['pickup_lat'] - airport_lat) ** 2 +
            (df_eng['pickup_lng'] - airport_lng) ** 2
        )
        df_eng['dropoff_airport_dist'] = np.sqrt(
            (df_eng['dropoff_lat'] - airport_lat) ** 2 +
            (df_eng['dropoff_lng'] - airport_lng) ** 2
        )
        df_eng['is_airport_trip'] = (
            (df_eng['pickup_airport_dist'] < 0.02) | 
            (df_eng['dropoff_airport_dist'] < 0.02)
        ).astype(int)
        
        # Downtown proximity (business trips)
        df_eng['pickup_downtown_dist'] = np.sqrt(
            (df_eng['pickup_lat'] - downtown_lat) ** 2 +
            (df_eng['pickup_lng'] - downtown_lng) ** 2
        )
        df_eng['is_downtown_trip'] = (df_eng['pickup_downtown_dist'] < 0.05).astype(int)
        
        # Beach proximity (tourist/leisure)
        df_eng['pickup_beach_dist'] = np.sqrt(
            (df_eng['pickup_lat'] - beach_lat) ** 2 +
            (df_eng['pickup_lng'] - beach_lng) ** 2
        )
        df_eng['is_beach_trip'] = (df_eng['pickup_beach_dist'] < 0.05).astype(int)
        
        # Time features
        # Rush hours (XL might be preferred)
        df_eng['is_morning_rush'] = df_eng['hour_of_day'].apply(
            lambda x: 1 if 7 <= x <= 9 else 0
        )
        df_eng['is_evening_rush'] = df_eng['hour_of_day'].apply(
            lambda x: 1 if 17 <= x <= 19 else 0
        )
        df_eng['is_rush_hour'] = (df_eng['is_morning_rush'] | df_eng['is_evening_rush']).astype(int)
        
        # Late night (premium services surge more)
        df_eng['is_late_night'] = df_eng['hour_of_day'].apply(
            lambda x: 1 if x >= 22 or x <= 5 else 0
        )
        
        # Weekend nights (party hours - XL and Premier popular)
        df_eng['is_weekend'] = df_eng['day_of_week'].apply(lambda x: 1 if x in [5, 6] else 0)
        df_eng['is_weekend_night'] = (df_eng['is_weekend'] & df_eng['is_late_night']).astype(int)
        
        # Service preference indicators
        # Long trips might prefer comfort (Premier/SUV)
        df_eng['prefers_comfort'] = (df_eng['distance_km'] > 20).astype(int)
        
        # Groups might prefer XL
        df_eng['likely_group'] = (df_eng['is_weekend_night'] | df_eng['is_evening_rush']).astype(int)
        
        # Business indicators (Premier preference)
        df_eng['likely_business'] = (
            df_eng['is_downtown_trip'] & 
            (df_eng['hour_of_day'].between(8, 18)) &
            (df_eng['day_of_week'] < 5)
        ).astype(int)
        
        return df_eng
    
    def train_models(self):
        """Train multi-output model for all services"""
        print("\n🚀 Training Multi-Service Model...")
        
        # Load data
        df = self.load_data()
        if df is None:
            return False
        
        # Preprocess
        X, y = self.preprocess_features(df)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        print(f"\n📊 Training on {len(X_train)} samples, testing on {len(X_test)} samples")
        
        # Train base model (UberX only)
        print("\n1️⃣ Training Base Model (UberX)...")
        self.base_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        # Train on UberX prices
        self.base_model.fit(X_train, y_train['uberx_price'])
        
        # Evaluate base model
        uberx_pred = self.base_model.predict(X_test)
        uberx_r2 = r2_score(y_test['uberx_price'], uberx_pred)
        uberx_rmse = np.sqrt(mean_squared_error(y_test['uberx_price'], uberx_pred))
        
        print(f"   UberX R²: {uberx_r2:.4f}")
        print(f"   UberX RMSE: ${uberx_rmse:.2f}")
        
        # Train multi-output model
        print("\n2️⃣ Training Multi-Output Model (All Services)...")
        
        # Create base estimator
        base_estimator = RandomForestRegressor(
            n_estimators=50,  # Less trees per output
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        
        # Multi-output wrapper
        self.multi_model = MultiOutputRegressor(base_estimator, n_jobs=-1)
        
        # Train on all services
        self.multi_model.fit(X_train, y_train)
        
        # Evaluate multi-output model
        y_pred = self.multi_model.predict(X_test)
        
        print("\n📈 Multi-Service Model Performance:")
        print("-" * 50)
        
        service_performance = {}
        for i, service in enumerate(self.service_names):
            service_true = y_test.iloc[:, i]
            service_pred = y_pred[:, i]
            
            r2 = r2_score(service_true, service_pred)
            rmse = np.sqrt(mean_squared_error(service_true, service_pred))
            mae = mean_absolute_error(service_true, service_pred)
            
            service_performance[service] = {
                'r2': r2,
                'rmse': rmse,
                'mae': mae
            }
            
            print(f"{service:15s}: R²={r2:.3f}, RMSE=${rmse:.2f}, MAE=${mae:.2f}")
        
        # Calculate dynamic multipliers based on features
        self._calculate_dynamic_multipliers(X_train, y_train)
        
        # Feature importance
        print("\n🏆 Top Features for Base Model:")
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': self.base_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        for _, row in feature_importance.head(10).iterrows():
            print(f"  {row['feature']:25s}: {row['importance']*100:.1f}%")
        
        self.is_trained = True
        
        return service_performance
    
    def _calculate_dynamic_multipliers(self, X, y):
        """Calculate feature-based multiplier adjustments"""
        print("\n🔧 Calculating Dynamic Multipliers...")
        
        # For each service, calculate how multipliers change with features
        for service in self.service_names[1:]:  # Skip UberX (base)
            service_col = f"{service}_price"
            
            # Calculate ratios
            ratios = y[service_col] / y['uberx_price']
            
            # Analyze how ratios change with key features
            adjustments = {}
            
            # Airport trips tend to have higher multipliers for premium
            if 'is_airport_trip' in X.columns:
                airport_mask = X['is_airport_trip'] == 1
                if airport_mask.sum() > 10:
                    airport_ratio = ratios[airport_mask].mean()
                    regular_ratio = ratios[~airport_mask].mean()
                    adjustments['airport'] = airport_ratio / regular_ratio
            
            # Long trips might prefer comfort
            if 'is_long_trip' in X.columns:
                long_mask = X['is_long_trip'] == 1
                if long_mask.sum() > 10:
                    long_ratio = ratios[long_mask].mean()
                    short_ratio = ratios[~long_mask].mean()
                    adjustments['long_trip'] = long_ratio / short_ratio
            
            # Weekend nights
            if 'is_weekend_night' in X.columns:
                weekend_mask = X['is_weekend_night'] == 1
                if weekend_mask.sum() > 10:
                    weekend_ratio = ratios[weekend_mask].mean()
                    regular_ratio = ratios[~weekend_mask].mean()
                    adjustments['weekend_night'] = weekend_ratio / regular_ratio
            
            self.dynamic_multipliers[service] = adjustments
        
        # Display adjustments
        for service, adjustments in self.dynamic_multipliers.items():
            print(f"\n{service} multiplier adjustments:")
            for feature, adjustment in adjustments.items():
                print(f"  {feature}: {adjustment:.2f}x")
    
    def predict_all_services(self, distance_km, pickup_lat, pickup_lng, 
                           dropoff_lat, dropoff_lng, hour_of_day=12, 
                           day_of_week=0, surge_multiplier=1.0,
                           traffic_level='light', weather_condition='clear'):
        """
        Predict prices for all 4 services
        
        Returns:
            dict: Prices for each service
        """
        if not self.is_trained:
            # Fallback to simple multiplier-based prediction
            base_price = 2.50 + (distance_km * 1.86)
            return {
                'uberx': base_price * surge_multiplier,
                'uber_xl': base_price * self.service_multipliers['uber_xl'] * surge_multiplier,
                'uber_premier': base_price * self.service_multipliers['uber_premier'] * surge_multiplier,
                'premier_suv': base_price * self.service_multipliers['premier_suv'] * surge_multiplier
            }
        
        try:
            # Create input DataFrame
            input_data = pd.DataFrame({
                'distance_km': [distance_km],
                'pickup_lat': [pickup_lat],
                'pickup_lng': [pickup_lng],
                'dropoff_lat': [dropoff_lat],
                'dropoff_lng': [dropoff_lng],
                'hour_of_day': [hour_of_day],
                'day_of_week': [day_of_week],
                'surge_multiplier': [surge_multiplier],
                'traffic_level': [traffic_level],
                'weather_condition': [weather_condition]
            })
            
            # Encode categorical features
            for feature in ['traffic_level', 'weather_condition']:
                if feature in self.label_encoders:
                    input_data[feature] = self.label_encoders[feature].transform(input_data[feature])
            
            # Engineer features
            input_processed = self._engineer_features(input_data)
            
            # Ensure all features are present
            input_processed = input_processed.reindex(columns=self.feature_names, fill_value=0)
            
            # Get predictions
            predictions = self.multi_model.predict(input_processed)[0]
            
            # Apply surge multiplier (if not already in features)
            predictions *= surge_multiplier
            
            # Create result dictionary
            result = {}
            for i, service in enumerate(self.service_names):
                # Ensure minimum fare
                price = max(predictions[i], 2.50)
                
                # Apply any dynamic adjustments
                if service in self.dynamic_multipliers:
                    if 'is_airport_trip' in input_processed.columns and input_processed['is_airport_trip'].iloc[0] == 1:
                        if 'airport' in self.dynamic_multipliers[service]:
                            price *= self.dynamic_multipliers[service]['airport']
                
                result[service] = round(price, 2)
            
            return result
            
        except Exception as e:
            print(f"⚠️ Prediction error: {e}")
            # Fallback prediction
            base_price = 2.50 + (distance_km * 1.86)
            return {
                'uberx': base_price * surge_multiplier,
                'uber_xl': base_price * self.service_multipliers['uber_xl'] * surge_multiplier,
                'uber_premier': base_price * self.service_multipliers['uber_premier'] * surge_multiplier,
                'premier_suv': base_price * self.service_multipliers['premier_suv'] * surge_multiplier
            }
    
    def save_model(self, filepath='multi_service_uber_model.pkl'):
        """Save the multi-service model"""
        if not self.is_trained:
            print("❌ Model not trained yet!")
            return
        
        model_data = {
            'base_model': self.base_model,
            'multi_model': self.multi_model,
            'label_encoders': self.label_encoders,
            'feature_names': self.feature_names,
            'service_names': self.service_names,
            'service_multipliers': self.service_multipliers,
            'dynamic_multipliers': self.dynamic_multipliers
        }
        
        joblib.dump(model_data, filepath)
        print(f"✅ Multi-service model saved to {filepath}")
    
    def load_model(self, filepath='multi_service_uber_model.pkl'):
        """Load the multi-service model"""
        try:
            model_data = joblib.load(filepath)
            
            self.base_model = model_data['base_model']
            self.multi_model = model_data['multi_model']
            self.label_encoders = model_data['label_encoders']
            self.feature_names = model_data['feature_names']
            self.service_names = model_data['service_names']
            self.service_multipliers = model_data['service_multipliers']
            self.dynamic_multipliers = model_data.get('dynamic_multipliers', {})
            
            self.is_trained = True
            print(f"✅ Multi-service model loaded from {filepath}")
            return True
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False

def main():
    """Train and test the multi-service model"""
    print("🚀 Multi-Service Uber Model Training")
    
    # Initialize model
    model = MultiServiceUberModel()
    
    # Train models
    performance = model.train_models()
    
    if performance:
        # Save model
        model.save_model()
        
        # Test predictions
        print("\n🎯 Testing Multi-Service Predictions:")
        print("=" * 70)
        
        test_cases = [
            {
                'desc': "Short downtown trip (3km)",
                'distance_km': 3,
                'pickup_lat': 25.7617, 'pickup_lng': -80.1918,
                'dropoff_lat': 25.7750, 'dropoff_lng': -80.1900,
                'hour_of_day': 14
            },
            {
                'desc': "Airport to Beach (15km)",
                'distance_km': 15,
                'pickup_lat': 25.7959, 'pickup_lng': -80.2870,
                'dropoff_lat': 25.7907, 'dropoff_lng': -80.1300,
                'hour_of_day': 19
            },
            {
                'desc': "Long weekend night trip (35km)",
                'distance_km': 35,
                'pickup_lat': 25.8000, 'pickup_lng': -80.2000,
                'dropoff_lat': 26.1000, 'dropoff_lng': -80.1500,
                'hour_of_day': 23,
                'day_of_week': 5
            }
        ]
        
        for test in test_cases:
            print(f"\n📍 {test['desc']}")
            
            prices = model.predict_all_services(
                distance_km=test['distance_km'],
                pickup_lat=test['pickup_lat'],
                pickup_lng=test['pickup_lng'],
                dropoff_lat=test['dropoff_lat'],
                dropoff_lng=test['dropoff_lng'],
                hour_of_day=test.get('hour_of_day', 12),
                day_of_week=test.get('day_of_week', 0),
                surge_multiplier=1.0
            )
            
            print("  Prices:")
            for service, price in prices.items():
                print(f"    {service:15s}: ${price:.2f}")
        
        print("\n✅ Multi-service model ready for deployment!")

if __name__ == "__main__":
    main() 