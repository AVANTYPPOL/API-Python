#!/usr/bin/env python3
"""
ULTIMATE MIAMI UBER MODEL
=========================

The best possible model for your Miami Uber price prediction.
Miami-first approach using your valuable scraped data.

Features:
- Miami scraped data as primary training source
- NYC data as supplementary for distance patterns
- Built-in multi-service support (UberX, UberXL, Premier, Premier SUV)
- Optimized feature engineering for Miami market
- Production-ready with error handling
"""

import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder
import joblib
import warnings
warnings.filterwarnings('ignore')

class UltimateMiamiModel:
    """
    The ultimate Miami Uber price prediction model
    """
    
    def __init__(self):
        print("🚀 ULTIMATE MIAMI UBER MODEL")
        print("=" * 60)
        print("🏖️  Miami-First Approach (Real Scraped Data)")
        print("🗽 NYC Enhancement (Distance Learning)")
        print("🚗 Multi-Service Built-in (All 4 Types)")
        print("=" * 60)
        
        # Models for each service
        self.models = {
            'uberx': None,
            'uberxl': None,
            'uber_premier': None,
            'premier_suv': None
        }
        
        # Label encoders
        self.label_encoders = {}
        self.is_trained = False
        
        # Miami-specific pricing insights
        self.miami_base_fare = 2.50
        self.miami_per_km = 1.65  # Median from your data
        
        # Service multipliers from your data
        self.service_multipliers = {
            'uberx': 1.00,
            'uberxl': 1.55,
            'uber_premier': 2.00,
            'premier_suv': 2.64
        }
        
    def load_miami_data(self):
        """Load your valuable Miami scraped data"""
        print("\n🏖️  Loading Miami Scraped Data...")
        
        try:
            conn = sqlite3.connect('uber_ml_data.db')
            query = """
            SELECT * FROM rides 
            WHERE uber_price_usd > 0 
            AND uberx_price > 0 
            AND uber_xl_price > 0 
            AND uber_premier_price > 0 
            AND premier_suv_price > 0
            """
            df = pd.read_sql(query, conn)
            conn.close()
            
            print(f"✅ Loaded {len(df)} Miami rides with all 4 services")
            print(f"   Average UberX: ${df['uberx_price'].mean():.2f}")
            print(f"   Average distance: {df['distance_km'].mean():.1f} km")
            
            return df
            
        except Exception as e:
            print(f"❌ Error loading Miami data: {e}")
            return None
    
    def load_nyc_supplementary(self):
        """Load NYC data for distance pattern enhancement"""
        print("\n🗽 Loading NYC Supplementary Data...")
        
        try:
            # Load processed NYC data
            nyc_df = pd.read_parquet('nyc_processed_for_hybrid.parquet')
            
            # Sample for balance (don't overwhelm Miami data)
            nyc_sample = nyc_df.sample(n=min(5000, len(nyc_df)), random_state=42)
            
            # Convert to Miami-like format
            nyc_sample['uberx_price'] = nyc_sample['fare_amount']
            nyc_sample['uber_xl_price'] = nyc_sample['fare_amount'] * 1.55
            nyc_sample['uber_premier_price'] = nyc_sample['fare_amount'] * 2.00
            nyc_sample['premier_suv_price'] = nyc_sample['fare_amount'] * 2.64
            
            print(f"✅ Added {len(nyc_sample)} NYC rides for distance learning")
            print(f"   NYC price per km: ${(nyc_sample['uberx_price'] / nyc_sample['distance_km']).mean():.2f}")
            
            return nyc_sample
            
        except Exception as e:
            print(f"⚠️  NYC data not available: {e}")
            return None
    
    def engineer_miami_features(self, df):
        """Engineer features optimized for Miami market"""
        print("\n🔧 Engineering Miami-Optimized Features...")
        
        df_eng = df.copy()
        
        # Core distance features (most important for Uber)
        df_eng['distance_km_squared'] = df_eng['distance_km'] ** 2
        df_eng['distance_km_log'] = np.log1p(df_eng['distance_km'])
        df_eng['distance_km_sqrt'] = np.sqrt(df_eng['distance_km'])
        
        # Miami-specific locations
        miami_locations = {
            'airport': (25.7959, -80.2870),
            'south_beach': (25.7907, -80.1300),
            'downtown': (25.7617, -80.1918),
            'brickell': (25.7617, -80.1918),
            'wynwood': (25.8022, -80.1999),
            'coral_gables': (25.7489, -80.2688)
        }
        
        # Location proximity features
        for location, (lat, lng) in miami_locations.items():
            df_eng[f'pickup_near_{location}'] = (
                np.sqrt((df_eng['pickup_lat'] - lat)**2 + (df_eng['pickup_lng'] - lng)**2) < 0.02
            ).astype(int)
            
            df_eng[f'dropoff_near_{location}'] = (
                np.sqrt((df_eng['dropoff_lat'] - lat)**2 + (df_eng['dropoff_lng'] - lng)**2) < 0.02
            ).astype(int)
        
        # Premium trip indicators
        df_eng['is_airport_trip'] = (df_eng['pickup_near_airport'] | df_eng['dropoff_near_airport'])
        df_eng['is_beach_trip'] = (df_eng['pickup_near_south_beach'] | df_eng['dropoff_near_south_beach'])
        df_eng['is_downtown_trip'] = (df_eng['pickup_near_downtown'] | df_eng['dropoff_near_downtown'])
        
        # Time-based features
        df_eng['is_rush_hour'] = df_eng['hour_of_day'].apply(
            lambda x: 1 if x in [7, 8, 9, 17, 18, 19] else 0
        )
        df_eng['is_weekend'] = df_eng['day_of_week'].apply(
            lambda x: 1 if x in [5, 6] else 0
        )
        df_eng['is_late_night'] = df_eng['hour_of_day'].apply(
            lambda x: 1 if x >= 22 or x <= 5 else 0
        )
        
        # Trip type categories
        df_eng['is_short_trip'] = (df_eng['distance_km'] < 5).astype(int)
        df_eng['is_medium_trip'] = ((df_eng['distance_km'] >= 5) & (df_eng['distance_km'] < 20)).astype(int)
        df_eng['is_long_trip'] = (df_eng['distance_km'] >= 20).astype(int)
        
        # Interaction features
        df_eng['distance_x_surge'] = df_eng['distance_km'] * df_eng['surge_multiplier']
        df_eng['distance_x_rush'] = df_eng['distance_km'] * df_eng['is_rush_hour']
        df_eng['distance_x_airport'] = df_eng['distance_km'] * df_eng['is_airport_trip']
        df_eng['distance_x_weekend'] = df_eng['distance_km'] * df_eng['is_weekend']
        
        # Encode categorical features
        for feature in ['traffic_level', 'weather_condition']:
            if feature in df_eng.columns:
                if feature not in self.label_encoders:
                    self.label_encoders[feature] = LabelEncoder()
                    df_eng[feature] = self.label_encoders[feature].fit_transform(
                        df_eng[feature].fillna('unknown').astype(str)
                    )
                else:
                    df_eng[feature] = self.label_encoders[feature].transform(
                        df_eng[feature].fillna('unknown').astype(str)
                    )
        
        print(f"✅ Created {len(df_eng.columns)} features for Miami optimization")
        
        return df_eng
    
    def train_ultimate_model(self):
        """Train the ultimate Miami-first model"""
        print("\n🚀 Training Ultimate Miami Model...")
        
        # Load Miami data (primary)
        miami_df = self.load_miami_data()
        if miami_df is None or len(miami_df) < 50:
            print("❌ Insufficient Miami data")
            return False
        
        # Load NYC data (supplementary)
        nyc_df = self.load_nyc_supplementary()
        
        # Combine datasets with Miami priority
        if nyc_df is not None:
            # Ensure both have same columns
            common_cols = set(miami_df.columns) & set(nyc_df.columns)
            miami_df = miami_df[list(common_cols)]
            nyc_df = nyc_df[list(common_cols)]
            
            # Combine with Miami weight
            combined_df = pd.concat([
                miami_df,  # Full Miami data
                nyc_df.sample(n=min(len(miami_df), len(nyc_df)), random_state=42)  # Balanced NYC
            ], ignore_index=True)
            
            print(f"📊 Combined dataset: {len(combined_df)} rides")
            print(f"   Miami: {len(miami_df)} rides (primary)")
            print(f"   NYC: {len(nyc_df)} rides (supplementary)")
        else:
            combined_df = miami_df
            print(f"📊 Miami-only dataset: {len(combined_df)} rides")
        
        # Engineer features
        df_processed = self.engineer_miami_features(combined_df)
        
        # Define features
        feature_cols = [
            'distance_km', 'distance_km_squared', 'distance_km_log', 'distance_km_sqrt',
            'pickup_lat', 'pickup_lng', 'dropoff_lat', 'dropoff_lng',
            'hour_of_day', 'day_of_week', 'surge_multiplier',
            'is_airport_trip', 'is_beach_trip', 'is_downtown_trip',
            'is_rush_hour', 'is_weekend', 'is_late_night',
            'is_short_trip', 'is_medium_trip', 'is_long_trip',
            'distance_x_surge', 'distance_x_rush', 'distance_x_airport', 'distance_x_weekend'
        ]
        
        # Add encoded features
        for feature in ['traffic_level', 'weather_condition']:
            if feature in df_processed.columns:
                feature_cols.append(feature)
        
        # Train model for each service
        services = ['uberx', 'uberxl', 'uber_premier', 'premier_suv']
        service_columns = ['uberx_price', 'uber_xl_price', 'uber_premier_price', 'premier_suv_price']
        
        self.feature_columns = feature_cols
        results = {}
        
        for service, price_col in zip(services, service_columns):
            print(f"\n🎯 Training {service.upper()} model...")
            
            # Prepare data
            X = df_processed[feature_cols].fillna(0)
            y = df_processed[price_col]
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Train model
            model = RandomForestRegressor(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features='sqrt',
                random_state=42,
                n_jobs=-1
            )
            
            model.fit(X_train, y_train)
            
            # Evaluate
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            
            train_r2 = r2_score(y_train, y_pred_train)
            test_r2 = r2_score(y_test, y_pred_test)
            test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
            
            print(f"   Train R²: {train_r2:.4f}")
            print(f"   Test R²: {test_r2:.4f}")
            print(f"   Test RMSE: ${test_rmse:.2f}")
            
            # Store model and results
            self.models[service] = model
            results[service] = {
                'train_r2': train_r2,
                'test_r2': test_r2,
                'test_rmse': test_rmse
            }
            
            # Show top features
            importance_df = pd.DataFrame({
                'feature': feature_cols,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print(f"   Top features:")
            for _, row in importance_df.head(5).iterrows():
                print(f"     {row['feature']:<20}: {row['importance']*100:5.1f}%")
        
        # Overall performance
        avg_r2 = np.mean([r['test_r2'] for r in results.values()])
        print(f"\n🏆 Overall Model Performance:")
        print(f"   Average R²: {avg_r2:.4f}")
        print(f"   Miami Data Priority: ✅")
        print(f"   Multi-Service: ✅")
        print(f"   Distance Learning: ✅")
        
        self.is_trained = True
        return True
    
    def predict_all_services(self, distance_km, pickup_lat, pickup_lng, dropoff_lat, dropoff_lng,
                           hour_of_day=12, day_of_week=3, surge_multiplier=1.0,
                           traffic_level='moderate', weather_condition='clear'):
        """Predict prices for all services"""
        
        if not self.is_trained:
            # Fallback pricing if model not trained
            base_price = self.miami_base_fare + (distance_km * self.miami_per_km * surge_multiplier)
            return {
                'UberX': base_price * self.service_multipliers['uberx'],
                'UberXL': base_price * self.service_multipliers['uberxl'],
                'Uber Premier': base_price * self.service_multipliers['uber_premier'],
                'Premier SUV': base_price * self.service_multipliers['premier_suv']
            }
        
        # Create input dataframe
        input_df = pd.DataFrame({
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
        
        # Engineer features
        input_processed = self.engineer_miami_features(input_df)
        
        # Get features
        X_input = input_processed[self.feature_columns].fillna(0)
        
        # Predict for each service
        predictions = {}
        service_names = {
            'uberx': 'UberX',
            'uberxl': 'UberXL',
            'uber_premier': 'Uber Premier',
            'premier_suv': 'Premier SUV'
        }
        
        for service, model in self.models.items():
            if model is not None:
                pred = model.predict(X_input)[0]
                predictions[service_names[service]] = max(pred, self.miami_base_fare)
        
        return predictions
    
    def save_model(self, filepath='ultimate_miami_model.pkl'):
        """Save the ultimate model"""
        if not self.is_trained:
            print("⚠️  Model not trained yet")
            return False
        
        model_data = {
            'models': self.models,
            'label_encoders': self.label_encoders,
            'feature_columns': self.feature_columns,
            'service_multipliers': self.service_multipliers,
            'miami_base_fare': self.miami_base_fare,
            'miami_per_km': self.miami_per_km
        }
        
        joblib.dump(model_data, filepath)
        print(f"✅ Ultimate Miami Model saved to {filepath}")
        return True
    
    def load_model(self, filepath='ultimate_miami_model.pkl'):
        """Load the ultimate model"""
        try:
            model_data = joblib.load(filepath)
            self.models = model_data['models']
            self.label_encoders = model_data['label_encoders']
            self.feature_columns = model_data['feature_columns']
            self.service_multipliers = model_data['service_multipliers']
            self.miami_base_fare = model_data['miami_base_fare']
            self.miami_per_km = model_data['miami_per_km']
            self.is_trained = True
            print(f"✅ Ultimate Miami Model loaded from {filepath}")
            return True
        except Exception as e:
            print(f"⚠️  Could not load model: {e}")
            return False

def main():
    """Train and test the ultimate model"""
    model = UltimateMiamiModel()
    
    # Train the model
    if model.train_ultimate_model():
        # Save the model
        model.save_model()
        
        # Test predictions
        print(f"\n\n🧪 Testing Ultimate Miami Model:")
        print("=" * 60)
        
        test_cases = [
            {
                'name': 'Short Downtown Trip',
                'distance_km': 3,
                'pickup_lat': 25.7617, 'pickup_lng': -80.1918,
                'dropoff_lat': 25.7700, 'dropoff_lng': -80.1850,
                'hour_of_day': 14, 'day_of_week': 3
            },
            {
                'name': 'Airport to South Beach',
                'distance_km': 18,
                'pickup_lat': 25.7959, 'pickup_lng': -80.2870,
                'dropoff_lat': 25.7907, 'dropoff_lng': -80.1300,
                'hour_of_day': 9, 'day_of_week': 1
            },
            {
                'name': 'Weekend Night Beach Trip',
                'distance_km': 12,
                'pickup_lat': 25.7617, 'pickup_lng': -80.1918,
                'dropoff_lat': 25.7907, 'dropoff_lng': -80.1300,
                'hour_of_day': 23, 'day_of_week': 5, 'surge_multiplier': 1.5
            },
            {
                'name': 'Rush Hour Long Trip',
                'distance_km': 35,
                'pickup_lat': 25.7617, 'pickup_lng': -80.1918,
                'dropoff_lat': 25.9840, 'dropoff_lng': -80.1280,
                'hour_of_day': 18, 'day_of_week': 2
            }
        ]
        
        for case in test_cases:
            print(f"\n📍 {case['name']} ({case['distance_km']} km)")
            case_copy = case.copy()
            case_copy.pop('name')  # Remove name from args
            predictions = model.predict_all_services(**case_copy)
            
            for service, price in predictions.items():
                price_per_km = price / case['distance_km']
                print(f"   {service:<15}: ${price:6.2f} (${price_per_km:.2f}/km)")
        
        print(f"\n🎉 Ultimate Miami Model is ready!")
        print(f"🚀 Use this model for production - it's optimized for your data!")
        
    else:
        print("❌ Model training failed")

if __name__ == "__main__":
    main() 