"""
Miami-Only Uber Price Prediction Model
======================================

Pure Miami model trained exclusively on local data.
No NYC influence - just Miami market pricing patterns.

Author: AI Assistant
Date: 2024
"""

import sqlite3
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import joblib
import warnings
warnings.filterwarnings('ignore')

class MiamiOnlyModel:
    def __init__(self):
        print("🏖️  MIAMI-ONLY UBER MODEL")
        print("=" * 50)
        print("🎯 Training exclusively on Miami data")
        print("📊 No NYC influence - pure local market")
        print("=" * 50)
        
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        
    def load_data(self):
        """Load Miami data from database"""
        print("📥 Loading Miami Dataset...")
        
        conn = sqlite3.connect('uber_ml_data.db')
        
        # Load all Miami data
        query = """
        SELECT * FROM rides 
        WHERE uberx_price > 0 
        AND uber_xl_price > 0 
        AND uber_premier_price > 0 
        AND premier_suv_price > 0
        ORDER BY id
        """
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        print(f"✅ Loaded {len(df)} Miami rides")
        print(f"📅 Data range: {df['created_at'].min()} to {df['created_at'].max()}")
        print(f"💰 Avg UberX: ${df['uberx_price'].mean():.2f}")
        print(f"📏 Avg distance: {df['distance_km'].mean():.1f} km")
        
        return df
    
    def engineer_features(self, df):
        """Create Miami-specific features"""
        print("🔧 Engineering Miami-specific features...")
        
        # Copy dataframe
        df = df.copy()
        
        # Distance transformations
        df['distance_km_log'] = np.log1p(df['distance_km'])
        df['distance_km_sqrt'] = np.sqrt(df['distance_km'])
        df['distance_km_squared'] = df['distance_km'] ** 2
        
        # Miami-specific geographic features
        # Downtown Miami center: 25.7617, -80.1918
        downtown_lat, downtown_lng = 25.7617, -80.1918
        df['pickup_downtown_dist'] = np.sqrt(
            (df['pickup_lat'] - downtown_lat)**2 + 
            (df['pickup_lng'] - downtown_lng)**2
        )
        df['dropoff_downtown_dist'] = np.sqrt(
            (df['dropoff_lat'] - downtown_lat)**2 + 
            (df['dropoff_lng'] - downtown_lng)**2
        )
        
        # Miami Airport: 25.7932, -80.2906
        airport_lat, airport_lng = 25.7932, -80.2906
        df['pickup_airport_dist'] = np.sqrt(
            (df['pickup_lat'] - airport_lat)**2 + 
            (df['pickup_lng'] - airport_lng)**2
        )
        df['dropoff_airport_dist'] = np.sqrt(
            (df['dropoff_lat'] - airport_lat)**2 + 
            (df['dropoff_lng'] - airport_lng)**2
        )
        
        # South Beach: 25.7907, -80.1300
        beach_lat, beach_lng = 25.7907, -80.1300
        df['pickup_beach_dist'] = np.sqrt(
            (df['pickup_lat'] - beach_lat)**2 + 
            (df['pickup_lng'] - beach_lng)**2
        )
        df['dropoff_beach_dist'] = np.sqrt(
            (df['dropoff_lat'] - beach_lat)**2 + 
            (df['dropoff_lng'] - beach_lng)**2
        )
        
        # Time features
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['is_rush_hour'] = ((df['hour_of_day'] >= 7) & (df['hour_of_day'] <= 9) | 
                              (df['hour_of_day'] >= 17) & (df['hour_of_day'] <= 19)).astype(int)
        df['is_night'] = ((df['hour_of_day'] >= 22) | (df['hour_of_day'] <= 5)).astype(int)
        
        # Traffic encoding
        traffic_map = {'light': 1, 'moderate': 2, 'heavy': 3}
        df['traffic_encoded'] = df['traffic_level'].map(traffic_map).fillna(2)
        
        # Weather encoding
        weather_map = {'clear': 1, 'cloudy': 2, 'rainy': 3}
        df['weather_encoded'] = df['weather_condition'].map(weather_map).fillna(1)
        
        # Trip type features
        df['is_airport_trip'] = ((df['pickup_airport_dist'] < 0.02) | 
                                (df['dropoff_airport_dist'] < 0.02)).astype(int)
        df['is_beach_trip'] = ((df['pickup_beach_dist'] < 0.02) | 
                              (df['dropoff_beach_dist'] < 0.02)).astype(int)
        df['is_downtown_trip'] = ((df['pickup_downtown_dist'] < 0.02) | 
                                 (df['dropoff_downtown_dist'] < 0.02)).astype(int)
        
        # Long trip indicator
        df['is_long_trip'] = (df['distance_km'] > 20).astype(int)
        
        print(f"✅ Created {len([col for col in df.columns if col not in ['id', 'created_at']])} features")
        
        return df
    
    def train_model(self, df):
        """Train Miami-only model"""
        print("\n🚀 Training Miami-Only Model...")
        
        # Select features
        feature_cols = [
            'distance_km', 'distance_km_log', 'distance_km_sqrt', 'distance_km_squared',
            'pickup_lat', 'pickup_lng', 'dropoff_lat', 'dropoff_lng',
            'pickup_downtown_dist', 'dropoff_downtown_dist',
            'pickup_airport_dist', 'dropoff_airport_dist',
            'pickup_beach_dist', 'dropoff_beach_dist',
            'hour_of_day', 'day_of_week', 'surge_multiplier',
            'traffic_encoded', 'weather_encoded',
            'is_weekend', 'is_rush_hour', 'is_night',
            'is_airport_trip', 'is_beach_trip', 'is_downtown_trip', 'is_long_trip'
        ]
        
        # Prepare data
        X = df[feature_cols]
        y = df['uberx_price']
        
        self.feature_names = feature_cols
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        print(f"📊 Training on {len(X_train)} samples, testing on {len(X_test)} samples")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.model = RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        train_pred = self.model.predict(X_train_scaled)
        test_pred = self.model.predict(X_test_scaled)
        
        train_r2 = r2_score(y_train, train_pred)
        test_r2 = r2_score(y_test, test_pred)
        test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
        test_mae = mean_absolute_error(y_test, test_pred)
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=5, scoring='r2')
        
        print(f"\n📈 Miami-Only Model Performance:")
        print(f"   Training R²: {train_r2:.4f}")
        print(f"   Test R²: {test_r2:.4f}")
        print(f"   Test RMSE: ${test_rmse:.2f}")
        print(f"   Test MAE: ${test_mae:.2f}")
        print(f"   CV R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\n🏆 Top Features for Miami Model:")
        for _, row in feature_importance.head(10).iterrows():
            print(f"   {row['feature']:<20} : {row['importance']*100:.1f}%")
        
        return {
            'train_r2': train_r2,
            'test_r2': test_r2,
            'test_rmse': test_rmse,
            'test_mae': test_mae,
            'cv_r2_mean': cv_scores.mean(),
            'cv_r2_std': cv_scores.std()
        }
    
    def train_multi_service_model(self, df):
        """Train models for all service types"""
        print("\n🚀 Training Multi-Service Miami Models...")
        
        # Service multipliers based on Miami data
        service_multipliers = {
            'uberx': 1.0,
            'uber_xl': df['uber_xl_price'].mean() / df['uberx_price'].mean(),
            'uber_premier': df['uber_premier_price'].mean() / df['uberx_price'].mean(),
            'premier_suv': df['premier_suv_price'].mean() / df['uberx_price'].mean()
        }
        
        print(f"📊 Miami Service Multipliers:")
        for service, mult in service_multipliers.items():
            print(f"   {service}: {mult:.2f}x")
        
        # Store multipliers
        self.service_multipliers = service_multipliers
        
        # Test multi-service predictions
        sample_features = df[self.feature_names].iloc[0:1]
        sample_scaled = self.scaler.transform(sample_features)
        base_pred = self.model.predict(sample_scaled)[0]
        
        print(f"\n🎯 Sample Multi-Service Predictions:")
        for service, mult in service_multipliers.items():
            price = base_pred * mult
            print(f"   {service}: ${price:.2f}")
    
    def predict(self, pickup_lat, pickup_lng, dropoff_lat, dropoff_lng, 
                distance_km, hour_of_day=12, day_of_week=2, surge_multiplier=1.0,
                traffic_level='moderate', weather_condition='clear'):
        """Make prediction for a single trip"""
        
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        # Create feature vector
        features = pd.DataFrame({
            'distance_km': [distance_km],
            'distance_km_log': [np.log1p(distance_km)],
            'distance_km_sqrt': [np.sqrt(distance_km)],
            'distance_km_squared': [distance_km ** 2],
            'pickup_lat': [pickup_lat],
            'pickup_lng': [pickup_lng],
            'dropoff_lat': [dropoff_lat],
            'dropoff_lng': [dropoff_lng],
            'hour_of_day': [hour_of_day],
            'day_of_week': [day_of_week],
            'surge_multiplier': [surge_multiplier]
        })
        
        # Miami-specific distances
        downtown_lat, downtown_lng = 25.7617, -80.1918
        airport_lat, airport_lng = 25.7932, -80.2906
        beach_lat, beach_lng = 25.7907, -80.1300
        
        features['pickup_downtown_dist'] = np.sqrt(
            (pickup_lat - downtown_lat)**2 + (pickup_lng - downtown_lng)**2
        )
        features['dropoff_downtown_dist'] = np.sqrt(
            (dropoff_lat - downtown_lat)**2 + (dropoff_lng - downtown_lng)**2
        )
        features['pickup_airport_dist'] = np.sqrt(
            (pickup_lat - airport_lat)**2 + (pickup_lng - airport_lng)**2
        )
        features['dropoff_airport_dist'] = np.sqrt(
            (dropoff_lat - airport_lat)**2 + (dropoff_lng - airport_lng)**2
        )
        features['pickup_beach_dist'] = np.sqrt(
            (pickup_lat - beach_lat)**2 + (pickup_lng - beach_lng)**2
        )
        features['dropoff_beach_dist'] = np.sqrt(
            (dropoff_lat - beach_lat)**2 + (dropoff_lng - beach_lng)**2
        )
        
        # Time features
        features['is_weekend'] = int(day_of_week >= 5)
        features['is_rush_hour'] = int((hour_of_day >= 7 and hour_of_day <= 9) or 
                                     (hour_of_day >= 17 and hour_of_day <= 19))
        features['is_night'] = int((hour_of_day >= 22) or (hour_of_day <= 5))
        
        # Encoded features
        traffic_map = {'light': 1, 'moderate': 2, 'heavy': 3}
        weather_map = {'clear': 1, 'cloudy': 2, 'rainy': 3}
        features['traffic_encoded'] = traffic_map.get(traffic_level, 2)
        features['weather_encoded'] = weather_map.get(weather_condition, 1)
        
        # Trip type features
        features['is_airport_trip'] = int((features['pickup_airport_dist'].iloc[0] < 0.02) or 
                                         (features['dropoff_airport_dist'].iloc[0] < 0.02))
        features['is_beach_trip'] = int((features['pickup_beach_dist'].iloc[0] < 0.02) or 
                                       (features['dropoff_beach_dist'].iloc[0] < 0.02))
        features['is_downtown_trip'] = int((features['pickup_downtown_dist'].iloc[0] < 0.02) or 
                                          (features['dropoff_downtown_dist'].iloc[0] < 0.02))
        features['is_long_trip'] = int(distance_km > 20)
        
        # Scale features
        features_scaled = self.scaler.transform(features[self.feature_names])
        
        # Predict
        base_price = self.model.predict(features_scaled)[0]
        
        # Multi-service predictions
        predictions = {}
        for service, mult in self.service_multipliers.items():
            predictions[service] = base_price * mult
        
        return predictions
    
    def save_model(self, filename='miami_only_model.pkl'):
        """Save the trained model"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'service_multipliers': self.service_multipliers
        }
        joblib.dump(model_data, filename)
        print(f"✅ Miami-only model saved to {filename}")
    
    def load_model(self, filename='miami_only_model.pkl'):
        """Load a trained model"""
        model_data = joblib.load(filename)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.service_multipliers = model_data['service_multipliers']
        print(f"✅ Miami-only model loaded from {filename}")

def main():
    """Main training pipeline"""
    
    # Initialize model
    model = MiamiOnlyModel()
    
    # Load data
    df = model.load_data()
    
    # Engineer features
    df = model.engineer_features(df)
    
    # Train model
    performance = model.train_model(df)
    
    # Train multi-service
    model.train_multi_service_model(df)
    
    # Save model
    model.save_model()
    
    # Test predictions
    print(f"\n🧪 Testing Miami-Only Model:")
    print("=" * 50)
    
    test_cases = [
        {
            'name': 'Downtown to Airport',
            'pickup_lat': 25.7617, 'pickup_lng': -80.1918,
            'dropoff_lat': 25.7932, 'dropoff_lng': -80.2906,
            'distance_km': 12.5
        },
        {
            'name': 'Airport to South Beach',
            'pickup_lat': 25.7932, 'pickup_lng': -80.2906,
            'dropoff_lat': 25.7907, 'dropoff_lng': -80.1300,
            'distance_km': 18.2
        },
        {
            'name': 'Short Downtown Trip',
            'pickup_lat': 25.7617, 'pickup_lng': -80.1918,
            'dropoff_lat': 25.7700, 'dropoff_lng': -80.1850,
            'distance_km': 2.5
        }
    ]
    
    for case in test_cases:
        print(f"\n📍 {case['name']} ({case['distance_km']} km)")
        predictions = model.predict(
            case['pickup_lat'], case['pickup_lng'],
            case['dropoff_lat'], case['dropoff_lng'],
            case['distance_km']
        )
        
        for service, price in predictions.items():
            print(f"   {service}: ${price:.2f}")
    
    print(f"\n🎉 Miami-Only Model Training Complete!")
    print(f"🎯 Model Performance: R² = {performance['test_r2']:.4f}")
    print(f"📊 Trained on {len(df)} Miami rides")

if __name__ == "__main__":
    main() 