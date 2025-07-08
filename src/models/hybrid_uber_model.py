"""
Hybrid Uber Price Prediction Model
==================================

Transfer Learning Approach:
1. Pre-train on large NYC dataset (universal patterns)
2. Fine-tune on Miami dataset (local market adaptation)
3. Combine predictions for optimal accuracy

This approach leverages:
- NYC data: Large dataset for robust distance/time patterns
- Miami data: Local market specifics and pricing adjustments
- Transfer learning: Best of both worlds

Author: AI Assistant
Date: 2024
"""

import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime, timedelta
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("🔄 HYBRID UBER PRICE PREDICTION MODEL")
print("=" * 70)
print("🌆 Step 1: Pre-train on NYC Data (Universal Patterns)")
print("🏖️  Step 2: Fine-tune on Miami Data (Local Adaptation)")
print("🎯 Step 3: Hybrid Predictions (Best of Both)")
print("=" * 70)

class HybridUberPriceModel:
    """
    Hybrid model using transfer learning approach
    """
    
    def __init__(self, miami_db_path='uber_ml_data.db'):
        self.miami_db_path = miami_db_path
        
        # Models
        self.nyc_base_model = None
        self.miami_fine_tuned_model = None
        self.hybrid_model = None
        
        # Encoders and scalers
        self.label_encoders = {}
        self.scaler = StandardScaler()
        
        # Training status
        self.nyc_trained = False
        self.miami_trained = False
        self.hybrid_trained = False
        
        # Feature importance
        self.feature_importance = None
        
        # Baseline pricing
        self.miami_baseline_price_per_km = 1.86
        
        print("🏗️  Hybrid model initialized")
    
    def generate_nyc_dataset(self, n_samples=50000):
        """
        Generate realistic NYC taxi dataset for pre-training
        """
        print(f"\n🌆 Generating NYC Dataset ({n_samples:,} samples)...")
        
        np.random.seed(42)
        
        # NYC coordinate bounds
        NYC_LAT_MIN, NYC_LAT_MAX = 40.63, 40.85
        NYC_LON_MIN, NYC_LON_MAX = -74.05, -73.75
        
        # Generate coordinates
        pickup_lat = np.random.uniform(NYC_LAT_MIN, NYC_LAT_MAX, n_samples)
        pickup_lng = np.random.uniform(NYC_LON_MIN, NYC_LON_MAX, n_samples)
        dropoff_lat = np.random.uniform(NYC_LAT_MIN, NYC_LAT_MAX, n_samples)
        dropoff_lng = np.random.uniform(NYC_LON_MIN, NYC_LON_MAX, n_samples)
        
        # Calculate distances
        distance_km = self._haversine_distance(pickup_lat, pickup_lng, dropoff_lat, dropoff_lng)
        
        # Generate time features
        dates = pd.date_range('2023-01-01', '2023-12-31', freq='H')
        random_dates = np.random.choice(dates, n_samples)
        random_dates = pd.to_datetime(random_dates)
        
        hour_of_day = np.array([d.hour for d in random_dates])
        day_of_week = np.array([d.weekday() for d in random_dates])
        
        # Generate surge multipliers
        surge_multiplier = np.ones(n_samples)
        
        # Rush hour surge
        rush_mask = np.isin(hour_of_day, [7, 8, 17, 18, 19])
        surge_multiplier[rush_mask] *= np.random.uniform(1.2, 2.0, rush_mask.sum())
        
        # Weekend night surge
        weekend_night_mask = np.isin(day_of_week, [4, 5]) & np.isin(hour_of_day, [22, 23, 0, 1, 2])
        surge_multiplier[weekend_night_mask] *= np.random.uniform(1.5, 3.0, weekend_night_mask.sum())
        
        # Generate traffic and weather
        traffic_levels = np.random.choice(['light', 'moderate', 'heavy'], n_samples, p=[0.5, 0.3, 0.2])
        weather_conditions = np.random.choice(['clear', 'clouds', 'rain'], n_samples, p=[0.6, 0.3, 0.1])
        
        # Generate realistic NYC fares
        base_fare = 2.50
        per_km_rate = 1.55  # NYC rate (converted from per-mile)
        per_minute_rate = 0.50
        
        # Estimate trip time based on distance and traffic
        avg_speeds = {'light': 25, 'moderate': 15, 'heavy': 10}  # km/h
        trip_time_hours = np.array([distance_km[i] / avg_speeds[traffic_levels[i]] for i in range(n_samples)])
        trip_time_minutes = trip_time_hours * 60
        
        # Calculate fares
        fares = (base_fare + 
                (distance_km * per_km_rate) + 
                (trip_time_minutes * per_minute_rate)) * surge_multiplier
        
        # Add weather premium
        weather_multiplier = np.where(weather_conditions == 'rain', 1.2, 1.0)
        fares *= weather_multiplier
        
        # Add noise and clean
        fares += np.random.normal(0, 1, n_samples)
        fares = np.clip(fares, 2.50, 200.0)
        
        # Create DataFrame
        nyc_df = pd.DataFrame({
            'distance_km': distance_km,
            'pickup_lat': pickup_lat,
            'pickup_lng': pickup_lng,
            'dropoff_lat': dropoff_lat,
            'dropoff_lng': dropoff_lng,
            'hour_of_day': hour_of_day,
            'day_of_week': day_of_week,
            'surge_multiplier': surge_multiplier,
            'traffic_level': traffic_levels,
            'weather_condition': weather_conditions,
            'fare_amount': fares
        })
        
        # Filter realistic trips
        nyc_df = nyc_df[
            (nyc_df['distance_km'] >= 0.5) & 
            (nyc_df['distance_km'] <= 50) &
            (nyc_df['fare_amount'] >= 2.50) &
            (nyc_df['fare_amount'] <= 150)
        ].reset_index(drop=True)
        
        print(f"✅ Generated {len(nyc_df):,} NYC rides")
        print(f"   Average fare: ${nyc_df['fare_amount'].mean():.2f}")
        print(f"   Average distance: {nyc_df['distance_km'].mean():.2f} km")
        print(f"   Price per km: ${(nyc_df['fare_amount'] / nyc_df['distance_km']).mean():.2f}")
        
        return nyc_df
    
    def load_miami_dataset(self):
        """
        Load Miami dataset from database
        """
        print(f"\n🏖️  Loading Miami Dataset...")
        
        try:
            conn = sqlite3.connect(self.miami_db_path)
            query = "SELECT * FROM rides WHERE uber_price_usd > 0"
            miami_df = pd.read_sql(query, conn)
            conn.close()
            
            if len(miami_df) == 0:
                raise ValueError("No Miami data found")
            
            # Standardize column names
            miami_df = miami_df.rename(columns={'uber_price_usd': 'fare_amount'})
            
            # Drop non-feature columns
            columns_to_drop = ['id', 'created_at']
            for col in columns_to_drop:
                if col in miami_df.columns:
                    miami_df = miami_df.drop(columns=[col])
            
            # Fill missing values
            miami_df['traffic_level'].fillna('light', inplace=True)
            miami_df['weather_condition'].fillna('clear', inplace=True)
            
            # Ensure all required columns exist with defaults
            required_columns = {
                'distance_km': 0.0,
                'pickup_lat': 25.7617,
                'pickup_lng': -80.1918,
                'dropoff_lat': 25.7617,
                'dropoff_lng': -80.1918,
                'hour_of_day': 12,
                'day_of_week': 0,
                'surge_multiplier': 1.0,
                'traffic_level': 'light',
                'weather_condition': 'clear'
            }
            
            for col, default_val in required_columns.items():
                if col not in miami_df.columns:
                    miami_df[col] = default_val
            
            print(f"✅ Loaded {len(miami_df):,} Miami rides")
            print(f"   Average fare: ${miami_df['fare_amount'].mean():.2f}")
            print(f"   Average distance: {miami_df['distance_km'].mean():.2f} km")
            print(f"   Price per km: ${(miami_df['fare_amount'] / miami_df['distance_km']).mean():.2f}")
            
            return miami_df
            
        except Exception as e:
            print(f"❌ Error loading Miami data: {e}")
            return None
    
    def _haversine_distance(self, lat1, lng1, lat2, lng2):
        """Calculate distance between coordinates"""
        lat1, lng1, lat2, lng2 = map(np.radians, [lat1, lng1, lat2, lng2])
        dlat, dlng = lat2 - lat1, lng2 - lng1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlng/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        return 6371 * c  # Earth radius in km
    
    def preprocess_features(self, df, is_miami=False):
        """
        Preprocess features for training
        """
        df_processed = df.copy()
        
        # Core features
        features = ['distance_km', 'pickup_lat', 'pickup_lng', 'dropoff_lat', 'dropoff_lng',
                   'hour_of_day', 'day_of_week', 'surge_multiplier']
        
        # Encode categorical features
        categorical_features = ['traffic_level', 'weather_condition']
        
        for feature in categorical_features:
            if feature in df_processed.columns:
                if feature not in self.label_encoders:
                    self.label_encoders[feature] = LabelEncoder()
                    df_processed[feature] = self.label_encoders[feature].fit_transform(df_processed[feature].astype(str))
                else:
                    # Handle new categories
                    unique_vals = df_processed[feature].astype(str).unique()
                    known_classes = set(self.label_encoders[feature].classes_)
                    
                    for val in unique_vals:
                        if val not in known_classes:
                            self.label_encoders[feature].classes_ = np.append(
                                self.label_encoders[feature].classes_, val
                            )
                    
                    df_processed[feature] = self.label_encoders[feature].transform(df_processed[feature].astype(str))
                
                features.append(feature)
        
        # Engineer features
        df_processed = self._engineer_features(df_processed, is_miami)
        
        # Get all engineered features
        engineered_features = [col for col in df_processed.columns if col not in ['fare_amount']]
        
        X = df_processed[engineered_features]
        y = df_processed['fare_amount']
        
        return X, y
    
    def _engineer_features(self, df, is_miami=False):
        """
        Engineer features for better prediction
        """
        df_eng = df.copy()
        
        # Distance features
        if 'distance_km' in df_eng.columns:
            df_eng['distance_km_squared'] = df_eng['distance_km'] ** 2
            df_eng['distance_km_log'] = np.log1p(df_eng['distance_km'])
            df_eng['distance_km_sqrt'] = np.sqrt(df_eng['distance_km'])
        
        # Time features
        if 'hour_of_day' in df_eng.columns:
            df_eng['is_rush_hour'] = df_eng['hour_of_day'].apply(
                lambda x: 1 if x in [7, 8, 17, 18, 19] else 0
            )
            df_eng['is_late_night'] = df_eng['hour_of_day'].apply(
                lambda x: 1 if x in [22, 23, 0, 1, 2, 3, 4, 5] else 0
            )
        
        if 'day_of_week' in df_eng.columns:
            df_eng['is_weekend'] = df_eng['day_of_week'].apply(
                lambda x: 1 if x in [5, 6] else 0
            )
        
        # Miami-specific features
        if is_miami and all(col in df_eng.columns for col in ['pickup_lat', 'pickup_lng', 'dropoff_lat', 'dropoff_lng']):
            # Key Miami locations
            airport_lat, airport_lng = 25.7959, -80.2870
            beach_lat, beach_lng = 25.7907, -80.1300
            downtown_lat, downtown_lng = 25.7617, -80.1918
            
            # Distance to key locations
            df_eng['pickup_to_airport'] = np.sqrt(
                (df_eng['pickup_lat'] - airport_lat) ** 2 +
                (df_eng['pickup_lng'] - airport_lng) ** 2
            )
            
            df_eng['dropoff_to_airport'] = np.sqrt(
                (df_eng['dropoff_lat'] - airport_lat) ** 2 +
                (df_eng['dropoff_lng'] - airport_lng) ** 2
            )
            
            # Airport trip indicators
            airport_threshold = 0.02
            df_eng['is_airport_pickup'] = (df_eng['pickup_to_airport'] < airport_threshold).astype(int)
            df_eng['is_airport_dropoff'] = (df_eng['dropoff_to_airport'] < airport_threshold).astype(int)
            df_eng['is_airport_trip'] = (df_eng['is_airport_pickup'] | df_eng['is_airport_dropoff']).astype(int)
        
        return df_eng
    
    def train_nyc_base_model(self, nyc_df):
        """
        Step 1: Train base model on NYC data
        """
        print(f"\n🌆 Step 1: Training NYC Base Model...")
        
        X_nyc, y_nyc = self.preprocess_features(nyc_df, is_miami=False)
        
        # Train Random Forest on NYC data
        self.nyc_base_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1
        )
        
        # Cross-validation
        cv_scores = cross_val_score(self.nyc_base_model, X_nyc, y_nyc, cv=5, scoring='r2')
        print(f"   NYC Model CV R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        # Train final model
        self.nyc_base_model.fit(X_nyc, y_nyc)
        
        # Evaluate
        y_pred = self.nyc_base_model.predict(X_nyc)
        r2 = r2_score(y_nyc, y_pred)
        rmse = np.sqrt(mean_squared_error(y_nyc, y_pred))
        
        print(f"   NYC Model Training R²: {r2:.4f}")
        print(f"   NYC Model RMSE: ${rmse:.2f}")
        
        self.nyc_trained = True
        return X_nyc.columns.tolist()
    
    def train_miami_fine_tuned_model(self, miami_df, nyc_feature_columns):
        """
        Step 2: Fine-tune on Miami data with NYC predictions as features
        """
        print(f"\n🏖️  Step 2: Fine-tuning on Miami Data...")
        
        X_miami, y_miami = self.preprocess_features(miami_df, is_miami=True)
        
        # Align features with NYC model
        common_features = [col for col in nyc_feature_columns if col in X_miami.columns]
        missing_features = [col for col in nyc_feature_columns if col not in X_miami.columns]
        
        if missing_features:
            print(f"   ⚠️  Missing features in Miami data: {missing_features}")
            # Add missing features with default values
            for feature in missing_features:
                X_miami[feature] = 0
        
        # Reorder columns to match NYC model
        X_miami_aligned = X_miami[nyc_feature_columns]
        
        # Get NYC model predictions as features
        nyc_predictions = self.nyc_base_model.predict(X_miami_aligned)
        
        # Create enhanced Miami features
        X_miami_enhanced = X_miami.copy()
        X_miami_enhanced['nyc_prediction'] = nyc_predictions
        X_miami_enhanced['miami_nyc_ratio'] = y_miami / (nyc_predictions + 0.01)  # Avoid division by zero
        
        # Train Miami-specific model
        self.miami_fine_tuned_model = RandomForestRegressor(
            n_estimators=50,  # Smaller for small dataset
            max_depth=8,      # Prevent overfitting
            min_samples_split=3,
            min_samples_leaf=2,
            random_state=42
        )
        
        # Cross-validation
        cv_scores = cross_val_score(self.miami_fine_tuned_model, X_miami_enhanced, y_miami, cv=3, scoring='r2')
        print(f"   Miami Model CV R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        # Train final model
        self.miami_fine_tuned_model.fit(X_miami_enhanced, y_miami)
        
        # Evaluate
        y_pred = self.miami_fine_tuned_model.predict(X_miami_enhanced)
        r2 = r2_score(y_miami, y_pred)
        rmse = np.sqrt(mean_squared_error(y_miami, y_pred))
        
        print(f"   Miami Model Training R²: {r2:.4f}")
        print(f"   Miami Model RMSE: ${rmse:.2f}")
        
        self.miami_trained = True
        return X_miami_enhanced.columns.tolist()
    
    def create_hybrid_model(self, miami_df, nyc_feature_columns, miami_feature_columns):
        """
        Step 3: Create hybrid ensemble model
        """
        print(f"\n🎯 Step 3: Creating Hybrid Ensemble Model...")
        
        X_miami, y_miami = self.preprocess_features(miami_df, is_miami=True)
        
        # Get predictions from both models
        X_miami_aligned = X_miami.reindex(columns=nyc_feature_columns, fill_value=0)
        nyc_predictions = self.nyc_base_model.predict(X_miami_aligned)
        
        X_miami_enhanced = X_miami.copy()
        X_miami_enhanced['nyc_prediction'] = nyc_predictions
        X_miami_enhanced['miami_nyc_ratio'] = y_miami / (nyc_predictions + 0.01)
        X_miami_enhanced = X_miami_enhanced.reindex(columns=miami_feature_columns, fill_value=0)
        
        miami_predictions = self.miami_fine_tuned_model.predict(X_miami_enhanced)
        
        # Create ensemble features
        ensemble_features = pd.DataFrame({
            'nyc_prediction': nyc_predictions,
            'miami_prediction': miami_predictions,
            'prediction_diff': miami_predictions - nyc_predictions,
            'prediction_ratio': miami_predictions / (nyc_predictions + 0.01),
            'distance_km': X_miami['distance_km'],
            'is_airport_trip': X_miami.get('is_airport_trip', 0)
        })
        
        # Train ensemble model
        self.hybrid_model = RandomForestRegressor(
            n_estimators=30,
            max_depth=5,
            random_state=42
        )
        
        # Cross-validation
        cv_scores = cross_val_score(self.hybrid_model, ensemble_features, y_miami, cv=3, scoring='r2')
        print(f"   Hybrid Model CV R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        # Train final model
        self.hybrid_model.fit(ensemble_features, y_miami)
        
        # Evaluate
        y_pred = self.hybrid_model.predict(ensemble_features)
        r2 = r2_score(y_miami, y_pred)
        rmse = np.sqrt(mean_squared_error(y_miami, y_pred))
        
        print(f"   Hybrid Model Training R²: {r2:.4f}")
        print(f"   Hybrid Model RMSE: ${rmse:.2f}")
        
        # Feature importance
        self.feature_importance = pd.DataFrame({
            'feature': ensemble_features.columns,
            'importance': self.hybrid_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\n🏆 Hybrid Model Feature Importance:")
        for _, row in self.feature_importance.iterrows():
            print(f"   {row['feature']:<20}: {row['importance']*100:5.1f}%")
        
        self.hybrid_trained = True
    
    def train_full_pipeline(self):
        """
        Train the complete hybrid model pipeline
        """
        print(f"🚀 Starting Hybrid Model Training Pipeline...")
        
        # Step 1: Generate NYC data and train base model
        nyc_df = self.generate_nyc_dataset(n_samples=50000)
        nyc_features = self.train_nyc_base_model(nyc_df)
        
        # Step 2: Load Miami data and fine-tune
        miami_df = self.load_miami_dataset()
        if miami_df is None:
            print("❌ Cannot proceed without Miami data")
            return False
        
        miami_features = self.train_miami_fine_tuned_model(miami_df, nyc_features)
        
        # Step 3: Create hybrid ensemble
        self.create_hybrid_model(miami_df, nyc_features, miami_features)
        
        print(f"\n✅ Hybrid Model Training Complete!")
        return True
    
    def predict_price(self, distance_km, pickup_lat, pickup_lng, dropoff_lat, dropoff_lng,
                     hour_of_day=19, day_of_week=0, surge_multiplier=1.0,
                     traffic_level='light', weather_condition='clear'):
        """
        Make hybrid prediction
        """
        if not self.hybrid_trained:
            print("❌ Hybrid model not trained yet!")
            return self._fallback_prediction(distance_km, surge_multiplier)
        
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
                'surge_multiplier': [surge_multiplier],
                'traffic_level': [traffic_level],
                'weather_condition': [weather_condition]
            })
            
            # Preprocess features (without target variable)
            df_processed = input_data.copy()
            
            # Encode categorical features
            categorical_features = ['traffic_level', 'weather_condition']
            
            for feature in categorical_features:
                if feature in df_processed.columns and feature in self.label_encoders:
                    # Handle new categories gracefully
                    try:
                        df_processed[feature] = self.label_encoders[feature].transform(df_processed[feature].astype(str))
                    except ValueError:
                        # If category not seen during training, use most common category
                        df_processed[feature] = 0  # Default to first category
            
            # Engineer features
            df_processed = self._engineer_features(df_processed, is_miami=True)
            
            # Get all feature columns (exclude target)
            feature_columns = [col for col in df_processed.columns if col != 'fare_amount']
            X_processed = df_processed[feature_columns]
            
            # NYC prediction (align features)
            nyc_feature_names = self.nyc_base_model.feature_names_in_
            X_nyc_aligned = X_processed.reindex(columns=nyc_feature_names, fill_value=0)
            nyc_pred = self.nyc_base_model.predict(X_nyc_aligned)[0]
            
            # Miami prediction
            X_enhanced = X_processed.copy()
            X_enhanced['nyc_prediction'] = nyc_pred
            X_enhanced['miami_nyc_ratio'] = 1.0  # Default ratio
            
            miami_feature_names = self.miami_fine_tuned_model.feature_names_in_
            X_miami_aligned = X_enhanced.reindex(columns=miami_feature_names, fill_value=0)
            miami_pred = self.miami_fine_tuned_model.predict(X_miami_aligned)[0]
            
            # Hybrid prediction
            ensemble_input = pd.DataFrame({
                'nyc_prediction': [nyc_pred],
                'miami_prediction': [miami_pred],
                'prediction_diff': [miami_pred - nyc_pred],
                'prediction_ratio': [miami_pred / (nyc_pred + 0.01)],
                'distance_km': [distance_km],
                'is_airport_trip': [X_processed.get('is_airport_trip', [0])[0]]
            })
            
            hybrid_pred = self.hybrid_model.predict(ensemble_input)[0]
            
            return max(hybrid_pred, 2.50)  # Minimum fare
            
        except Exception as e:
            print(f"⚠️  Prediction error: {e}, using fallback")
            return self._fallback_prediction(distance_km, surge_multiplier)
    
    def _fallback_prediction(self, distance_km, surge_multiplier=1.0):
        """Fallback prediction"""
        base_fare = 2.50
        return (base_fare + (distance_km * self.miami_baseline_price_per_km)) * surge_multiplier
    
    def save_model(self, filepath='hybrid_uber_model.pkl'):
        """Save hybrid model"""
        if not self.hybrid_trained:
            print("❌ Hybrid model not trained yet!")
            return
        
        model_data = {
            'nyc_base_model': self.nyc_base_model,
            'miami_fine_tuned_model': self.miami_fine_tuned_model,
            'hybrid_model': self.hybrid_model,
            'label_encoders': self.label_encoders,
            'feature_importance': self.feature_importance,
            'miami_baseline_price_per_km': self.miami_baseline_price_per_km
        }
        
        joblib.dump(model_data, filepath)
        print(f"✅ Hybrid model saved to {filepath}")
    
    def load_model(self, filepath='hybrid_uber_model.pkl'):
        """Load hybrid model"""
        try:
            model_data = joblib.load(filepath)
            self.nyc_base_model = model_data['nyc_base_model']
            self.miami_fine_tuned_model = model_data['miami_fine_tuned_model']
            self.hybrid_model = model_data['hybrid_model']
            self.label_encoders = model_data['label_encoders']
            self.feature_importance = model_data['feature_importance']
            self.miami_baseline_price_per_km = model_data['miami_baseline_price_per_km']
            
            self.nyc_trained = True
            self.miami_trained = True
            self.hybrid_trained = True
            
            print(f"✅ Hybrid model loaded from {filepath}")
        except Exception as e:
            print(f"❌ Error loading model: {e}")

def main():
    """Main function to train and test hybrid model"""
    
    # Initialize hybrid model
    hybrid_model = HybridUberPriceModel()
    
    # Train the full pipeline
    success = hybrid_model.train_full_pipeline()
    
    if not success:
        print("❌ Training failed")
        return
    
    # Save model
    hybrid_model.save_model()
    
    # Test predictions
    print(f"\n🧪 Testing Hybrid Model Predictions:")
    print("=" * 50)
    
    test_cases = [
        (5.2, 25.7617, -80.1918, 25.7907, -80.1300, "Downtown to Beach"),
        (19.8, 25.7959, -80.2870, 25.7617, -80.1918, "Airport to Downtown"),
        (2.0, 25.7617, -80.1918, 25.7700, -80.1850, "Short Downtown Trip"),
        (35.0, 25.2783, -80.8107, 25.9571, -80.1309, "Long Distance Trip")
    ]
    
    for distance, pickup_lat, pickup_lng, dropoff_lat, dropoff_lng, description in test_cases:
        price = hybrid_model.predict_price(
            distance_km=distance,
            pickup_lat=pickup_lat,
            pickup_lng=pickup_lng,
            dropoff_lat=dropoff_lat,
            dropoff_lng=dropoff_lng,
            hour_of_day=19,
            surge_multiplier=1.0,
            traffic_level='light',
            weather_condition='clear'
        )
        
        price_per_km = price / distance
        print(f"{description:20s}: ${price:6.2f} (${price_per_km:.2f}/km)")
    
    print(f"\n🎉 Hybrid Model Training and Testing Complete!")

if __name__ == "__main__":
    main() 