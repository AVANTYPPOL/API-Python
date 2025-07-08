"""
Production-Ready Uber Price Prediction Model
===========================================

Optimized for Miami market with small dataset considerations:
- Uses regularization to prevent overfitting
- Focuses on high-impact features (distance, location)
- Handles missing traffic/weather data gracefully
- Implements cross-validation for small datasets
- Production-ready with error handling

Key Insights from Data Analysis:
- Distance has 95% correlation with price
- Geographic location (longitude) is important  
- Traffic/weather data is sparse but valuable when available
- Average pricing: $1.86/km baseline

Author: AI Assistant
Date: 2024
"""

import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import LabelEncoder, StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline
import joblib
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("🚀 PRODUCTION UBER PRICE PREDICTION MODEL")
print("=" * 70)
print("🎯 Optimized for Miami Market")
print("📊 Small Dataset + Regularization")
print("🌟 Production Ready")
print("=" * 70)

class ProductionUberPriceModel:
    """
    Production-ready Uber price prediction model
    Optimized for small datasets and real-world deployment
    """
    
    def __init__(self, db_path='uber_ml_data.db'):
        self.db_path = db_path
        self.model = None
        self.fallback_model = None
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_importance = None
        self.is_trained = False
        self.baseline_price_per_km = 1.86  # From data analysis
        
        # Core features (always available)
        self.core_features = [
            'distance_km', 'pickup_lat', 'pickup_lng', 'dropoff_lat', 'dropoff_lng'
        ]
        
        # Enhanced features (may be missing)
        self.enhanced_features = [
            'hour_of_day', 'day_of_week', 'surge_multiplier', 
            'traffic_level', 'weather_condition'
        ]
        
        self.all_features = self.core_features + self.enhanced_features
        
    def load_data(self, use_all_records=True):
        """
        Load data with flexible completeness requirements
        
        Args:
            use_all_records (bool): If True, use all records; if False, only complete records
        """
        print(f"\n📥 Loading Data (use_all_records={use_all_records})...")
        
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Base query
            query = "SELECT * FROM rides WHERE uber_price_usd > 0"
            
            # Filter for complete data if requested
            if not use_all_records:
                query += " AND traffic_level IS NOT NULL AND weather_condition IS NOT NULL"
            
            df = pd.read_sql(query, conn)
            conn.close()
            
            if len(df) == 0:
                raise ValueError("No valid data found in database")
            
            print(f"✅ Loaded {len(df):,} rides")
            
            # Display completeness
            complete_core = df[self.core_features].dropna().shape[0]
            complete_all = df[self.all_features].dropna().shape[0]
            
            print(f"   Core features complete: {complete_core}/{len(df)} ({complete_core/len(df)*100:.1f}%)")
            print(f"   All features complete: {complete_all}/{len(df)} ({complete_all/len(df)*100:.1f}%)")
            
            return df
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return None
    
    def preprocess_data(self, df):
        """
        Preprocess data with robust missing value handling
        """
        print("\n🔧 Preprocessing Data...")
        
        df_processed = df.copy()
        
        # Handle missing values intelligently
        
        # Core features - use median for numerical
        for feature in ['distance_km', 'pickup_lat', 'pickup_lng', 'dropoff_lat', 'dropoff_lng']:
            if feature in df_processed.columns:
                median_val = df_processed[feature].median()
                df_processed[feature].fillna(median_val, inplace=True)
        
        # Time features - use mode or reasonable defaults
        if 'hour_of_day' in df_processed.columns:
            df_processed['hour_of_day'].fillna(19, inplace=True)  # Peak hour default
        
        if 'day_of_week' in df_processed.columns:
            df_processed['day_of_week'].fillna(0, inplace=True)  # Monday default
        
        if 'surge_multiplier' in df_processed.columns:
            df_processed['surge_multiplier'].fillna(1.0, inplace=True)  # No surge default
        
        # Categorical features - use mode or 'unknown'
        if 'traffic_level' in df_processed.columns:
            mode_traffic = df_processed['traffic_level'].mode()
            default_traffic = mode_traffic.iloc[0] if len(mode_traffic) > 0 else 'light'
            df_processed['traffic_level'].fillna(default_traffic, inplace=True)
        
        if 'weather_condition' in df_processed.columns:
            mode_weather = df_processed['weather_condition'].mode()
            default_weather = mode_weather.iloc[0] if len(mode_weather) > 0 else 'clear'
            df_processed['weather_condition'].fillna(default_weather, inplace=True)
        
        # Extract features and target
        X = df_processed[self.all_features].copy()
        y = df_processed['uber_price_usd'].copy()
        
        # Encode categorical features
        X = self._encode_categorical_features(X)
        
        # Engineer features
        X = self._engineer_features(X)
        
        print(f"✅ Preprocessed {len(X)} samples with {X.shape[1]} features")
        
        return X, y
    
    def _encode_categorical_features(self, X):
        """Encode categorical features with robust error handling"""
        
        X_encoded = X.copy()
        categorical_features = ['traffic_level', 'weather_condition']
        
        for feature in categorical_features:
            if feature in X_encoded.columns:
                # Initialize encoder if not exists
                if feature not in self.label_encoders:
                    self.label_encoders[feature] = LabelEncoder()
                    X_encoded[feature] = self.label_encoders[feature].fit_transform(X_encoded[feature].astype(str))
                else:
                    # Handle unseen categories
                    unique_vals = X_encoded[feature].astype(str).unique()
                    known_classes = set(self.label_encoders[feature].classes_)
                    
                    for val in unique_vals:
                        if val not in known_classes:
                            # Add new class to encoder
                            self.label_encoders[feature].classes_ = np.append(
                                self.label_encoders[feature].classes_, val
                            )
                    
                    X_encoded[feature] = self.label_encoders[feature].transform(X_encoded[feature].astype(str))
        
        return X_encoded
    
    def _engineer_features(self, X):
        """Create advanced features optimized for small datasets"""
        
        X_engineered = X.copy()
        
        # Distance-based features (most important based on analysis)
        if 'distance_km' in X_engineered.columns:
            # Non-linear distance relationships
            X_engineered['distance_km_squared'] = X_engineered['distance_km'] ** 2
            X_engineered['distance_km_log'] = np.log1p(X_engineered['distance_km'])
            X_engineered['distance_km_sqrt'] = np.sqrt(X_engineered['distance_km'])
            
            # Distance categories for better generalization
            X_engineered['distance_category'] = pd.cut(
                X_engineered['distance_km'], 
                bins=[0, 5, 15, 30, 50, 100], 
                labels=[0, 1, 2, 3, 4], 
                include_lowest=True
            ).astype(float)
        
        # Location-based features (important based on analysis)
        if all(col in X_engineered.columns for col in ['pickup_lat', 'pickup_lng', 'dropoff_lat', 'dropoff_lng']):
            
            # Miami-specific location zones
            downtown_lat, downtown_lng = 25.7617, -80.1918
            airport_lat, airport_lng = 25.7959, -80.2870
            beach_lat, beach_lng = 25.7907, -80.1300
            
            # Distance to key locations
            X_engineered['pickup_to_downtown'] = np.sqrt(
                (X_engineered['pickup_lat'] - downtown_lat) ** 2 +
                (X_engineered['pickup_lng'] - downtown_lng) ** 2
            )
            
            X_engineered['dropoff_to_downtown'] = np.sqrt(
                (X_engineered['dropoff_lat'] - downtown_lat) ** 2 +
                (X_engineered['dropoff_lng'] - downtown_lng) ** 2
            )
            
            X_engineered['pickup_to_airport'] = np.sqrt(
                (X_engineered['pickup_lat'] - airport_lat) ** 2 +
                (X_engineered['pickup_lng'] - airport_lng) ** 2
            )
            
            X_engineered['dropoff_to_airport'] = np.sqrt(
                (X_engineered['dropoff_lat'] - airport_lat) ** 2 +
                (X_engineered['dropoff_lng'] - airport_lng) ** 2
            )
            
            # Airport trip indicator (high-value trips)
            airport_threshold = 0.02  # degrees
            X_engineered['is_airport_pickup'] = (X_engineered['pickup_to_airport'] < airport_threshold).astype(int)
            X_engineered['is_airport_dropoff'] = (X_engineered['dropoff_to_airport'] < airport_threshold).astype(int)
            X_engineered['is_airport_trip'] = (X_engineered['is_airport_pickup'] | X_engineered['is_airport_dropoff']).astype(int)
        
        # Time-based features
        if 'hour_of_day' in X_engineered.columns:
            # Rush hour indicators
            X_engineered['is_morning_rush'] = X_engineered['hour_of_day'].apply(
                lambda x: 1 if x in [7, 8, 9] else 0
            )
            X_engineered['is_evening_rush'] = X_engineered['hour_of_day'].apply(
                lambda x: 1 if x in [17, 18, 19] else 0
            )
            X_engineered['is_peak_hour'] = (X_engineered['is_morning_rush'] | X_engineered['is_evening_rush']).astype(int)
            
            # Late night premium
            X_engineered['is_late_night'] = X_engineered['hour_of_day'].apply(
                lambda x: 1 if x in [22, 23, 0, 1, 2, 3, 4, 5] else 0
            )
        
        if 'day_of_week' in X_engineered.columns:
            # Weekend indicator
            X_engineered['is_weekend'] = X_engineered['day_of_week'].apply(
                lambda x: 1 if x in [5, 6] else 0
            )
        
        return X_engineered
    
    def train_model(self, X, y):
        """
        Train production model with cross-validation and regularization
        """
        print("\n🤖 Training Production Model...")
        
        # Handle small dataset with cross-validation
        if len(X) < 100:
            print("⚠️  Small dataset detected - using cross-validation")
            return self._train_with_cross_validation(X, y)
        else:
            return self._train_with_holdout(X, y)
    
    def _train_with_cross_validation(self, X, y):
        """Train model using cross-validation for small datasets"""
        
        # Primary model: Regularized Random Forest
        self.model = RandomForestRegressor(
            n_estimators=100,  # Reduced for small datasets
            max_depth=8,       # Prevent overfitting
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        # Fallback model: Ridge regression (linear, robust)
        self.fallback_model = Ridge(alpha=1.0, random_state=42)
        
        # Cross-validation scores
        cv_scores_rf = cross_val_score(self.model, X, y, cv=5, scoring='r2')
        cv_scores_ridge = cross_val_score(self.fallback_model, X, y, cv=5, scoring='r2')
        
        print(f"📊 Cross-Validation Results:")
        print(f"   Random Forest R²: {cv_scores_rf.mean():.4f} ± {cv_scores_rf.std():.4f}")
        print(f"   Ridge Regression R²: {cv_scores_ridge.mean():.4f} ± {cv_scores_ridge.std():.4f}")
        
        # Train final models on full dataset
        self.model.fit(X, y)
        self.fallback_model.fit(X, y)
        
        # Calculate feature importance
        self.feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        self.is_trained = True
        
        # Final evaluation
        y_pred_rf = self.model.predict(X)
        y_pred_ridge = self.fallback_model.predict(X)
        
        rf_r2 = r2_score(y, y_pred_rf)
        ridge_r2 = r2_score(y, y_pred_ridge)
        
        print(f"\n✅ Training Complete:")
        print(f"   Random Forest R²: {rf_r2:.4f}")
        print(f"   Ridge Regression R²: {ridge_r2:.4f}")
        print(f"   Using Random Forest as primary model")
        
        return {
            'cv_scores_rf': cv_scores_rf,
            'cv_scores_ridge': cv_scores_ridge,
            'final_r2_rf': rf_r2,
            'final_r2_ridge': ridge_r2
        }
    
    def predict_price(self, distance_km, pickup_lat, pickup_lng, dropoff_lat, dropoff_lng,
                     hour_of_day=19, day_of_week=0, surge_multiplier=1.0, 
                     traffic_level='light', weather_condition='clear'):
        """
        Predict price with fallback mechanisms
        """
        if not self.is_trained:
            return self._fallback_prediction(distance_km, surge_multiplier)
        
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
            
            # Preprocess
            input_processed = self._encode_categorical_features(input_data)
            input_processed = self._engineer_features(input_processed)
            
            # Make prediction
            prediction = self.model.predict(input_processed)[0]
            
            # Sanity check
            if prediction < 0 or prediction > 500:  # Unrealistic price
                print("⚠️  Unrealistic prediction, using fallback")
                return self._fallback_prediction(distance_km, surge_multiplier)
            
            return max(prediction, 2.50)  # Minimum fare
            
        except Exception as e:
            print(f"⚠️  Prediction error: {e}, using fallback")
            return self._fallback_prediction(distance_km, surge_multiplier)
    
    def _fallback_prediction(self, distance_km, surge_multiplier=1.0):
        """
        Fallback prediction using baseline pricing
        """
        base_fare = 2.50
        per_km_rate = self.baseline_price_per_km
        
        # Simple distance-based pricing with surge
        price = (base_fare + (distance_km * per_km_rate)) * surge_multiplier
        
        return max(price, 2.50)
    
    def display_feature_importance(self, top_n=10):
        """Display feature importance"""
        if self.feature_importance is None:
            print("❌ Model not trained yet!")
            return
        
        print(f"\n🏆 Top {top_n} Most Important Features:")
        print("=" * 50)
        
        for idx, (_, row) in enumerate(self.feature_importance.head(top_n).iterrows(), 1):
            importance_pct = row['importance'] * 100
            print(f"{idx:2d}. {row['feature']:<25} {importance_pct:6.2f}%")
    
    def save_model(self, filepath='production_uber_model.pkl'):
        """Save production model"""
        if not self.is_trained:
            print("❌ Model not trained yet!")
            return
        
        model_data = {
            'model': self.model,
            'fallback_model': self.fallback_model,
            'label_encoders': self.label_encoders,
            'feature_importance': self.feature_importance,
            'baseline_price_per_km': self.baseline_price_per_km,
            'features': self.all_features
        }
        
        joblib.dump(model_data, filepath)
        print(f"✅ Production model saved to {filepath}")
    
    def load_model(self, filepath='production_uber_model.pkl'):
        """Load production model"""
        try:
            model_data = joblib.load(filepath)
            self.model = model_data['model']
            self.fallback_model = model_data['fallback_model']
            self.label_encoders = model_data['label_encoders']
            self.feature_importance = model_data['feature_importance']
            self.baseline_price_per_km = model_data['baseline_price_per_km']
            self.all_features = model_data['features']
            self.is_trained = True
            print(f"✅ Production model loaded from {filepath}")
        except Exception as e:
            print(f"❌ Error loading model: {e}")

def main():
    """Main function for production model"""
    print("🚀 Starting Production Uber Price Model Training")
    
    # Initialize model
    model = ProductionUberPriceModel()
    
    try:
        # Load data (use all available records)
        df = model.load_data(use_all_records=True)
        
        if df is None or len(df) == 0:
            print("❌ No data available")
            return
        
        # Preprocess data
        X, y = model.preprocess_data(df)
        
        # Train model
        results = model.train_model(X, y)
        
        # Display feature importance
        model.display_feature_importance()
        
        # Save model
        model.save_model()
        
        # Test predictions
        print("\n🎯 Testing Production Model:")
        print("-" * 40)
        
        # Test cases
        test_cases = [
            # (distance, pickup_lat, pickup_lng, dropoff_lat, dropoff_lng, description)
            (5.2, 25.7617, -80.1918, 25.7907, -80.1300, "Downtown to Beach"),
            (15.0, 25.7959, -80.2870, 25.7617, -80.1918, "Airport to Downtown"),
            (2.0, 25.7617, -80.1918, 25.7700, -80.1850, "Short Downtown Trip"),
            (35.0, 25.2783, -80.8107, 25.9571, -80.1309, "Long Distance Trip")
        ]
        
        for distance, pickup_lat, pickup_lng, dropoff_lat, dropoff_lng, description in test_cases:
            price = model.predict_price(
                distance_km=distance,
                pickup_lat=pickup_lat,
                pickup_lng=pickup_lng,
                dropoff_lat=dropoff_lat,
                dropoff_lng=dropoff_lng,
                hour_of_day=19,  # Peak hour
                surge_multiplier=1.0,
                traffic_level='light',
                weather_condition='clear'
            )
            
            price_per_km = price / distance
            print(f"{description:20s}: ${price:6.2f} (${price_per_km:.2f}/km)")
        
        print(f"\n✅ Production model ready for deployment!")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main() 