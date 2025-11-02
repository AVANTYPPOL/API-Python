"""
Booking Fee Prediction Model
============================

Predicts only the booking_fee component of Uber prices.
Used in conjunction with static pricing rules (base fare, per-mile rate, per-minute rate).
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import joblib
from dotenv import load_dotenv
import psycopg2
from datetime import datetime
import json

load_dotenv()


class BookingFeeModel:
    """
    XGBoost model to predict Uber booking fees
    """

    # Static pricing rules (constant per service type)
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

    def __init__(self):
        self.model = None
        self.label_encoders = {}
        self.feature_columns = None
        self.model_metadata = {}

    def load_training_data(self, limit=10000):
        """
        Load training data from PostgreSQL database
        Only loads recent data with booking_fee column populated
        """
        print(f"Loading training data from database (limit: {limit})...")

        conn = psycopg2.connect(
            host=os.getenv('DB_HOST', '127.0.0.1'),
            port=os.getenv('DB_PORT', '6546'),
            database=os.getenv('DB_NAME', 'appdb'),
            user=os.getenv('DB_USER', 'postgres'),
            password=os.getenv('DB_PASSWORD')
        )

        query = f"""
            SELECT
                pickup_lat, pickup_lng, dropoff_lat, dropoff_lng,
                distance_km, service_type, hour_of_day, day_of_week,
                traffic_level, weather_condition, booking_fee
            FROM ride_services
            WHERE booking_fee IS NOT NULL
            AND distance_km > 0
            AND booking_fee > 0
            ORDER BY created_at DESC
            LIMIT {limit}
        """

        df = pd.read_sql(query, conn)
        conn.close()

        print(f"Loaded {len(df):,} records with booking fee data")
        print(f"  Service types: {df['service_type'].unique()}")
        print(f"  Booking fee range: ${df['booking_fee'].min():.2f} - ${df['booking_fee'].max():.2f}")
        print(f"  Average booking fee: ${df['booking_fee'].mean():.2f}")

        return df

    def preprocess_data(self, df):
        """
        Preprocess data for model training
        """
        print("\nPreprocessing data...")

        # Calculate Haversine distance (additional feature)
        def haversine_distance(lat1, lon1, lat2, lon2):
            R = 6371  # Earth's radius in kilometers
            lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
            return 2 * R * np.arcsin(np.sqrt(a))

        df['haversine_km'] = haversine_distance(
            df['pickup_lat'], df['pickup_lng'],
            df['dropoff_lat'], df['dropoff_lng']
        )

        # Cyclical encoding for time features
        df['hour_sin'] = np.sin(2 * np.pi * df['hour_of_day'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour_of_day'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

        # Label encode categorical features
        categorical_cols = ['service_type', 'traffic_level', 'weather_condition']
        for col in categorical_cols:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                self.label_encoders[col].fit(df[col])
            df[f'{col}_encoded'] = self.label_encoders[col].transform(df[col])

        # Define feature columns
        self.feature_columns = [
            'pickup_lat', 'pickup_lng', 'dropoff_lat', 'dropoff_lng',
            'distance_km', 'haversine_km',
            'hour_of_day', 'day_of_week',
            'hour_sin', 'hour_cos', 'day_sin', 'day_cos',
            'service_type_encoded', 'traffic_level_encoded', 'weather_condition_encoded'
        ]

        X = df[self.feature_columns]
        y = df['booking_fee']

        print(f"Features: {len(self.feature_columns)} columns")
        print(f"Target: booking_fee (${y.min():.2f} - ${y.max():.2f})")

        return X, y, df

    def train(self, test_size=0.2, random_state=42):
        """
        Train XGBoost model to predict booking fees
        """
        print("\n" + "="*70)
        print("BOOKING FEE MODEL TRAINING")
        print("="*70)

        # Load data
        df = self.load_training_data()
        X, y, df_processed = self.preprocess_data(df)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        print(f"\nTraining set: {len(X_train):,} samples")
        print(f"Test set: {len(X_test):,} samples")

        # Train XGBoost model
        print("\nTraining XGBoost model...")
        self.model = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            n_jobs=-1
        )

        self.model.fit(X_train, y_train)

        # Evaluate
        y_pred_train = self.model.predict(X_train)
        y_pred_test = self.model.predict(X_test)

        train_mae = mean_absolute_error(y_train, y_pred_train)
        test_mae = mean_absolute_error(y_test, y_pred_test)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)

        print("\n" + "-"*70)
        print("MODEL PERFORMANCE")
        print("-"*70)
        print(f"Training Set:")
        print(f"  R² Score: {train_r2:.4f}")
        print(f"  RMSE: ${train_rmse:.2f}")
        print(f"  MAE: ${train_mae:.2f}")
        print(f"\nTest Set:")
        print(f"  R² Score: {test_r2:.4f}")
        print(f"  RMSE: ${test_rmse:.2f}")
        print(f"  MAE: ${test_mae:.2f}")

        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

        print("\n" + "-"*70)
        print("TOP 10 FEATURE IMPORTANCE")
        print("-"*70)
        print(feature_importance.head(10).to_string(index=False))

        # Store metadata
        self.model_metadata = {
            'model_type': 'XGBoost Booking Fee Predictor',
            'trained_date': datetime.now().isoformat(),
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'features': self.feature_columns,
            'test_r2': float(test_r2),
            'test_rmse': float(test_rmse),
            'test_mae': float(test_mae),
            'pricing_rules': self.PRICING_RULES
        }

        print("\n" + "="*70)
        print("TRAINING COMPLETE")
        print("="*70)

        return self.model_metadata

    def predict_booking_fee(self, pickup_lat, pickup_lng, dropoff_lat, dropoff_lng,
                           service_type='UBERX', hour_of_day=12, day_of_week=2,
                           traffic_level='moderate', weather_condition='clear'):
        """
        Predict booking fee for a single trip
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded")

        # Calculate haversine distance
        def haversine_distance(lat1, lon1, lat2, lon2):
            R = 6371
            lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
            return 2 * R * np.arcsin(np.sqrt(a))

        distance_km = haversine_distance(pickup_lat, pickup_lng, dropoff_lat, dropoff_lng)

        # Cyclical encoding
        hour_sin = np.sin(2 * np.pi * hour_of_day / 24)
        hour_cos = np.cos(2 * np.pi * hour_of_day / 24)
        day_sin = np.sin(2 * np.pi * day_of_week / 7)
        day_cos = np.cos(2 * np.pi * day_of_week / 7)

        # Encode categoricals
        service_type_encoded = self.label_encoders['service_type'].transform([service_type])[0]
        traffic_encoded = self.label_encoders['traffic_level'].transform([traffic_level])[0]
        weather_encoded = self.label_encoders['weather_condition'].transform([weather_condition])[0]

        # Create feature vector
        features = pd.DataFrame([[
            pickup_lat, pickup_lng, dropoff_lat, dropoff_lng,
            distance_km, distance_km,  # Use same distance for haversine_km
            hour_of_day, day_of_week,
            hour_sin, hour_cos, day_sin, day_cos,
            service_type_encoded, traffic_encoded, weather_encoded
        ]], columns=self.feature_columns)

        # Predict
        predicted_fee = float(self.model.predict(features)[0])

        # Ensure non-negative
        predicted_fee = max(0, predicted_fee)

        return predicted_fee

    def save_model(self, filepath='booking_fee_model.pkl'):
        """
        Save model and metadata
        """
        model_data = {
            'model': self.model,
            'label_encoders': self.label_encoders,
            'feature_columns': self.feature_columns,
            'metadata': self.model_metadata,
            'pricing_rules': self.PRICING_RULES
        }

        joblib.dump(model_data, filepath)
        print(f"\n[OK] Model saved to {filepath}")

        # Also save metadata as JSON
        metadata_file = filepath.replace('.pkl', '_metadata.json')
        with open(metadata_file, 'w') as f:
            json.dump(self.model_metadata, f, indent=2)
        print(f"[OK] Metadata saved to {metadata_file}")

    def load_model(self, filepath='booking_fee_model.pkl'):
        """
        Load model and metadata
        """
        model_data = joblib.load(filepath)

        self.model = model_data['model']
        self.label_encoders = model_data['label_encoders']
        self.feature_columns = model_data['feature_columns']
        self.model_metadata = model_data.get('metadata', {})

        print(f"[OK] Booking fee model loaded from {filepath}")
        print(f"  Model type: {self.model_metadata.get('model_type', 'Unknown')}")
        print(f"  Test R²: {self.model_metadata.get('test_r2', 0):.4f}")
        print(f"  Test MAE: ${self.model_metadata.get('test_mae', 0):.2f}")


if __name__ == "__main__":
    # Train the model
    model = BookingFeeModel()
    model.train()
    model.save_model('booking_fee_model.pkl')

    # Test prediction
    print("\n" + "="*70)
    print("TEST PREDICTION")
    print("="*70)
    print("\nTest Route: Miami Airport → South Beach")

    booking_fee = model.predict_booking_fee(
        pickup_lat=25.7959, pickup_lng=-80.2870,  # Airport
        dropoff_lat=25.7907, dropoff_lng=-80.1300,  # South Beach
        service_type='UBERX',
        hour_of_day=14,
        day_of_week=2,
        traffic_level='moderate',
        weather_condition='clear'
    )

    print(f"\nPredicted booking fee: ${booking_fee:.2f}")
    print("="*70)
