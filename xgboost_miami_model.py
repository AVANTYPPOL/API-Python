"""
XGBoost Miami Multi-Service Model
=================================

A pure XGBoost model trained on Miami ride_services data
supporting multiple Uber service types (UberX, UberXL, Premier, Premier SUV)

Key Features:
- Uses only local Miami data from ride_services table
- XGBoost for better performance and accuracy
- Multi-service pricing prediction
- Feature engineering optimized for Miami market

Author: AI Assistant
Date: 2025
"""

import pandas as pd
import numpy as np
import sqlite3
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
import joblib
import warnings
warnings.filterwarnings('ignore')

# Add numpy._core compatibility if needed
if not hasattr(np, '_core') and hasattr(np, 'core'):
    # Create a compatibility shim for numpy._core
    class NumpyCoreCompat:
        multiarray = np.core.multiarray
        umath = np.core.umath
        _internal = np.core._internal if hasattr(np.core, '_internal') else None
        numeric = np.core.numeric if hasattr(np.core, 'numeric') else None
        fromnumeric = np.core.fromnumeric if hasattr(np.core, 'fromnumeric') else None
    np._core = NumpyCoreCompat()

class XGBoostMiamiModel:
    """
    XGBoost model for Miami multi-service Uber pricing
    """
    
    def __init__(self, db_path='uber_ml_data.db'):
        self.db_path = db_path
        self.model = None
        self.label_encoders = {}
        self.feature_columns = None
        self.is_trained = False
        
        # Service type multipliers based on data analysis
        self.service_multipliers = {
            'UBERX': 1.0,
            'UBERXL': 1.4,
            'PREMIER': 1.8,
            'SUV_PREMIER': 2.2
        }
        
    def load_data(self):
        """Load data from ride_services table"""
        print("📊 Loading Miami ride_services data...")
        
        conn = sqlite3.connect(self.db_path)
        
        query = """
        SELECT 
            pickup_lat,
            pickup_lng,
            dropoff_lat,
            dropoff_lng,
            distance_km,
            service_type,
            price_usd,
            hour_of_day,
            day_of_week,
            traffic_level,
            weather_condition
        FROM ride_services
        WHERE price_usd > 0 
        AND distance_km > 0
        """
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        print(f"✅ Loaded {len(df):,} ride records")
        print(f"   Service types: {df['service_type'].unique()}")
        print(f"   Average price: ${df['price_usd'].mean():.2f}")
        print(f"   Price range: ${df['price_usd'].min():.2f} - ${df['price_usd'].max():.2f}")
        
        return df
    
    def engineer_features(self, df):
        """Create additional features for better prediction"""
        df_eng = df.copy()
        
        # Distance-based features
        df_eng['distance_squared'] = df_eng['distance_km'] ** 2
        df_eng['distance_log'] = np.log1p(df_eng['distance_km'])
        df_eng['distance_sqrt'] = np.sqrt(df_eng['distance_km'])
        
        # Time-based features
        df_eng['is_rush_hour'] = df_eng['hour_of_day'].apply(
            lambda x: 1 if x in [7, 8, 9, 16, 17, 18, 19] else 0
        )
        df_eng['is_late_night'] = df_eng['hour_of_day'].apply(
            lambda x: 1 if x in [22, 23, 0, 1, 2, 3, 4, 5] else 0
        )
        df_eng['is_weekend'] = df_eng['day_of_week'].apply(
            lambda x: 1 if x in [5, 6] else 0
        )
        
        # Miami-specific location features
        # Airport location
        airport_lat, airport_lng = 25.7959, -80.2870
        df_eng['pickup_to_airport'] = np.sqrt(
            (df_eng['pickup_lat'] - airport_lat) ** 2 +
            (df_eng['pickup_lng'] - airport_lng) ** 2
        )
        df_eng['dropoff_to_airport'] = np.sqrt(
            (df_eng['dropoff_lat'] - airport_lat) ** 2 +
            (df_eng['dropoff_lng'] - airport_lng) ** 2
        )
        
        # South Beach location
        beach_lat, beach_lng = 25.7907, -80.1300
        df_eng['pickup_to_beach'] = np.sqrt(
            (df_eng['pickup_lat'] - beach_lat) ** 2 +
            (df_eng['pickup_lng'] - beach_lng) ** 2
        )
        df_eng['dropoff_to_beach'] = np.sqrt(
            (df_eng['dropoff_lat'] - beach_lat) ** 2 +
            (df_eng['dropoff_lng'] - beach_lng) ** 2
        )
        
        # Downtown Miami
        downtown_lat, downtown_lng = 25.7617, -80.1918
        df_eng['pickup_to_downtown'] = np.sqrt(
            (df_eng['pickup_lat'] - downtown_lat) ** 2 +
            (df_eng['pickup_lng'] - downtown_lng) ** 2
        )
        df_eng['dropoff_to_downtown'] = np.sqrt(
            (df_eng['dropoff_lat'] - downtown_lat) ** 2 +
            (df_eng['dropoff_lng'] - downtown_lng) ** 2
        )
        
        # Special trip indicators
        airport_threshold = 0.02
        df_eng['is_airport_pickup'] = (df_eng['pickup_to_airport'] < airport_threshold).astype(int)
        df_eng['is_airport_dropoff'] = (df_eng['dropoff_to_airport'] < airport_threshold).astype(int)
        df_eng['is_airport_trip'] = (df_eng['is_airport_pickup'] | df_eng['is_airport_dropoff']).astype(int)
        
        # Service type numeric encoding for interactions
        service_numeric = {
            'UBERX': 1,
            'UBERXL': 2,
            'PREMIER': 3,
            'SUV_PREMIER': 4
        }
        df_eng['service_numeric'] = df_eng['service_type'].map(service_numeric)
        
        # Interaction features
        df_eng['distance_service_interaction'] = df_eng['distance_km'] * df_eng['service_numeric']
        df_eng['rush_hour_service_interaction'] = df_eng['is_rush_hour'] * df_eng['service_numeric']
        df_eng['airport_service_interaction'] = df_eng['is_airport_trip'] * df_eng['service_numeric']
        
        return df_eng
    
    def preprocess_data(self, df):
        """Prepare data for XGBoost"""
        df_processed = self.engineer_features(df)
        
        # Encode categorical variables
        categorical_cols = ['service_type', 'traffic_level', 'weather_condition']
        
        for col in categorical_cols:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                df_processed[f'{col}_encoded'] = self.label_encoders[col].fit_transform(df_processed[col])
            else:
                # Handle new categories
                df_processed[f'{col}_encoded'] = df_processed[col].map(
                    lambda x: self.label_encoders[col].transform([x])[0] 
                    if x in self.label_encoders[col].classes_ else -1
                )
        
        # Select features for model
        feature_cols = [
            'distance_km', 'distance_squared', 'distance_log', 'distance_sqrt',
            'pickup_lat', 'pickup_lng', 'dropoff_lat', 'dropoff_lng',
            'hour_of_day', 'day_of_week',
            'is_rush_hour', 'is_late_night', 'is_weekend',
            'pickup_to_airport', 'dropoff_to_airport',
            'pickup_to_beach', 'dropoff_to_beach',
            'pickup_to_downtown', 'dropoff_to_downtown',
            'is_airport_pickup', 'is_airport_dropoff', 'is_airport_trip',
            'service_type_encoded', 'traffic_level_encoded', 'weather_condition_encoded',
            'service_numeric', 'distance_service_interaction',
            'rush_hour_service_interaction', 'airport_service_interaction'
        ]
        
        X = df_processed[feature_cols]
        y = df_processed['price_usd']
        
        return X, y, feature_cols
    
    def train(self):
        """Train the XGBoost model"""
        print("\n🚀 Training XGBoost Miami Multi-Service Model...")
        
        # Load and preprocess data
        df = self.load_data()
        X, y, feature_cols = self.preprocess_data(df)
        self.feature_columns = feature_cols
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        print(f"\n📊 Training set: {len(X_train)} samples")
        print(f"📊 Test set: {len(X_test)} samples")
        
        # XGBoost parameters
        self.model = xgb.XGBRegressor(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            gamma=0.1,
            reg_alpha=0.1,
            reg_lambda=1,
            random_state=42,
            n_jobs=-1
        )
        
        # Cross-validation
        print("\n🔄 Performing cross-validation...")
        cv_scores = cross_val_score(self.model, X_train, y_train, cv=5, scoring='r2')
        print(f"   CV R² Score: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        # Train model
        print("\n📈 Training model...")
        self.model.fit(X_train, y_train)
        
        # Evaluate on test set
        y_pred = self.model.predict(X_test)
        
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        
        print(f"\n📊 Test Set Performance:")
        print(f"   R² Score: {r2:.4f}")
        print(f"   RMSE: ${rmse:.2f}")
        print(f"   MAE: ${mae:.2f}")
        print(f"   MAPE: {mape:.2f}%")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\n🔝 Top 10 Most Important Features:")
        for idx, row in feature_importance.head(10).iterrows():
            print(f"   {row['feature']}: {row['importance']:.4f}")
        
        # Analyze by service type
        print(f"\n📊 Performance by Service Type:")
        for service in df['service_type'].unique():
            mask = df['service_type'] == service
            X_service = X[mask]
            y_service = y[mask]
            y_pred_service = self.model.predict(X_service)
            
            service_r2 = r2_score(y_service, y_pred_service)
            service_rmse = np.sqrt(mean_squared_error(y_service, y_pred_service))
            
            print(f"   {service}: R²={service_r2:.4f}, RMSE=${service_rmse:.2f}")
        
        self.is_trained = True
        print("\n✅ Model training complete!")
        
        return self
    
    def predict(self, pickup_lat, pickup_lng, dropoff_lat, dropoff_lng,
                service_type='UBERX', hour_of_day=12, day_of_week=0,
                traffic_level='moderate', weather_condition='clear'):
        """Make price prediction for a single trip"""
        
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        # Calculate distance
        distance_km = self._calculate_distance(pickup_lat, pickup_lng, dropoff_lat, dropoff_lng)
        
        # Create input dataframe
        input_data = pd.DataFrame({
            'pickup_lat': [pickup_lat],
            'pickup_lng': [pickup_lng],
            'dropoff_lat': [dropoff_lat],
            'dropoff_lng': [dropoff_lng],
            'distance_km': [distance_km],
            'service_type': [service_type],
            'hour_of_day': [hour_of_day],
            'day_of_week': [day_of_week],
            'traffic_level': [traffic_level],
            'weather_condition': [weather_condition],
            'price_usd': [0]  # Dummy for preprocessing
        })
        
        # Preprocess
        X_input, _, _ = self.preprocess_data(input_data)
        X_input = X_input[self.feature_columns]
        
        # Predict
        prediction = self.model.predict(X_input)[0]
        
        return {
            'service_type': service_type,
            'predicted_price': prediction,
            'distance_km': distance_km
        }
    
    def predict_all_services(self, pickup_lat, pickup_lng, dropoff_lat, dropoff_lng,
                            hour_of_day=12, day_of_week=0,
                            traffic_level='moderate', weather_condition='clear'):
        """Predict prices for all service types"""
        
        results = {}
        for service in ['UBERX', 'UBERXL', 'PREMIER', 'SUV_PREMIER']:
            pred = self.predict(
                pickup_lat, pickup_lng, dropoff_lat, dropoff_lng,
                service, hour_of_day, day_of_week,
                traffic_level, weather_condition
            )
            results[service] = pred['predicted_price']
        
        return results
    
    def _calculate_distance(self, lat1, lng1, lat2, lng2):
        """Calculate haversine distance between two points"""
        lat1, lng1, lat2, lng2 = map(np.radians, [lat1, lng1, lat2, lng2])
        dlat = lat2 - lat1
        dlng = lng2 - lng1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlng/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        return 6371 * c  # Earth radius in km
    
    def save_model(self, filepath='xgboost_miami_model.pkl'):
        """Save the trained model"""
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        model_data = {
            'model': self.model,
            'label_encoders': self.label_encoders,
            'feature_columns': self.feature_columns,
            'service_multipliers': self.service_multipliers
        }
        
        joblib.dump(model_data, filepath)
        print(f"✅ Model saved to {filepath}")
    
    def load_model(self, filepath='xgboost_miami_model.pkl'):
        """Load a trained model with numpy version compatibility"""
        # Print numpy version for debugging
        print(f"🔍 Current numpy version: {np.__version__}")
        print(f"🔍 Has numpy._core: {hasattr(np, '_core')}")
        
        try:
            # First try normal joblib load
            model_data = joblib.load(filepath)
        except Exception as e:
            error_str = str(e)
            print(f"⚠️  Initial load failed: {error_str}")
            
            if "numpy._core" in error_str or "numpy.core._multiarray_umath" in error_str:
                print("🔧 Attempting compatibility fix...")
                
                # Try multiple approaches
                try:
                    # Approach 1: Custom unpickler
                    import pickle
                    
                    class NumpyCompatUnpickler(pickle.Unpickler):
                        def find_class(self, module, name):
                            # Remap numpy._core modules to numpy.core
                            if 'numpy._core' in module:
                                module = module.replace('numpy._core', 'numpy.core')
                            # Also handle specific submodules
                            if module == 'numpy.core._multiarray_umath':
                                try:
                                    return getattr(np.core, 'multiarray')
                                except AttributeError:
                                    return getattr(np.core, '_multiarray_umath')
                            # Handle XGBRegressor import issue
                            if name == 'XGBRegressor' and module == '__main__':
                                import xgboost as xgb
                                return xgb.XGBRegressor
                            return super().find_class(module, name)
                    
                    # Load with custom unpickler
                    with open(filepath, 'rb') as f:
                        unpickler = NumpyCompatUnpickler(f)
                        model_data = unpickler.load()
                    
                    print("✅ Model loaded with compatibility layer")
                    
                except Exception as e2:
                    print(f"❌ Compatibility fix also failed: {e2}")
                    # Last resort - try loading with pickle directly
                    try:
                        with open(filepath, 'rb') as f:
                            model_data = pickle.load(f, encoding='latin1')
                        print("✅ Model loaded with pickle encoding='latin1'")
                    except Exception as e3:
                        print(f"❌ All loading attempts failed")
                        raise e3
            else:
                # Re-raise if it's a different error
                raise
        
        self.model = model_data['model']
        self.label_encoders = model_data['label_encoders']
        self.feature_columns = model_data['feature_columns']
        self.service_multipliers = model_data.get('service_multipliers', self.service_multipliers)
        self.is_trained = True
        
        print(f"✅ Model loaded from {filepath}")


if __name__ == "__main__":
    # Train model
    model = XGBoostMiamiModel()
    model.train()
    model.save_model()
    
    # Test predictions
    print("\n" + "="*70)
    print("🧪 TESTING PREDICTIONS")
    print("="*70)
    
    # Test route: Miami Airport to South Beach
    print("\n📍 Route: Miami Airport → South Beach")
    prices = model.predict_all_services(
        pickup_lat=25.7959, pickup_lng=-80.2870,  # Miami Airport
        dropoff_lat=25.7907, dropoff_lng=-80.1300,  # South Beach
        hour_of_day=14,
        day_of_week=2,
        traffic_level='moderate',
        weather_condition='clear'
    )
    
    for service, price in prices.items():
        print(f"   {service}: ${price:.2f}")
    
    # Test route 2: Downtown to Wynwood
    print("\n📍 Route: Downtown Miami → Wynwood")
    prices = model.predict_all_services(
        pickup_lat=25.7617, pickup_lng=-80.1918,  # Downtown
        dropoff_lat=25.8103, dropoff_lng=-80.1934,  # Wynwood
        hour_of_day=20,
        day_of_week=5,  # Saturday
        traffic_level='heavy',
        weather_condition='rain'
    )
    
    for service, price in prices.items():
        print(f"   {service}: ${price:.2f}")