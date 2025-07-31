"""
Miami Uber Pricing - Hybrid ML Model
====================================

A production-ready hybrid machine learning model for predicting Uber ride prices in Miami.
Uses transfer learning: pre-trained on 50K NYC rides, fine-tuned on 1,966 Miami rides.

🎯 Key Features:
- Hybrid Model: 99.9% Miami weight + NYC transfer learning
- High Accuracy: R² = 0.8983, RMSE = $1.46
- Multi-Service: UberX, UberXL, Premier, Premier SUV
- Real-time: Traffic, weather, surge pricing integration

Quick Start:
    from miami_uber_pricing import HybridUberPriceModel
    
    # Initialize and load/train model
    model = HybridUberPriceModel()
    if not model.load_model('hybrid_uber_model.pkl'):
        model.train_full_pipeline()
        model.save_model('hybrid_uber_model.pkl')
    
    # Predict price
    price = model.predict_price(
        distance_km=10,
        pickup_lat=25.7617,  # Downtown Miami
        pickup_lng=-80.1918,
        dropoff_lat=25.7907,  # Miami Beach
        dropoff_lng=-80.1300
    )
    print(f"Predicted price: ${price:.2f}")
"""

__version__ = "2.0.0"
__author__ = "Dynamic Pricing Team"
__email__ = "team@dynamicpricing.com"

# Import the hybrid model as the main class
from .models.hybrid_uber_model import HybridUberPriceModel

# Create easy-to-use aliases
UberPricePredictor = HybridUberPriceModel
HybridModel = HybridUberPriceModel

# Optional: Multi-service API for convenience
try:
    from .api.simple_multi_service_api import SimpleMultiServiceAPI
    MultiServiceAPI = SimpleMultiServiceAPI
    __all__ = ['HybridUberPriceModel', 'UberPricePredictor', 'HybridModel', 'MultiServiceAPI']
except ImportError:
    # If API dependencies are missing, just export the core model
    __all__ = ['HybridUberPriceModel', 'UberPricePredictor', 'HybridModel']

# Package metadata
__package_name__ = "miami_uber_pricing"
__description__ = "Hybrid ML model for Miami Uber pricing with transfer learning"
__url__ = "https://github.com/AVANTYPPOL/ML-Model-Python"
__license__ = "MIT"

# Model specifications
MODEL_SPECS = {
    'accuracy': {'r2_score': 0.8983, 'rmse_usd': 1.46},
    'training_data': {'nyc_rides': 50000, 'miami_rides': 1966},
    'model_type': 'hybrid_transfer_learning',
    'miami_weight': 0.999,
    'calibration_factor': 0.779,
    'services': ['UberX', 'UberXL', 'Uber Premier', 'Premier SUV']
}

