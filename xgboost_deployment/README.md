# XGBoost Miami Model Deployment Package

This folder contains all files needed to deploy the XGBoost Miami pricing model API.

## Files Included

- `xgboost_miami_model.pkl` - Trained XGBoost model (28,531 records)
- `src/models/xgboost_miami_model.py` - Model implementation
- `src/api/xgboost_pricing_api.py` - API wrapper for integration
- `requirements.txt` - Python dependencies

## Model Performance

- **Overall R² Score: 0.8822**
- **RMSE: $13.31**
- **Training Data: 28,531 Miami rides**

## Deployment Instructions

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Basic Usage:**
   ```python
   from src.api.xgboost_pricing_api import XGBoostPricingAPI
   
   # Initialize API
   api = XGBoostPricingAPI('xgboost_miami_model.pkl')
   
   # Get predictions for all services
   prices = api.predict_all_services(
       pickup_lat=25.7959, pickup_lng=-80.2870,  # Miami Airport
       dropoff_lat=25.7907, dropoff_lng=-80.1300,  # South Beach
       hour_of_day=14,
       day_of_week=2,
       traffic_level='moderate',
       weather_condition='clear'
   )
   
   # Result: {'UberX': 29.94, 'UberXL': 53.04, 'Uber Premier': 61.39, 'Premier SUV': 70.59}
   ```

3. **Integration with Existing Systems:**
   ```python
   # Use the wrapper class for GUI compatibility
   from src.api.xgboost_pricing_api import XGBoostModelWrapper
   
   model = XGBoostModelWrapper()
   prices = model.predict_all_services(
       pickup_lat, pickup_lng, 
       dropoff_lat, dropoff_lng
   )
   ```

## API Methods

- `predict_all_services()` - Returns prices for all 4 service types
- `predict_price()` - Single service prediction (backward compatible)
- `predict()` - General prediction method
- `get_model_info()` - Model status and information

## Service Types

- UBERX (UberX)
- UBERXL (UberXL) 
- PREMIER (Uber Premier)
- SUV_PREMIER (Premier SUV)

## Notes

- Model automatically handles time-based features if not provided
- Includes Miami-specific location features (airport, beach, downtown)
- No database required for predictions - all encoders stored in .pkl file