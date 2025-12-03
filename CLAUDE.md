# CLAUDE.md - Important Project Guidelines

## API Interface Stability

**CRITICAL: Do NOT change the API request/response formats**

The API endpoints must maintain their current interface to avoid breaking any internal applications that depend on this service.

### Why This Is Critical
We discovered that even small changes like:
- Adding a `note` field to indicate fallback pricing
- Changing service names from "UberX" to "UBERX" 
- Adding or removing ANY fields from the response

Can and WILL break production applications that parse the API responses. The API contract must remain EXACTLY as specified below.

### Endpoints (DO NOT CHANGE)
- `POST /predict` - Single ride prediction
- `POST /predict/batch` - Batch predictions
- `GET /health` - Health check
- `GET /model/info` - Model information

### Request Format for `/predict` (DO NOT CHANGE)
```json
{
  "pickup_latitude": 25.7959,
  "pickup_longitude": -80.2870,
  "dropoff_latitude": 25.7617,
  "dropoff_longitude": -80.1918
}
```

### Response Format for `/predict` (DO NOT CHANGE)
```json
{
  "success": true,
  "predictions": {
    "PREMIER": 68.20,
    "SUV_PREMIER": 85.90,
    "UBERX": 35.50,
    "UBERXL": 52.75
  },
  "request_details": {
    "distance_miles": 12.3,
    "distance_km": 19.8
  },
  "model_info": {
    "model_type": "xgboost_miami_model",
    "accuracy": "90.93%"
  }
}
```

### Request Format for `/predict/batch` (DO NOT CHANGE)
```json
{
  "rides": [
    {
      "pickup_latitude": 25.7959,
      "pickup_longitude": -80.2870,
      "dropoff_latitude": 25.7617,
      "dropoff_longitude": -80.1918
    }
  ]
}
```

### Optional Query Parameter: `?details=true`

**New Feature:** The `/predict` endpoint now supports an optional `?details=true` query parameter to get comprehensive pricing breakdowns.

**Key Points:**
- **Backward Compatible:** Existing clients are unaffected - the standard response format remains unchanged
- **Opt-in Only:** Detailed information only appears when explicitly requested via `?details=true`
- **Use Cases:** Internal dashboards, pricing analysis tools, debugging, customer support

**Usage:**
```bash
# Standard response (existing behavior)
POST /predict

# Enhanced response with detailed breakdown
POST /predict?details=true
```

**Enhanced Response Format (with `?details=true`):**
```json
{
  "success": true,
  "predictions": {
    "PREMIER": 20.90,
    "SUV_PREMIER": 29.18,
    "UBERX": 12.04,
    "UBERXL": 19.56
  },
  "request_details": {
    "distance_miles": 3.4,
    "distance_km": 5.4,
    "duration_minutes": 10.2
  },
  "service_details": {
    "UBERX": {
      "base_fare": 2.25,
      "per_mile_rate": 0.79,
      "per_minute_rate": 0.20,
      "minimum_fare": 5.70,
      "distance_cost": 4.86,
      "time_cost": 2.04,
      "booking_fee": 5.01,
      "subtotal": 14.17,
      "final_price_before_discount": 14.17,
      "discount_percentage": 15.0,
      "final_price": 12.04
    },
    "UBERXL": { ... },
    "PREMIER": { ... },
    "SUV_PREMIER": { ... }
  },
  "model_info": {
    "model_type": "hybrid_pricing_v3",
    "accuracy": "97.11%"
  }
}
```

**Service Details Fields:**
- `base_fare` - Initial pickup charge
- `per_mile_rate` - Rate per mile
- `per_minute_rate` - Rate per minute
- `minimum_fare` - Minimum price for this service
- `distance_cost` - Total distance charge (distance × per_mile_rate)
- `time_cost` - Total time charge (duration × per_minute_rate)
- `booking_fee` - ML-predicted booking fee based on demand, location, time
- `subtotal` - Sum of base_fare + distance_cost + time_cost + booking_fee
- `final_price_before_discount` - max(subtotal, minimum_fare)
- `discount_percentage` - Applied discount (currently 15%)
- `final_price` - Final price after discount (matches predictions)

**Important:** Without the `?details=true` parameter, the response format remains exactly as documented above in "Response Format for `/predict`" - no breaking changes.

## Development Guidelines

1. **Backend improvements are allowed** - You can fix bugs, improve model accuracy, optimize performance
2. **API contract must remain stable** - Never change field names, data types, or structure
3. **Test with existing clients** - Always verify changes don't break existing integrations
4. **NO EXTRA FIELDS** - Do not add helpful fields like `note`, `debug`, `info` etc.
5. **EXACT RESPONSE FORMAT** - Every response must have the exact same structure

## Model Updates

When updating the ML model:
- Ensure it still accepts the same inputs (coordinates only)
- Ensure it returns all 4 service types: PREMIER, SUV_PREMIER, UBERX, UBERXL
- Maintain the same response structure

## Testing Commands

After making changes, test locally:
```bash
# Start the API locally
python3 app.py

# Test with curl
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "pickup_latitude": 25.7959,
    "pickup_longitude": -80.2870,
    "dropoff_latitude": 25.7617,
    "dropoff_longitude": -80.1918
  }'
```

## Dependencies

When updating dependencies, ensure compatibility with:
- Python 3.9 (used in GitHub Actions and Cloud Run)
- Existing model files (.pkl format)
- Current API interface

**Required Python packages for local development:**
```bash
pip install flask googlemaps
```

## Service Name Consistency

**CRITICAL: Always use internal service names in responses:**
- `UBERX` (internal format)
- `UBERXL` (internal format)  
- `PREMIER` (internal format)
- `SUV_PREMIER` (internal format)

The API returns internal service names in the specific order: PREMIER, SUV_PREMIER, UBERX, UBERXL. This applies to:
- Model predictions
- Fallback pricing
- Error responses
- ALL scenarios

## Recent Critical Fixes (August 2025)

### 1. JSON Serialization Issue
- **Problem**: numpy float32 values were not JSON serializable
- **Solution**: Convert all predictions to Python float before returning
- **Files**: xgboost_miami_model.py, xgboost_pricing_api.py

### 2. Service Name Standardization  
- **Problem**: Inconsistent service naming between model and fallback
- **Solution**: Standardized to use internal names (UBERX, PREMIER, etc.)
- **File**: xgboost_pricing_api.py, app.py

### 3. Missing Required Fields
- **Problem**: Fallback responses missing `model_info` field
- **Solution**: All responses must include ALL required fields
- **File**: app.py

### 4. 503 Timeout Issues
- **Problem**: Model loading during startup caused timeouts
- **Solution**: Lazy loading - model loads on first request
- **Files**: app.py, gunicorn.conf.py

### 5. CORS Support
- **Problem**: Browser/Postman requests blocked
- **Solution**: Added Flask-CORS
- **Files**: app.py, requirements.txt

## Detecting Model vs Fallback Pricing

Since we cannot add fields to indicate fallback usage (breaks API contract), here's how to detect it:

### Fallback Pricing Formula
```python
base_price = 2.50 + (distance_km * 1.65)
UBERX = base_price
UBERXL = base_price * 1.55
PREMIER = base_price * 2.0
SUV_PREMIER = base_price * 2.64
```

### Detection Methods
1. **Check if prices match formula exactly** - Fallback uses simple multiplication
2. **Check server logs** - Look for "Model prediction error" or "fallback"
3. **Test price variation** - Model prices vary by location/time, fallback only by distance
4. **Response time** - Fallback is faster (<50ms) vs model (100-300ms)

### Log Indicators
- **Model Working**: "✅ XGBoost model loaded successfully"
- **Using Fallback**: "❌ Model prediction error" or "Model not loaded"

## Common Issues & Solutions

### Issue: API returns $25 for all trips
**Cause:** Model failed to load, using fallback prices
**Solution:** Check logs for model loading errors, usually dependency version mismatch

### Issue: "No module named 'numpy._core'"
**Cause:** Model was saved with numpy 2.x but loading with numpy 1.x
**Solution:** Ensure numpy versions are compatible between training and deployment

### Issue: GitHub Actions deployment fails
**Cause:** Package version incompatibility with Python 3.9
**Solution:** Use Python 3.9 compatible versions (see requirements.txt)

## Deployment

**GitHub Repository:** https://github.com/AVANTYPPOL/API-Python
**Cloud Provider:** Google Cloud Run
**Service Name:** rideshare-pricing-api
**Region:** us-central1

Deployment is automatic on push to master branch via GitHub Actions.

## Environment Variables

Optional environment variables (in .env file):
- `GOOGLE_MAPS_API_KEY` - For real-time distance calculations
- `WEATHER_API_KEY` - For weather-based pricing adjustments
- `PORT` - API port (default: 5000)

## Model Information

- **Model Type:** XGBoost Multi-Service Predictor
- **Accuracy:** 90.93% (R² score)
- **Training Data:** 157,446 Miami rides (PostgreSQL Cloud SQL)
- **Training Date:** October 18, 2025
- **Services:** PREMIER, SUV_PREMIER, UBERX, UBERXL
- **Model File:** `xgboost_miami_model.pkl`
- **Database:** PostgreSQL on Google Cloud SQL

## Project Structure

```
/
├── app.py                    # Main Flask API
├── xgboost_pricing_api.py    # Model wrapper
├── xgboost_miami_model.py    # Model implementation
├── xgboost_miami_model.pkl   # Trained model file
├── requirements.txt          # Python dependencies
├── Dockerfile               # Container configuration
├── .github/workflows/       # CI/CD configuration
└── CLAUDE.md               # This file
```