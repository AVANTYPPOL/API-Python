# CLAUDE.md - Important Project Guidelines

## API Interface Stability

**CRITICAL: Do NOT change the API request/response formats**

The API endpoints must maintain their current interface to avoid breaking any internal applications that depend on this service.

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
    "UberX": 35.50,
    "UberXL": 52.75,
    "Uber Premier": 68.20,
    "Premier SUV": 85.90
  },
  "request_details": {
    "distance_miles": 12.3,
    "distance_km": 19.8
  },
  "model_info": {
    "model_type": "xgboost_miami_model",
    "accuracy": "88.22%"
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

## Development Guidelines

1. **Backend improvements are allowed** - You can fix bugs, improve model accuracy, optimize performance
2. **API contract must remain stable** - Never change field names, data types, or structure
3. **Test with existing clients** - Always verify changes don't break existing integrations

## Model Updates

When updating the ML model:
- Ensure it still accepts the same inputs (coordinates only)
- Ensure it returns all 4 service types: UberX, UberXL, Uber Premier, Premier SUV
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
- **Accuracy:** 88.22% (R² score)
- **Training Data:** 28,531 Miami rides
- **Services:** UberX, UberXL, Uber Premier, Premier SUV
- **Model File:** `xgboost_miami_model.pkl`

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