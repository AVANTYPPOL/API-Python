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

After making changes, run:
```bash
npm run lint
npm run typecheck
```

## Dependencies

When updating dependencies, ensure compatibility with:
- Python 3.9 (used in GitHub Actions and Cloud Run)
- Existing model files (.pkl format)
- Current API interface