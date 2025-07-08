# API Documentation

Complete reference for the Rideshare Pricing API endpoints.

## Base URL

**Production**: https://rideshare-pricing-api-721577626239.us-central1.run.app

**Local Development**: http://localhost:5000

## Authentication

No authentication required. The API is publicly accessible.

## Rate Limiting

- **Limit**: 1000 requests per minute per IP address
- **Headers**: Rate limit information included in response headers
- **Exceeded**: Returns HTTP 429 with retry-after header

## Content Type

All requests and responses use `application/json` content type.

## Response Format

All responses follow this structure:

```json
{
  "success": true,
  "data": { ... },
  "error": null,
  "timestamp": "2025-01-07T10:30:00Z"
}
```

## Endpoints

### Health Check

Check API health and model status.

**Endpoint**: `GET /health`

**Response**:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_info": {
    "model_type": "simple_calibrated",
    "accuracy": "89.8%",
    "features": ["distance", "duration", "time_of_day", "surge_multiplier"]
  },
  "version": "1.0.0",
  "timestamp": "2025-01-07T10:30:00Z"
}
```

**Status Codes**:
- `200` - API healthy
- `503` - API unhealthy (model not loaded)

---

### Model Information

Get detailed information about the loaded model.

**Endpoint**: `GET /model/info`

**Response**:
```json
{
  "success": true,
  "model_info": {
    "model_type": "simple_calibrated",
    "accuracy": "89.8%",
    "features": ["distance", "duration", "time_of_day", "surge_multiplier"],
    "training_data": {
      "total_samples": 50000,
      "cities": ["NYC", "Miami"],
      "date_range": "2023-2024"
    },
    "performance": {
      "rmse": 1.46,
      "r2_score": 0.8983,
      "response_time_ms": 2000
    }
  }
}
```

---

### Single Ride Prediction

Get price prediction for a single ride.

**Endpoint**: `POST /predict`

**Request Body**:
```json
{
  "pickup_latitude": 25.7617,
  "pickup_longitude": -80.1918,
  "dropoff_latitude": 25.7907,
  "dropoff_longitude": -80.1300,
  "service_type": "UberX"
}
```

**Request Parameters**:

| Parameter | Type | Required | Description | Range |
|-----------|------|----------|-------------|-------|
| `pickup_latitude` | float | Yes | Pickup latitude | -90 to 90 |
| `pickup_longitude` | float | Yes | Pickup longitude | -180 to 180 |
| `dropoff_latitude` | float | Yes | Dropoff latitude | -90 to 90 |
| `dropoff_longitude` | float | Yes | Dropoff longitude | -180 to 180 |
| `service_type` | string | No | Service type | See service types |

**Response**:
```json
{
  "success": true,
  "prediction": {
    "UberX": 12.45,
    "UberXL": 18.67,
    "Uber Premier": 24.89,
    "Premier SUV": 32.11
  },
  "request_details": {
    "distance_miles": 3.2,
    "duration_minutes": 12,
    "surge_multiplier": 1.0,
    "time_of_day": "afternoon",
    "weather_condition": "clear"
  },
  "model_info": {
    "model_type": "simple_calibrated",
    "accuracy": "89.8%",
    "prediction_confidence": 0.92
  },
  "timestamp": "2025-01-07T10:30:00Z"
}
```

**Status Codes**:
- `200` - Success
- `400` - Invalid request (bad coordinates, etc.)
- `422` - Validation error
- `500` - Internal server error

---

### Batch Predictions

Get price predictions for multiple rides in a single request.

**Endpoint**: `POST /predict/batch`

**Request Body**:
```json
{
  "rides": [
    {
      "pickup_latitude": 25.7617,
      "pickup_longitude": -80.1918,
      "dropoff_latitude": 25.7907,
      "dropoff_longitude": -80.1300,
      "service_type": "UberX"
    },
    {
      "pickup_latitude": 25.7800,
      "pickup_longitude": -80.2000,
      "dropoff_latitude": 25.7900,
      "dropoff_longitude": -80.1500,
      "service_type": "UberXL"
    }
  ]
}
```

**Response**:
```json
{
  "success": true,
  "predictions": [
    {
      "ride_index": 0,
      "prediction": {
        "UberX": 12.45,
        "UberXL": 18.67,
        "Uber Premier": 24.89,
        "Premier SUV": 32.11
      },
      "request_details": {
        "distance_miles": 3.2,
        "duration_minutes": 12,
        "surge_multiplier": 1.0
      }
    },
    {
      "ride_index": 1,
      "prediction": {
        "UberX": 8.32,
        "UberXL": 12.48,
        "Uber Premier": 16.64,
        "Premier SUV": 21.60
      },
      "request_details": {
        "distance_miles": 2.1,
        "duration_minutes": 8,
        "surge_multiplier": 1.0
      }
    }
  ],
  "summary": {
    "total_rides": 2,
    "successful_predictions": 2,
    "failed_predictions": 0,
    "average_price_uberx": 10.39
  }
}
```

**Limits**:
- Maximum 50 rides per batch request
- Each ride validated individually
- Failed predictions included with error details

---

## Service Types

| Service Type | Description | Vehicle Type | Capacity |
|--------------|-------------|--------------|----------|
| `UberX` | Standard ride | Sedan | 1-4 passengers |
| `UberXL` | Larger vehicle | SUV/Minivan | 1-6 passengers |
| `Uber Premier` | Premium service | Luxury sedan | 1-4 passengers |
| `Premier SUV` | Premium SUV | Luxury SUV | 1-6 passengers |

## Error Handling

### Error Response Format

```json
{
  "success": false,
  "error": "validation_error",
  "message": "Invalid coordinates provided",
  "details": {
    "field": "pickup_latitude",
    "value": 91.5,
    "constraint": "Must be between -90 and 90"
  },
  "timestamp": "2025-01-07T10:30:00Z"
}
```

### Common Error Codes

| Code | HTTP Status | Description |
|------|-------------|-------------|
| `validation_error` | 400 | Invalid input data |
| `coordinates_invalid` | 400 | Coordinates out of range |
| `service_type_invalid` | 400 | Unknown service type |
| `rate_limit_exceeded` | 429 | Too many requests |
| `model_unavailable` | 503 | ML model not loaded |
| `internal_error` | 500 | Server error |

### Validation Rules

**Coordinates**:
- Latitude: -90 to 90 degrees
- Longitude: -180 to 180 degrees
- Must be numeric (float/int)

**Service Types**:
- Case-insensitive
- Must match available service types
- Defaults to "UberX" if not specified

**Distance Limits**:
- Maximum distance: 100 miles
- Minimum distance: 0.1 miles

## Response Headers

All responses include these headers:

```
Content-Type: application/json
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 999
X-RateLimit-Reset: 1673097600
X-Response-Time: 1234ms
X-Model-Version: 1.0.0
```

## SDKs and Examples

### cURL Examples

**Health Check**:
```bash
curl -X GET https://rideshare-pricing-api-721577626239.us-central1.run.app/health
```

**Single Prediction**:
```bash
curl -X POST https://rideshare-pricing-api-721577626239.us-central1.run.app/predict \
  -H "Content-Type: application/json" \
  -d '{
    "pickup_latitude": 25.7617,
    "pickup_longitude": -80.1918,
    "dropoff_latitude": 25.7907,
    "dropoff_longitude": -80.1300,
    "service_type": "UberX"
  }'
```

**Batch Prediction**:
```bash
curl -X POST https://rideshare-pricing-api-721577626239.us-central1.run.app/predict/batch \
  -H "Content-Type: application/json" \
  -d '{
    "rides": [
      {
        "pickup_latitude": 25.7617,
        "pickup_longitude": -80.1918,
        "dropoff_latitude": 25.7907,
        "dropoff_longitude": -80.1300,
        "service_type": "UberX"
      }
    ]
  }'
```

### Python SDK

```python
import requests

class RideshareAPI:
    def __init__(self, base_url="https://rideshare-pricing-api-721577626239.us-central1.run.app"):
        self.base_url = base_url
    
    def health_check(self):
        response = requests.get(f"{self.base_url}/health")
        return response.json()
    
    def predict_price(self, pickup_lat, pickup_lng, dropoff_lat, dropoff_lng, service_type="UberX"):
        data = {
            "pickup_latitude": pickup_lat,
            "pickup_longitude": pickup_lng,
            "dropoff_latitude": dropoff_lat,
            "dropoff_longitude": dropoff_lng,
            "service_type": service_type
        }
        response = requests.post(f"{self.base_url}/predict", json=data)
        return response.json()
    
    def predict_batch(self, rides):
        data = {"rides": rides}
        response = requests.post(f"{self.base_url}/predict/batch", json=data)
        return response.json()

# Usage
api = RideshareAPI()
result = api.predict_price(25.7617, -80.1918, 25.7907, -80.1300)
print(result)
```

### JavaScript SDK

```javascript
class RideshareAPI {
  constructor(baseUrl = 'https://rideshare-pricing-api-721577626239.us-central1.run.app') {
    this.baseUrl = baseUrl;
  }

  async healthCheck() {
    const response = await fetch(`${this.baseUrl}/health`);
    return response.json();
  }

  async predictPrice(pickupLat, pickupLng, dropoffLat, dropoffLng, serviceType = 'UberX') {
    const data = {
      pickup_latitude: pickupLat,
      pickup_longitude: pickupLng,
      dropoff_latitude: dropoffLat,
      dropoff_longitude: dropoffLng,
      service_type: serviceType
    };
    
    const response = await fetch(`${this.baseUrl}/predict`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(data)
    });
    
    return response.json();
  }

  async predictBatch(rides) {
    const data = { rides };
    
    const response = await fetch(`${this.baseUrl}/predict/batch`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(data)
    });
    
    return response.json();
  }
}

// Usage
const api = new RideshareAPI();
const result = await api.predictPrice(25.7617, -80.1918, 25.7907, -80.1300);
console.log(result);
```

## Performance

### Expected Response Times

| Endpoint | Average | P95 | P99 |
|----------|---------|-----|-----|
| `/health` | 50ms | 100ms | 150ms |
| `/model/info` | 75ms | 150ms | 200ms |
| `/predict` | 2000ms | 3000ms | 4000ms |
| `/predict/batch` | 3000ms | 5000ms | 7000ms |

### Optimization Tips

1. **Use batch predictions** for multiple rides
2. **Cache results** for identical requests
3. **Implement retry logic** for failed requests
4. **Use connection pooling** for high-volume usage

## Monitoring and Metrics

### Health Monitoring

Monitor these endpoints for service health:
- `GET /health` - Overall API health
- Response time monitoring
- Error rate tracking

### Business Metrics

Track these metrics for business insights:
- Prediction accuracy vs actual prices
- Most requested service types
- Geographic usage patterns
- Peak usage times

---

For additional support or questions, please create an issue in the GitHub repository. 