# Ultimate Miami Pricing API 🚀

A high-performance machine learning API for rideshare price prediction using the Ultimate Miami Model with transfer learning approach optimized for Miami market specifics.

## 🌍 Live API

**Production URL**: https://rideshare-pricing-api-721577626239.us-central1.run.app

## 📊 Model Performance

- **Accuracy**: 72.86% (R² Score)
- **Response Time**: ~2 seconds
- **Throughput**: 0.49 requests/second
- **Model Type**: Ultimate Miami Model (Miami-first approach with NYC enhancement)

## 🚀 Quick Start

### Test the API
```bash
# Health check
curl https://rideshare-pricing-api-721577626239.us-central1.run.app/health

# Get price prediction
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

## 📡 API Endpoints

### `GET /health`
Returns API health status and model information.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_info": {
    "model_type": "ultimate_miami_model",
    "accuracy": "72.86%",
    "description": "Miami-first approach with NYC enhancement",
    "features": ["distance", "location", "time_patterns", "surge_multiplier", "miami_specifics"],
    "services": ["UberX", "UberXL", "Uber Premier", "Premier SUV"]
  },
  "version": "1.0.0"
}
```

### `POST /predict`
Get price prediction for a single ride.

**Request:**
```json
{
  "pickup_latitude": 25.7617,
  "pickup_longitude": -80.1918,
  "dropoff_latitude": 25.7907,
  "dropoff_longitude": -80.1300,
  "service_type": "UberX"
}
```

**Response:**
```json
{
  "success": true,
  "prediction": {"UberX": 12.45},
  "request_details": {
    "distance_miles": 3.2,
    "distance_km": 5.1,
    "surge_multiplier": 1.0,
    "service_type": "UberX"
  },
  "model_info": {
    "model_type": "ultimate_miami_model",
    "accuracy": "72.86%"
  }
}
```

### `POST /predict/batch`
Get price predictions for multiple rides.

**Request:**
```json
{
  "rides": [
    {
      "pickup_latitude": 25.7617,
      "pickup_longitude": -80.1918,
      "dropoff_latitude": 25.7907,
      "dropoff_longitude": -80.1300,
      "service_type": "UberX"
    }
  ]
}
```

### `GET /model/info`
Get detailed model information.

## ⚙️ API Configuration

### Environment Variables

| Variable | Description | Required | Default |
|----------|-------------|----------|---------|
| `GOOGLE_MAPS_API_KEY` | Google Maps API for real driving distances | No | Haversine distance |
| `WEATHER_API_KEY` | OpenWeatherMap API for weather-based pricing | No | Clear weather assumed |

### API Key Benefits

#### Google Maps API 🗺️
- **Real driving distances** vs straight-line distance
- **Traffic conditions** for accurate timing
- **Cost**: 100,000 free requests/month, then $0.005/request

#### Weather API 🌦️
- **Weather-based surge pricing** (rain/snow = higher prices)
- **Cost**: 60,000 free requests/month, then $0.0015/request

## 🛠️ Development Setup

### Local Development

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd ML-API
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run locally**
   ```bash
   python app.py
   ```

4. **Test locally**
   ```bash
   curl http://localhost:5000/health
   ```

### Environment Setup

Create a `.env` file for local development:
```bash
GOOGLE_MAPS_API_KEY=your_google_maps_key_here
WEATHER_API_KEY=your_weather_api_key_here
```

## 🚀 Deployment

### Automatic Deployment (GitHub Actions)

Every push to `main` branch automatically deploys to Google Cloud Run.

**Required GitHub Secrets:**
- `GCP_PROJECT_ID` - Your Google Cloud Project ID
- `GCP_SA_KEY` - Service Account JSON key (base64 encoded)
- `GOOGLE_MAPS_API_KEY` - Google Maps API key
- `WEATHER_API_KEY` - OpenWeatherMap API key

### Manual Deployment

```bash
gcloud run deploy rideshare-pricing-api \
  --source . \
  --region us-central1 \
  --platform managed \
  --allow-unauthenticated
```

### Add API Keys to Production

```bash
gcloud run services update rideshare-pricing-api \
  --set-env-vars="GOOGLE_MAPS_API_KEY=your_key,WEATHER_API_KEY=your_key" \
  --region=us-central1
```

## 📱 Integration Examples

### JavaScript/React
```javascript
const getPriceEstimate = async (pickup, dropoff) => {
  const response = await fetch('https://rideshare-pricing-api-721577626239.us-central1.run.app/predict', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      pickup_latitude: pickup.lat,
      pickup_longitude: pickup.lng,
      dropoff_latitude: dropoff.lat,
      dropoff_longitude: dropoff.lng,
      service_type: 'UberX'
    })
  });
  
  return await response.json();
};
```

### Python
```python
import requests

def get_price_estimate(pickup_lat, pickup_lng, dropoff_lat, dropoff_lng):
    url = "https://rideshare-pricing-api-721577626239.us-central1.run.app/predict"
    
    data = {
        "pickup_latitude": pickup_lat,
        "pickup_longitude": pickup_lng,
        "dropoff_latitude": dropoff_lat,
        "dropoff_longitude": dropoff_lng,
        "service_type": "UberX"
    }
    
    response = requests.post(url, json=data)
    return response.json()
```

### Swift/iOS
```swift
func getPriceEstimate(pickup: CLLocationCoordinate2D, dropoff: CLLocationCoordinate2D) async throws -> PriceResponse {
    let url = URL(string: "https://rideshare-pricing-api-721577626239.us-central1.run.app/predict")!
    
    let requestData = PriceRequest(
        pickup_latitude: pickup.latitude,
        pickup_longitude: pickup.longitude,
        dropoff_latitude: dropoff.latitude,
        dropoff_longitude: dropoff.longitude,
        service_type: "UberX"
    )
    
    var request = URLRequest(url: url)
    request.httpMethod = "POST"
    request.setValue("application/json", forHTTPHeaderField: "Content-Type")
    request.httpBody = try JSONEncoder().encode(requestData)
    
    let (data, _) = try await URLSession.shared.data(for: request)
    return try JSONDecoder().decode(PriceResponse.self, from: data)
}
```

## 🚗 Service Types

| Service Type | Description | Typical Use Case |
|--------------|-------------|------------------|
| `UberX` | Standard ride | Most common |
| `UberXL` | Larger vehicle | Groups of 5-6 |
| `Uber Premier` | Premium vehicle | Business travel |
| `Premier SUV` | Premium SUV | Luxury + space |

## 📈 Performance Metrics

- **Latency**: P95 < 3 seconds
- **Availability**: 99.9% uptime
- **Accuracy**: 72.86% within 15% of actual price
- **Rate Limiting**: 1000 requests/minute per IP

## 🔒 Security

- **HTTPS only** - All communications encrypted
- **Rate limiting** - Prevents abuse
- **Input validation** - Prevents malicious data
- **API key rotation** - Secure credential management

## ❌ Error Handling

### Common Error Responses

```json
{
  "success": false,
  "error": "Invalid coordinates",
  "message": "Latitude must be between -90 and 90"
}
```

### HTTP Status Codes

- `200` - Success
- `400` - Bad Request (invalid input)
- `429` - Rate Limited
- `500` - Internal Server Error

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📞 Support

For technical support or questions:
- Create an issue in this repository
- Contact the ML team
- Check the API documentation

## 📝 Version History

- **v1.0.0** - Initial release with Ultimate Miami Model
- **v1.1.0** - Added Google Maps API integration
- **v1.2.0** - Added Weather API integration

---

**Built with ❤️ by the ML Team**# CI/CD Pipeline Status: ✅ Working
