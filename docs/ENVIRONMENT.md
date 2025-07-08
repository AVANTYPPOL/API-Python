# Environment Configuration

This document describes the environment variables used by the Rideshare Pricing API.

## Required Variables

None - the API works without any environment variables using fallback methods.

## Optional Variables

### API Keys

#### Google Maps API
```bash
GOOGLE_MAPS_API_KEY=your_google_maps_api_key_here
```
- **Purpose**: Real driving distances and traffic data
- **Get Key**: https://console.cloud.google.com/apis/credentials
- **Cost**: 100,000 free requests/month, then $0.005/request
- **Fallback**: Haversine distance calculation

#### Weather API
```bash
WEATHER_API_KEY=your_openweather_api_key_here
```
- **Purpose**: Weather-based surge pricing
- **Get Key**: https://openweathermap.org/api
- **Cost**: 60,000 free requests/month, then $0.0015/request
- **Fallback**: Clear weather assumed

### Development Settings

```bash
# Flask environment
FLASK_ENV=development

# Server port
PORT=5000

# Debug mode
DEBUG=true
```

### Production Settings

```bash
# Flask environment
FLASK_ENV=production

# Debug mode
DEBUG=false

# Server port
PORT=8080
```

## Local Development Setup

1. Create a `.env` file in the project root:
   ```bash
   touch .env
   ```

2. Add your environment variables:
   ```bash
   # Optional API keys
   GOOGLE_MAPS_API_KEY=your_key_here
   WEATHER_API_KEY=your_key_here
   
   # Development settings
   FLASK_ENV=development
   PORT=5000
   DEBUG=true
   ```

3. The API will automatically load these variables.

## Cloud Deployment

For Google Cloud Run, set environment variables:

```bash
gcloud run services update rideshare-pricing-api \
  --set-env-vars="GOOGLE_MAPS_API_KEY=your_key,WEATHER_API_KEY=your_key" \
  --region=us-central1
```

## Security Notes

- Never commit API keys to version control
- Use different keys for development and production
- Rotate keys regularly
- Monitor usage to avoid unexpected charges

## Testing Without API Keys

The API is designed to work without API keys:
- Distance calculation uses Haversine formula
- Weather defaults to clear conditions
- All endpoints remain functional
- Model accuracy: 89.8% (same as with API keys) 