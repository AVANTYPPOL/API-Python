# 🚀 Deployment Guide - Miami Uber Model

**Production deployment guide for the Ultimate Miami Uber price prediction model**

## 📋 Quick Overview

This guide covers how to deploy your trained Miami Uber model to production environments. The model achieves **72.86% accuracy** across all services and is optimized for real-time prediction.

## 🎯 Deployment Options

### 1. **Flask API (Recommended)**
Simple REST API for web applications

### 2. **FastAPI**
High-performance API with automatic documentation

### 3. **AWS Lambda**
Serverless deployment for scalable applications

### 4. **Docker Container**
Containerized deployment for any environment

### 5. **Direct Integration**
Embed directly into your Python application

## 🔧 Option 1: Flask API Deployment

### Basic Flask API

```python
# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
from ultimate_miami_model import UltimateMiamiModel
import os
import logging

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load model once at startup
model = UltimateMiamiModel()
model.load_model('ultimate_miami_model.pkl')
logger.info("🚗 Miami Uber Model loaded successfully")

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': True,
        'services': ['UberX', 'UberXL', 'Uber Premier', 'Premier SUV']
    })

@app.route('/predict', methods=['POST'])
def predict_prices():
    """Predict Uber prices for all services"""
    try:
        data = request.json
        
        # Validate required fields
        required_fields = [
            'distance_km', 'pickup_lat', 'pickup_lng', 
            'dropoff_lat', 'dropoff_lng', 'hour_of_day', 
            'day_of_week'
        ]
        
        for field in required_fields:
            if field not in data:
                return jsonify({
                    'error': f'Missing required field: {field}'
                }), 400
        
        # Set defaults for optional fields
        data.setdefault('surge_multiplier', 1.0)
        data.setdefault('traffic_level', 'moderate')
        data.setdefault('weather_condition', 'clear')
        
        # Validate ranges
        if not (0.1 <= data['distance_km'] <= 100):
            return jsonify({'error': 'distance_km must be between 0.1 and 100'}), 400
        
        if not (0 <= data['hour_of_day'] <= 23):
            return jsonify({'error': 'hour_of_day must be between 0 and 23'}), 400
        
        if not (0 <= data['day_of_week'] <= 6):
            return jsonify({'error': 'day_of_week must be between 0 and 6'}), 400
        
        # Get predictions
        prices = model.predict_all_services(**data)
        
        # Log prediction
        logger.info(f"Prediction: {data['distance_km']}km, UberX: ${prices['UberX']:.2f}")
        
        return jsonify({
            'success': True,
            'prices': prices,
            'input': data,
            'model_version': '1.0',
            'accuracy': '72.86%'
        })
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({
            'error': 'Prediction failed',
            'details': str(e)
        }), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
```

### Run Flask API

```bash
# Install Flask
pip install flask flask-cors

# Start the API
python app.py

# Test the API
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "distance_km": 12,
    "pickup_lat": 25.7617,
    "pickup_lng": -80.1918,
    "dropoff_lat": 25.7907,
    "dropoff_lng": -80.1300,
    "hour_of_day": 18,
    "day_of_week": 1,
    "surge_multiplier": 1.2
  }'
```

## 🔧 Option 2: FastAPI Deployment

### FastAPI Implementation

```python
# fastapi_app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
from ultimate_miami_model import UltimateMiamiModel
import logging

app = FastAPI(
    title="Miami Uber Price Prediction API",
    description="Predict Uber prices for Miami rides with 72% accuracy",
    version="1.0.0"
)

# Load model
model = UltimateMiamiModel()
model.load_model('ultimate_miami_model.pkl')

class PredictionRequest(BaseModel):
    distance_km: float = Field(..., ge=0.1, le=100, description="Distance in kilometers")
    pickup_lat: float = Field(..., ge=25.0, le=26.0, description="Pickup latitude")
    pickup_lng: float = Field(..., ge=-81.0, le=-80.0, description="Pickup longitude")
    dropoff_lat: float = Field(..., ge=25.0, le=26.0, description="Dropoff latitude")
    dropoff_lng: float = Field(..., ge=-81.0, le=-80.0, description="Dropoff longitude")
    hour_of_day: int = Field(..., ge=0, le=23, description="Hour of day (0-23)")
    day_of_week: int = Field(..., ge=0, le=6, description="Day of week (0=Monday)")
    surge_multiplier: Optional[float] = Field(1.0, ge=0.5, le=5.0, description="Surge multiplier")
    traffic_level: Optional[str] = Field("moderate", regex="^(light|moderate|heavy)$")
    weather_condition: Optional[str] = Field("clear", regex="^(clear|clouds|rain|thunderstorm)$")

class PredictionResponse(BaseModel):
    success: bool
    prices: dict
    input: PredictionRequest
    model_version: str
    accuracy: str

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": True,
        "services": ["UberX", "UberXL", "Uber Premier", "Premier SUV"]
    }

@app.post("/predict", response_model=PredictionResponse)
def predict_prices(request: PredictionRequest):
    try:
        # Convert to dict for model
        data = request.dict()
        
        # Get predictions
        prices = model.predict_all_services(**data)
        
        return PredictionResponse(
            success=True,
            prices=prices,
            input=request,
            model_version="1.0",
            accuracy="72.86%"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

# Run with: uvicorn fastapi_app:app --host 0.0.0.0 --port 8000
```

### Run FastAPI

```bash
# Install FastAPI
pip install fastapi uvicorn

# Start the API
uvicorn fastapi_app:app --host 0.0.0.0 --port 8000

# Access interactive docs at: http://localhost:8000/docs
```

## 🔧 Option 3: AWS Lambda Deployment

### Lambda Function

```python
# lambda_function.py
import json
import boto3
from ultimate_miami_model import UltimateMiamiModel
import os

# Global model instance (loaded once per Lambda container)
model = None

def lambda_handler(event, context):
    global model
    
    # Load model on first invocation
    if model is None:
        model = UltimateMiamiModel()
        model.load_model('ultimate_miami_model.pkl')
    
    try:
        # Parse request
        if 'body' in event:
            body = json.loads(event['body'])
        else:
            body = event
        
        # Required fields
        required_fields = [
            'distance_km', 'pickup_lat', 'pickup_lng', 
            'dropoff_lat', 'dropoff_lng', 'hour_of_day', 
            'day_of_week'
        ]
        
        for field in required_fields:
            if field not in body:
                return {
                    'statusCode': 400,
                    'body': json.dumps({
                        'error': f'Missing required field: {field}'
                    })
                }
        
        # Set defaults
        body.setdefault('surge_multiplier', 1.0)
        body.setdefault('traffic_level', 'moderate')
        body.setdefault('weather_condition', 'clear')
        
        # Get predictions
        prices = model.predict_all_services(**body)
        
        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({
                'success': True,
                'prices': prices,
                'model_version': '1.0'
            })
        }
        
    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': 'Prediction failed',
                'details': str(e)
            })
        }
```

### Lambda Deployment Package

```bash
# Create deployment package
mkdir lambda_package
cd lambda_package

# Copy files
cp ../ultimate_miami_model.py .
cp ../ultimate_miami_model.pkl .
cp ../lambda_function.py .

# Install dependencies
pip install -r ../requirements.txt -t .

# Create zip
zip -r ../miami_uber_lambda.zip .
```

## 🔧 Option 4: Docker Deployment

### Dockerfile

```dockerfile
# Dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for better caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY ultimate_miami_model.py .
COPY ultimate_miami_model.pkl .
COPY app.py .

# Set environment variables
ENV PORT=5000
ENV PYTHONUNBUFFERED=1

# Expose port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:5000/health || exit 1

# Run the application
CMD ["python", "app.py"]
```

### Docker Commands

```bash
# Build image
docker build -t miami-uber-model .

# Run container
docker run -p 5000:5000 miami-uber-model

# Run with environment variables
docker run -p 5000:5000 -e PORT=8000 miami-uber-model

# Run in production mode
docker run -d --name miami-uber-api \
  -p 5000:5000 \
  --restart unless-stopped \
  miami-uber-model
```

## 🔧 Option 5: Direct Integration

### Python Application Integration

```python
# your_application.py
from ultimate_miami_model import UltimateMiamiModel

class UberPriceService:
    def __init__(self):
        self.model = UltimateMiamiModel()
        self.model.load_model('ultimate_miami_model.pkl')
    
    def get_prices(self, trip_data):
        """Get prices for a trip"""
        return self.model.predict_all_services(**trip_data)
    
    def get_uberx_price(self, trip_data):
        """Get just UberX price"""
        prices = self.get_prices(trip_data)
        return prices['UberX']
    
    def estimate_trip_cost(self, distance_km, surge=1.0):
        """Quick estimate based on distance"""
        # Use default Miami coordinates
        return self.get_prices({
            'distance_km': distance_km,
            'pickup_lat': 25.7617,
            'pickup_lng': -80.1918,
            'dropoff_lat': 25.7907,
            'dropoff_lng': -80.1300,
            'hour_of_day': 14,
            'day_of_week': 1,
            'surge_multiplier': surge,
            'traffic_level': 'moderate',
            'weather_condition': 'clear'
        })

# Usage
price_service = UberPriceService()
prices = price_service.get_prices(your_trip_data)
```

## 📊 Performance Optimization

### 1. Model Caching

```python
# Cache predictions for identical inputs
import functools
import hashlib
import json

def cache_prediction(func):
    cache = {}
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Create cache key
        key = hashlib.md5(
            json.dumps(kwargs, sort_keys=True).encode()
        ).hexdigest()
        
        if key not in cache:
            cache[key] = func(*args, **kwargs)
        
        return cache[key]
    
    return wrapper

# Apply to prediction method
model.predict_all_services = cache_prediction(model.predict_all_services)
```

### 2. Batch Processing

```python
def batch_predict(model, requests):
    """Process multiple predictions efficiently"""
    results = []
    
    for request in requests:
        try:
            prices = model.predict_all_services(**request)
            results.append({
                'success': True,
                'prices': prices,
                'input': request
            })
        except Exception as e:
            results.append({
                'success': False,
                'error': str(e),
                'input': request
            })
    
    return results
```

### 3. Async Processing

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

async def async_predict(model, request):
    """Async wrapper for predictions"""
    loop = asyncio.get_event_loop()
    
    with ThreadPoolExecutor() as executor:
        result = await loop.run_in_executor(
            executor,
            model.predict_all_services,
            **request
        )
    
    return result
```

## 🔍 Monitoring & Logging

### 1. Prediction Logging

```python
import logging
import json
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def log_prediction(request, response, duration):
    """Log prediction details"""
    log_data = {
        'timestamp': datetime.now().isoformat(),
        'request': request,
        'response': response,
        'duration_ms': duration * 1000,
        'model_version': '1.0'
    }
    
    logger.info(f"PREDICTION: {json.dumps(log_data)}")
```

### 2. Health Monitoring

```python
def check_model_health(model):
    """Check if model is working correctly"""
    try:
        # Test prediction
        test_data = {
            'distance_km': 10,
            'pickup_lat': 25.7617,
            'pickup_lng': -80.1918,
            'dropoff_lat': 25.7907,
            'dropoff_lng': -80.1300,
            'hour_of_day': 14,
            'day_of_week': 1
        }
        
        prices = model.predict_all_services(**test_data)
        
        # Check if prices are reasonable
        if 5 <= prices['UberX'] <= 200:
            return True
        
        return False
        
    except Exception:
        return False
```

## 🚀 Production Checklist

### Pre-Deployment
- [ ] Model trained and tested (accuracy > 70%)
- [ ] All dependencies installed
- [ ] Environment variables configured
- [ ] API endpoints tested
- [ ] Error handling implemented
- [ ] Logging configured

### Security
- [ ] API authentication implemented
- [ ] Rate limiting enabled
- [ ] Input validation added
- [ ] HTTPS enabled
- [ ] CORS configured properly

### Performance
- [ ] Model loading optimized
- [ ] Caching implemented
- [ ] Request/response times < 500ms
- [ ] Memory usage monitored
- [ ] Horizontal scaling ready

### Monitoring
- [ ] Health checks configured
- [ ] Metrics collection enabled
- [ ] Alerts set up
- [ ] Logs aggregated
- [ ] Performance monitoring active

## 🎯 Expected Performance

### API Response Times
- **Model loading**: 2-5 seconds (startup only)
- **Prediction**: 10-50ms per request
- **Batch processing**: 5-15ms per prediction

### Resource Usage
- **Memory**: 200-500MB
- **CPU**: Low (< 10% under normal load)
- **Storage**: 50MB model + dependencies

### Scalability
- **Concurrent requests**: 100+ (depends on hardware)
- **Throughput**: 1000+ predictions/second
- **Latency**: < 100ms p95

## 📞 Support & Troubleshooting

### Common Issues

**"Model file not found"**
- Ensure `ultimate_miami_model.pkl` is in the same directory
- Check file permissions

**"Prediction takes too long"**
- Implement caching
- Use batch processing
- Check system resources

**"Low accuracy in production"**
- Verify input data quality
- Check for data drift
- Consider retraining

### Getting Help
- Check logs for detailed error messages
- Verify input data format
- Test with known good data
- Monitor system resources

---

**🏖️ Ready to deploy your Miami Uber model to production!**

Choose the deployment option that best fits your needs and follow the step-by-step instructions above. 