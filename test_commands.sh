#!/bin/bash

# XGBoost Miami API Test Commands
# ================================

echo "🧪 Testing XGBoost Miami Pricing API"
echo "===================================="

# Set API URL (change if needed)
API_URL="http://localhost:5000"

# 1. Health Check
echo -e "\n1️⃣ Testing Health Endpoint:"
curl -X GET "${API_URL}/health" | python -m json.tool

# 2. Model Info
echo -e "\n\n2️⃣ Testing Model Info Endpoint:"
curl -X GET "${API_URL}/model/info" | python -m json.tool

# 3. Single Prediction - Miami Airport to South Beach
echo -e "\n\n3️⃣ Testing Prediction (Airport → South Beach):"
curl -X POST "${API_URL}/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "pickup_latitude": 25.7959,
    "pickup_longitude": -80.2870,
    "dropoff_latitude": 25.7907,
    "dropoff_longitude": -80.1300
  }' | python -m json.tool

# 4. Single Prediction - Downtown to Wynwood
echo -e "\n\n4️⃣ Testing Prediction (Downtown → Wynwood):"
curl -X POST "${API_URL}/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "pickup_latitude": 25.7617,
    "pickup_longitude": -80.1918,
    "dropoff_latitude": 25.8103,
    "dropoff_longitude": -80.1934
  }' | python -m json.tool

# 5. Batch Prediction
echo -e "\n\n5️⃣ Testing Batch Prediction:"
curl -X POST "${API_URL}/predict/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "rides": [
      {
        "pickup_latitude": 25.7959,
        "pickup_longitude": -80.2870,
        "dropoff_latitude": 25.7907,
        "dropoff_longitude": -80.1300
      },
      {
        "pickup_latitude": 25.7617,
        "pickup_longitude": -80.1918,
        "dropoff_latitude": 25.8103,
        "dropoff_longitude": -80.1934
      }
    ]
  }' | python -m json.tool

echo -e "\n\n✅ Testing complete!"