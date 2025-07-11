# 📚 Training Guide - Miami Uber Model

**Step-by-step guide to train your own Miami Uber price prediction model**

## 🎯 Prerequisites

### System Requirements
- **Python**: 3.8 or higher
- **RAM**: 4GB minimum (8GB recommended)
- **Storage**: 500MB free space
- **OS**: Windows, Mac, or Linux

### Required Files
- ✅ `ultimate_miami_model.py` - Main model code
- ✅ `uber_ml_data.db` - Miami scraped data (1,966 trips)
- ✅ `nyc_processed_for_hybrid.parquet` - NYC supplementary data
- ✅ `requirements.txt` - Python dependencies

## 🚀 Step 1: Environment Setup

### Install Python Dependencies
```bash
# Install all required packages
pip install -r requirements.txt

# Or install individually if needed:
pip install pandas numpy scikit-learn joblib sqlite3
```

### Verify Installation
```python
# Test script - run this to verify setup
import pandas as pd
import numpy as np
import sklearn
import joblib
import sqlite3

print("✅ All dependencies installed successfully!")
print(f"   pandas: {pd.__version__}")
print(f"   numpy: {np.__version__}")
print(f"   scikit-learn: {sklearn.__version__}")
```

## 📊 Step 2: Data Verification

### Check Miami Data
```python
import sqlite3
import pandas as pd

# Connect to Miami database
conn = sqlite3.connect('uber_ml_data.db')

# Check data quality
df = pd.read_sql("""
    SELECT COUNT(*) as total_trips,
           COUNT(DISTINCT date(created_at)) as total_days,
           AVG(uberx_price) as avg_uberx_price,
           MIN(distance_km) as min_distance,
           MAX(distance_km) as max_distance
    FROM rides 
    WHERE uber_price_usd > 0
""", conn)

print("📊 Miami Data Summary:")
print(df)
conn.close()

# Expected output:
# total_trips: ~1966
# avg_uberx_price: ~$20-30
# distance range: 0.5-50km
```

### Check NYC Data
```python
# Check NYC supplementary data
nyc_df = pd.read_parquet('nyc_processed_for_hybrid.parquet')

print(f"📊 NYC Data: {len(nyc_df):,} trips")
print(f"   Distance range: {nyc_df['distance_km'].min():.1f} - {nyc_df['distance_km'].max():.1f} km")
print(f"   Price range: ${nyc_df['fare_amount'].min():.2f} - ${nyc_df['fare_amount'].max():.2f}")
```

## 🏗️ Step 3: Model Training

### Option A: Simple Training (Recommended)
```python
# Use the simple training script
python train_model.py
```

### Option B: Manual Training
```python
from ultimate_miami_model import UltimateMiamiModel

# Initialize model
model = UltimateMiamiModel()

# Train (this will take 5-10 minutes)
print("🚀 Starting training...")
success = model.train_ultimate_model()

if success:
    # Save the trained model
    model.save_model('ultimate_miami_model.pkl')
    print("✅ Model trained and saved successfully!")
else:
    print("❌ Training failed - check error messages above")
```

### Expected Training Output
```
🚀 ULTIMATE MIAMI UBER MODEL
============================================================
🏖️  Miami-First Approach (Real Scraped Data)
🗽 NYC Enhancement (Distance Learning)
🚗 Multi-Service Built-in (All 4 Types)

🚀 Training Ultimate Miami Model...

🏖️  Loading Miami Scraped Data...
✅ Loaded 1966 Miami rides with all 4 services
   Average UberX: $XX.XX
   Average distance: XX.X km

🗽 Loading NYC Supplementary Data...
✅ Added 5000 NYC rides for distance learning
   NYC price per km: $4.77
📊 Combined dataset: 3932 rides
   Miami: 1966 rides (primary)
   NYC: 5000 rides (supplementary)

🔧 Engineering Miami-Optimized Features...
✅ Created 42 features for Miami optimization

🎯 Training UBERX model...
   Train R²: 0.9152
   Test R²: 0.7612
   Test RMSE: $8.46
   Top features:
     distance_x_surge    :  17.4%
     distance_km_log     :  15.4%
     distance_km_sqrt    :  11.9%

🎯 Training UBERXL model...
   Train R²: 0.9057
   Test R²: 0.7194
   Test RMSE: $15.78

🎯 Training UBER_PREMIER model...
   Train R²: 0.9054
   Test R²: 0.7150
   Test RMSE: $19.90

🎯 Training PREMIER_SUV model...
   Train R²: 0.9069
   Test R²: 0.7187
   Test RMSE: $25.12

🏆 Overall Model Performance:
   Average R²: 0.7286
   Miami Data Priority: ✅
   Multi-Service: ✅
   Distance Learning: ✅
✅ Ultimate Miami Model saved to ultimate_miami_model.pkl
```

## ✅ Step 4: Model Testing

### Basic Test
```python
# Run the test script
python test_model.py
```

### Manual Testing
```python
from ultimate_miami_model import UltimateMiamiModel

# Load the trained model
model = UltimateMiamiModel()
model.load_model('ultimate_miami_model.pkl')

# Test prediction - Airport to South Beach
prices = model.predict_all_services(
    distance_km=18,
    pickup_lat=25.7959, pickup_lng=-80.2870,  # Miami Airport
    dropoff_lat=25.7907, dropoff_lng=-80.1300,  # South Beach
    hour_of_day=18,      # 6 PM
    day_of_week=1,       # Tuesday
    surge_multiplier=1.0,
    traffic_level='moderate',
    weather_condition='clear'
)

print("🧪 Test Prediction:")
for service, price in prices.items():
    print(f"   {service}: ${price:.2f}")

# Expected output:
# UberX: $32-35
# UberXL: $50-55
# Uber Premier: $60-65
# Premier SUV: $75-85
```

## 📈 Step 5: Performance Validation

### Accuracy Check
```python
# Check model performance on test cases
test_cases = [
    # Short trip
    {"distance": 3, "expected_range": (12, 18)},
    # Medium trip  
    {"distance": 15, "expected_range": (25, 40)},
    # Long trip
    {"distance": 35, "expected_range": (50, 80)}
]

for case in test_cases:
    price = model.predict_all_services(
        distance_km=case["distance"],
        pickup_lat=25.7617, pickup_lng=-80.1918,
        dropoff_lat=25.7907, dropoff_lng=-80.1300
    )["UberX"]
    
    min_exp, max_exp = case["expected_range"]
    status = "✅" if min_exp <= price <= max_exp else "❌"
    print(f"{status} {case['distance']}km: ${price:.2f} (expected ${min_exp}-${max_exp})")
```

## 🔧 Troubleshooting

### Common Training Issues

#### ❌ "ModuleNotFoundError: No module named 'sklearn'"
**Solution:**
```bash
pip install scikit-learn
```

#### ❌ "FileNotFoundError: uber_ml_data.db"
**Solution:**
- Ensure the database file is in the same directory
- Check file permissions
- Verify file wasn't corrupted during transfer

#### ❌ "Training R² is very low (< 0.5)"
**Possible causes:**
- Corrupted or insufficient data
- Wrong coordinate system (ensure lat/lng are correct)
- Data quality issues

**Solution:**
```python
# Check data quality
import sqlite3
conn = sqlite3.connect('uber_ml_data.db')
df = pd.read_sql("SELECT * FROM rides LIMIT 5", conn)
print(df)  # Verify data looks correct
```

#### ❌ "Model predicts constant prices"
**Cause:** Overfitting or bad feature engineering
**Solution:** The Ultimate Miami Model is designed to avoid this issue

### Performance Issues

#### Training Takes Too Long (> 30 minutes)
- **Normal time**: 5-10 minutes
- **If longer**: Check system resources, reduce data size for testing

#### Low Accuracy on New Routes
- **Expected**: Model works best on routes similar to training data
- **Solution**: Add more diverse training data

## 🎯 Advanced Configuration

### Custom Data Sources

#### Using Your Own Miami Data
1. **Replace database**: Update `uber_ml_data.db` with your data
2. **Ensure schema**: Keep same column structure
3. **Required columns**:
   ```sql
   CREATE TABLE rides (
       id INTEGER PRIMARY KEY,
       distance_km REAL,
       pickup_lat REAL,
       pickup_lng REAL,
       dropoff_lat REAL,
       dropoff_lng REAL,
       hour_of_day INTEGER,
       day_of_week INTEGER,
       surge_multiplier REAL,
       traffic_level TEXT,
       weather_condition TEXT,
       uberx_price REAL,
       uber_xl_price REAL,
       uber_premier_price REAL,
       premier_suv_price REAL,
       created_at TEXT
   );
   ```

#### NYC-Only Training
If you don't have Miami data:
```python
# Modify ultimate_miami_model.py
# Comment out Miami data loading
# Use only NYC data (accuracy will be lower ~60-65%)
```

### Model Tuning

#### Hyperparameter Adjustment
```python
# In ultimate_miami_model.py, modify:
model = RandomForestRegressor(
    n_estimators=300,     # Increase trees (slower but more accurate)
    max_depth=20,         # Deeper trees (risk overfitting)
    min_samples_split=3,  # Smaller splits (more complex)
    random_state=42
)
```

#### Feature Selection
```python
# Add custom features in engineer_miami_features():
df_eng['custom_feature'] = your_calculation
```

## 📋 Checklist

### Pre-Training
- [ ] Python 3.8+ installed
- [ ] All dependencies installed (`pip install -r requirements.txt`)
- [ ] `uber_ml_data.db` file present and accessible
- [ ] `nyc_processed_for_hybrid.parquet` file present
- [ ] At least 4GB RAM available

### Training Process
- [ ] Training completes without errors
- [ ] All 4 service models train successfully
- [ ] Test R² scores are > 0.65 for all services
- [ ] Model saves as `ultimate_miami_model.pkl`

### Post-Training
- [ ] Model loads without errors
- [ ] Test predictions return reasonable prices
- [ ] All 4 services return different prices
- [ ] Prices vary appropriately with distance

## 🎉 Success Metrics

### Good Model Performance
- **UberX R²**: > 0.70
- **Other services R²**: > 0.65
- **Price variation**: Prices increase with distance
- **Service differentiation**: Premier > XL > UberX
- **Reasonable ranges**: $10-100 for typical Miami trips

### Ready for Production
- ✅ All tests pass
- ✅ Predictions are consistent
- ✅ Performance meets requirements
- ✅ Model file < 50MB
- ✅ Prediction time < 1 second

---

**🏆 Congratulations! You now have a production-ready Miami Uber price prediction model trained on real scraped data!** 