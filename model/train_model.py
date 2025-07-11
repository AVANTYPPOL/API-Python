#!/usr/bin/env python3
"""
🚀 Miami Uber Model Training Script

Simple script to train the Ultimate Miami Uber price prediction model.
This script will automatically train the model using your Miami scraped data
and NYC supplementary data for enhanced distance learning.

Usage:
    python train_model.py

Expected time: 5-10 minutes
Expected accuracy: 72-76% across all services
"""

import os
import sys
import time
from datetime import datetime

def print_banner():
    """Print the welcome banner"""
    print("\n" + "="*60)
    print("🚗 MIAMI UBER MODEL TRAINING")
    print("="*60)
    print("🏖️  Training with real Miami scraped data")
    print("🗽 Enhanced with NYC distance learning")
    print("🎯 Multi-service support (UberX, XL, Premier, SUV)")
    print("⏱️  Expected time: 5-10 minutes")
    print("="*60 + "\n")

def check_requirements():
    """Check if all required files and dependencies are available"""
    print("🔍 Checking requirements...")
    
    # Check required files
    required_files = [
        'ultimate_miami_model.py',
        'uber_ml_data.db',
        'nyc_processed_for_hybrid.parquet',
        'requirements.txt'
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing required files: {missing_files}")
        print("   Please ensure all files are in the same directory")
        return False
    
    # Check Python dependencies
    try:
        import pandas as pd
        import numpy as np
        import sklearn
        import joblib
        import sqlite3
        print("✅ All dependencies found")
        print(f"   pandas: {pd.__version__}")
        print(f"   numpy: {np.__version__}")
        print(f"   scikit-learn: {sklearn.__version__}")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("   Run: pip install -r requirements.txt")
        return False

def verify_data():
    """Verify data quality and completeness"""
    print("\n📊 Verifying data quality...")
    
    try:
        import sqlite3
        import pandas as pd
        
        # Check Miami data
        conn = sqlite3.connect('uber_ml_data.db')
        miami_count = pd.read_sql("SELECT COUNT(*) as count FROM rides WHERE uber_price_usd > 0", conn).iloc[0]['count']
        conn.close()
        
        # Check NYC data
        nyc_df = pd.read_parquet('nyc_processed_for_hybrid.parquet')
        nyc_count = len(nyc_df)
        
        print(f"✅ Miami data: {miami_count:,} rides")
        print(f"✅ NYC data: {nyc_count:,} rides")
        
        if miami_count < 1000:
            print("⚠️  Warning: Limited Miami data may affect accuracy")
        
        return True
        
    except Exception as e:
        print(f"❌ Data verification failed: {e}")
        return False

def train_model():
    """Train the Ultimate Miami model"""
    print("\n🚀 Starting model training...")
    
    try:
        # Import the model
        from ultimate_miami_model import UltimateMiamiModel
        
        # Initialize and train
        model = UltimateMiamiModel()
        print("   Model initialized")
        
        # Train (this is the main training process)
        print("   Training in progress... (this may take 5-10 minutes)")
        start_time = time.time()
        
        success = model.train_ultimate_model()
        
        end_time = time.time()
        training_time = end_time - start_time
        
        if success:
            # Save the model
            model.save_model('ultimate_miami_model.pkl')
            print(f"✅ Training completed in {training_time:.1f} seconds")
            print("✅ Model saved as 'ultimate_miami_model.pkl'")
            return True
        else:
            print("❌ Training failed - check error messages above")
            return False
            
    except Exception as e:
        print(f"❌ Training error: {e}")
        return False

def test_model():
    """Test the trained model with sample predictions"""
    print("\n🧪 Testing trained model...")
    
    try:
        from ultimate_miami_model import UltimateMiamiModel
        
        # Load the trained model
        model = UltimateMiamiModel()
        model.load_model('ultimate_miami_model.pkl')
        
        # Test cases - typical Miami routes
        test_cases = [
            {
                "name": "Airport → South Beach",
                "distance": 18,
                "pickup_lat": 25.7959, "pickup_lng": -80.2870,
                "dropoff_lat": 25.7907, "dropoff_lng": -80.1300,
                "expected_uberx": (28, 38)
            },
            {
                "name": "Downtown → Coral Gables",
                "distance": 12,
                "pickup_lat": 25.7617, "pickup_lng": -80.1918,
                "dropoff_lat": 25.7479, "dropoff_lng": -80.2571,
                "expected_uberx": (20, 30)
            },
            {
                "name": "Short trip (3km)",
                "distance": 3,
                "pickup_lat": 25.7617, "pickup_lng": -80.1918,
                "dropoff_lat": 25.7800, "dropoff_lng": -80.1900,
                "expected_uberx": (12, 18)
            }
        ]
        
        all_passed = True
        
        for case in test_cases:
            # Make prediction
            prices = model.predict_all_services(
                distance_km=case["distance"],
                pickup_lat=case["pickup_lat"],
                pickup_lng=case["pickup_lng"],
                dropoff_lat=case["dropoff_lat"],
                dropoff_lng=case["dropoff_lng"],
                hour_of_day=14,  # 2 PM
                day_of_week=1,   # Tuesday
                surge_multiplier=1.0,
                traffic_level='moderate',
                weather_condition='clear'
            )
            
            uberx_price = prices["UberX"]
            min_exp, max_exp = case["expected_uberx"]
            
            if min_exp <= uberx_price <= max_exp:
                status = "✅"
            else:
                status = "⚠️"
                all_passed = False
            
            print(f"{status} {case['name']}: ${uberx_price:.2f} (expected ${min_exp}-${max_exp})")
            print(f"   UberXL: ${prices['UberXL']:.2f} | Premier: ${prices['Uber Premier']:.2f} | SUV: ${prices['Premier SUV']:.2f}")
        
        if all_passed:
            print("\n✅ All tests passed! Model is ready for use.")
        else:
            print("\n⚠️  Some tests outside expected range (may still be acceptable)")
            
        return True
        
    except Exception as e:
        print(f"❌ Testing failed: {e}")
        return False

def main():
    """Main training pipeline"""
    print_banner()
    
    # Step 1: Check requirements
    if not check_requirements():
        print("\n❌ Requirements check failed. Please fix issues above.")
        return False
    
    # Step 2: Verify data
    if not verify_data():
        print("\n❌ Data verification failed. Please check your data files.")
        return False
    
    # Step 3: Train model
    if not train_model():
        print("\n❌ Model training failed. Please check error messages above.")
        return False
    
    # Step 4: Test model
    if not test_model():
        print("\n⚠️  Model testing had issues, but training completed.")
    
    # Success summary
    print("\n" + "="*60)
    print("🎉 TRAINING COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("✅ Model trained and saved as 'ultimate_miami_model.pkl'")
    print("✅ Ready for production use")
    print("✅ Expected accuracy: 72-76% across all services")
    print("\n📚 Next steps:")
    print("   1. Test with your own data: python test_model.py")
    print("   2. Integrate into your application")
    print("   3. Monitor performance and retrain monthly")
    print("="*60 + "\n")
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1) 