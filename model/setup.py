#!/usr/bin/env python3
"""
🛠️ Miami Uber Model Kit - Setup Script

This script helps you quickly set up the Miami Uber Model Kit.
It will check dependencies, verify data, and guide you through the setup process.

Usage:
    python setup.py
"""

import os
import sys
import subprocess
import platform

def print_banner():
    """Print the welcome banner"""
    print("\n" + "="*60)
    print("🛠️  MIAMI UBER MODEL KIT - SETUP")
    print("="*60)
    print("🚗 Setting up your Miami Uber price prediction model")
    print("📦 This will install dependencies and prepare the environment")
    print("="*60 + "\n")

def check_python_version():
    """Check if Python version is compatible"""
    print("🐍 Checking Python version...")
    
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python {version.major}.{version.minor} is not supported")
        print("   Please install Python 3.8 or higher")
        return False
    
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
    return True

def check_required_files():
    """Check if all required files are present"""
    print("\n📁 Checking required files...")
    
    required_files = [
        'ultimate_miami_model.py',
        'uber_ml_data.db',
        'nyc_processed_for_hybrid.parquet',
        'requirements.txt',
        'train_model.py',
        'test_model.py',
        'example_usage.py'
    ]
    
    missing_files = []
    for file in required_files:
        if os.path.exists(file):
            print(f"✅ {file}")
        else:
            print(f"❌ {file} (missing)")
            missing_files.append(file)
    
    if missing_files:
        print(f"\n❌ Missing files: {missing_files}")
        print("   Please ensure all files are in the same directory")
        return False
    
    return True

def install_dependencies():
    """Install Python dependencies"""
    print("\n📦 Installing dependencies...")
    
    try:
        # Install requirements
        result = subprocess.run([
            sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Dependencies installed successfully")
            return True
        else:
            print(f"❌ Failed to install dependencies:")
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ Error installing dependencies: {e}")
        return False

def verify_installation():
    """Verify all dependencies are working"""
    print("\n🔍 Verifying installation...")
    
    try:
        import pandas as pd
        import numpy as np
        import sklearn
        import joblib
        import sqlite3
        
        print("✅ All dependencies are working correctly")
        print(f"   pandas: {pd.__version__}")
        print(f"   numpy: {np.__version__}")
        print(f"   scikit-learn: {sklearn.__version__}")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def verify_data():
    """Verify data files are valid"""
    print("\n📊 Verifying data files...")
    
    try:
        import sqlite3
        import pandas as pd
        
        # Check Miami database
        conn = sqlite3.connect('uber_ml_data.db')
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM rides WHERE uber_price_usd > 0")
        miami_count = cursor.fetchone()[0]
        conn.close()
        
        # Check NYC parquet file
        nyc_df = pd.read_parquet('nyc_processed_for_hybrid.parquet')
        nyc_count = len(nyc_df)
        
        print(f"✅ Miami data: {miami_count:,} rides")
        print(f"✅ NYC data: {nyc_count:,} rides")
        
        if miami_count < 1000:
            print("⚠️  Warning: Limited Miami data may affect model accuracy")
        
        return True
        
    except Exception as e:
        print(f"❌ Data verification failed: {e}")
        return False

def create_quick_start_script():
    """Create a quick start script"""
    print("\n📝 Creating quick start script...")
    
    script_content = """#!/usr/bin/env python3
# Quick Start - Miami Uber Model Kit
# This script trains the model and runs a quick test

print("🚀 Miami Uber Model Kit - Quick Start")
print("="*50)

# Step 1: Train the model
print("\\n1. Training model...")
import subprocess
result = subprocess.run(["python", "train_model.py"], capture_output=True, text=True)
if result.returncode != 0:
    print("❌ Training failed")
    print(result.stdout)
    print(result.stderr)
    exit(1)

print("✅ Model trained successfully")

# Step 2: Test the model
print("\\n2. Testing model...")
result = subprocess.run(["python", "test_model.py"], capture_output=True, text=True)
if result.returncode != 0:
    print("❌ Testing failed")
    print(result.stdout)
    print(result.stderr)
    exit(1)

print("✅ Model tested successfully")

print("\\n🎉 Setup complete! Your Miami Uber model is ready to use.")
print("\\n📚 Next steps:")
print("   1. Run: python example_usage.py")
print("   2. Integrate the model into your application")
print("   3. See README.md for detailed documentation")
"""
    
    try:
        with open('quick_start.py', 'w') as f:
            f.write(script_content)
        
        print("✅ Created quick_start.py")
        return True
        
    except Exception as e:
        print(f"❌ Failed to create quick start script: {e}")
        return False

def print_system_info():
    """Print system information"""
    print("\n💻 System Information:")
    print(f"   OS: {platform.system()} {platform.release()}")
    print(f"   Python: {sys.version}")
    print(f"   Architecture: {platform.machine()}")

def main():
    """Main setup process"""
    print_banner()
    
    # Print system info
    print_system_info()
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Check required files
    if not check_required_files():
        return False
    
    # Install dependencies
    if not install_dependencies():
        return False
    
    # Verify installation
    if not verify_installation():
        return False
    
    # Verify data
    if not verify_data():
        return False
    
    # Create quick start script
    if not create_quick_start_script():
        return False
    
    # Success message
    print("\n" + "="*60)
    print("🎉 SETUP COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("✅ All dependencies installed")
    print("✅ Data files verified")
    print("✅ Environment ready")
    
    print("\n🚀 Quick Start Options:")
    print("   1. Full setup: python quick_start.py")
    print("   2. Train only: python train_model.py")
    print("   3. Test only: python test_model.py")
    print("   4. Examples: python example_usage.py")
    
    print("\n📚 Documentation:")
    print("   • README.md - Complete guide")
    print("   • TRAINING_GUIDE.md - Step-by-step training")
    print("   • example_usage.py - Usage examples")
    
    print("\n💡 Expected Performance:")
    print("   • Training time: 5-10 minutes")
    print("   • Model accuracy: 72-76%")
    print("   • Prediction time: <0.1 seconds")
    
    print("="*60)
    print("🏖️  Ready to predict Miami Uber prices with 72% accuracy!")
    print("="*60 + "\n")
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⚠️  Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Setup error: {e}")
        sys.exit(1) 