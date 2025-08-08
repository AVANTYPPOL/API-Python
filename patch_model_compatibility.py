#!/usr/bin/env python3
"""
Patch the model file to be compatible with numpy 1.x
This creates a new model file that can be loaded with older numpy versions
"""

import pickle
import os
import shutil

class NumpyBackCompatUnpickler(pickle.Unpickler):
    """Unpickler that remaps numpy._core to numpy.core"""
    def find_class(self, module, name):
        if module.startswith('numpy._core'):
            module = module.replace('numpy._core', 'numpy.core')
        return super().find_class(module, name)

class NumpyForwardCompatPickler(pickle.Pickler):
    """Pickler that saves with numpy.core instead of numpy._core"""
    def save_global(self, obj, name=None):
        # Get the module name
        module_name = getattr(obj, '__module__', None)
        if module_name and 'numpy._core' in module_name:
            # Replace _core with core
            obj.__module__ = module_name.replace('numpy._core', 'numpy.core')
        super().save_global(obj, name)

def patch_model_file(input_file='xgboost_miami_model.pkl', output_file='xgboost_miami_model_patched.pkl'):
    """Load model with numpy 2.x and save it in numpy 1.x compatible format"""
    
    print(f"🔄 Patching model file: {input_file}")
    
    # First, backup the original
    backup_file = input_file + '.backup'
    if not os.path.exists(backup_file):
        shutil.copy(input_file, backup_file)
        print(f"📦 Created backup: {backup_file}")
    
    try:
        # Load with compatibility unpickler
        print("📖 Loading model with compatibility layer...")
        with open(input_file, 'rb') as f:
            unpickler = NumpyBackCompatUnpickler(f)
            model_data = unpickler.load()
        
        print("✅ Model loaded successfully")
        
        # Now save it in a compatible format
        print("💾 Saving model in compatible format...")
        
        # Use protocol 4 for better compatibility
        import joblib
        joblib.dump(model_data, output_file, protocol=4)
        
        print(f"✅ Patched model saved to: {output_file}")
        
        # Test loading the patched model
        print("\n🧪 Testing patched model...")
        test_data = joblib.load(output_file)
        print("✅ Patched model loads successfully!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error patching model: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function"""
    print("="*60)
    print("Model Compatibility Patcher")
    print("="*60)
    
    if os.path.exists('xgboost_miami_model.pkl'):
        if patch_model_file():
            print("\n✅ Model patching complete!")
            print("\nNext steps:")
            print("1. Replace the original model file:")
            print("   mv xgboost_miami_model_patched.pkl xgboost_miami_model.pkl")
            print("2. Commit and push:")
            print("   git add xgboost_miami_model.pkl")
            print("   git commit -m 'Use numpy 1.x compatible model file'")
            print("   git push")
    else:
        print("❌ Model file not found!")

if __name__ == "__main__":
    main()