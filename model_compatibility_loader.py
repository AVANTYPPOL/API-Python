#!/usr/bin/env python3
"""
Model compatibility loader that handles numpy version mismatches
"""

import pickle
import numpy as np
import warnings

def load_model_with_compatibility(filepath):
    """
    Load a model file with numpy version compatibility handling
    """
    
    class NumpyCoreCompat:
        """Compatibility shim for numpy._core"""
        multiarray = np.core.multiarray
        umath = np.core.umath
        _internal = np.core._internal
        numeric = np.core.numeric
        fromnumeric = np.core.fromnumeric
        
    # Temporarily inject numpy._core for compatibility
    if not hasattr(np, '_core'):
        np._core = NumpyCoreCompat()
    
    try:
        # Try to load with joblib first
        import joblib
        model_data = joblib.load(filepath)
        print("✅ Model loaded successfully with joblib")
        return model_data
    except Exception as e:
        print(f"⚠️  Joblib load failed: {e}")
        
        # Try with pickle directly
        try:
            with open(filepath, 'rb') as f:
                # Use encoding='latin1' for compatibility
                model_data = pickle.load(f, encoding='latin1')
            print("✅ Model loaded successfully with pickle")
            return model_data
        except Exception as e2:
            print(f"❌ Pickle load also failed: {e2}")
            
            # Last resort - try with custom unpickler
            try:
                class CompatUnpickler(pickle.Unpickler):
                    def find_class(self, module, name):
                        # Remap numpy._core to numpy.core
                        if module.startswith('numpy._core'):
                            module = module.replace('numpy._core', 'numpy.core')
                        return super().find_class(module, name)
                
                with open(filepath, 'rb') as f:
                    unpickler = CompatUnpickler(f)
                    model_data = unpickler.load()
                print("✅ Model loaded successfully with custom unpickler")
                return model_data
            except Exception as e3:
                print(f"❌ All loading methods failed: {e3}")
                raise

def test_compatibility_loader():
    """Test the compatibility loader"""
    print("Testing model compatibility loader...")
    
    try:
        # Load the model
        model_data = load_model_with_compatibility('xgboost_miami_model.pkl')
        
        print("\nModel data loaded successfully!")
        print(f"Keys in model data: {list(model_data.keys())}")
        
        # Test if we can use the model
        from xgboost_miami_model import XGBoostMiamiModel
        
        model = XGBoostMiamiModel()
        model.model = model_data['model']
        model.label_encoders = model_data['label_encoders']
        model.feature_columns = model_data['feature_columns']
        model.service_multipliers = model_data.get('service_multipliers', model.service_multipliers)
        model.is_trained = True
        
        # Test prediction
        predictions = model.predict_all_services(
            pickup_lat=25.7959,
            pickup_lng=-80.2870,
            dropoff_lat=25.7617,
            dropoff_lng=-80.1918
        )
        
        print("\nTest predictions (Airport to South Beach):")
        for service, price in predictions.items():
            print(f"  {service}: ${price:.2f}")
            
        print("\n✅ Compatibility loader works!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_compatibility_loader()