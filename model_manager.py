"""
Model Manager for ML API - Handles model loading, versioning, and deployment tracking

Maintains API contract stability while supporting automatic model updates.
"""

import joblib
import json
import os
from datetime import datetime
import logging
from typing import Dict, Any, Optional
import time
import threading
import hashlib
import gc

logger = logging.getLogger(__name__)

class ModelManager:
    """
    Manages ML model loading, versioning, and deployment tracking for the API.
    
    Ensures API contract stability while supporting automatic model updates
    from the scraper repository retraining system.
    """
    
    def __init__(self):
        self.model = None
        self.pricing_api = None
        self.model_version = None
        self.deployment_metadata = None
        self.is_loaded = False
        self.loading = False
        self.load_attempt_time = None
        self._loading_lock = threading.Lock()
        self._model_hash = None
        
        # Legacy compatibility
        self.legacy_model_info = {
            'model_type': 'xgboost_miami_model',
            'accuracy': '88.22%'
        }
        
    def load_model(self) -> bool:
        """
        Load the current model with enhanced version tracking.
        
        Returns:
            bool: True if model loaded successfully
        """
        # Thread-safe model loading
        with self._loading_lock:
            if self.loading:
                logger.info("⏳ Model is already being loaded by another process...")
                return self.is_loaded
                
            if self.is_loaded:
                logger.info("✅ Model already loaded, skipping...")
                return True
                
            self.loading = True
            self.load_attempt_time = datetime.now()
        
        try:
            # Try to load deployment metadata first
            self._load_deployment_metadata()
            
            # Load model file
            model_path = 'xgboost_miami_model.pkl'
            
            if not os.path.exists(model_path):
                logger.error(f"❌ Model file not found: {model_path}")
                return False
                
            logger.info("🚀 Loading XGBoost Miami model with version tracking...")
            
            # Monitor memory before loading
            try:
                import psutil
                process = psutil.Process()
                mem_before = process.memory_info().rss / 1024 / 1024  # MB
                logger.info(f"💾 Memory before model load: {mem_before:.2f} MB")
            except ImportError:
                mem_before = 0
                
            # Load model using existing API
            start_time = time.time()
            from xgboost_pricing_api import XGBoostPricingAPI
            self.pricing_api = XGBoostPricingAPI(model_path)
            load_time = time.time() - start_time
            
            logger.info(f"⏱️ Model loading took {load_time:.2f} seconds")
            
            # Monitor memory after loading
            try:
                mem_after = process.memory_info().rss / 1024 / 1024  # MB
                mem_used = mem_after - mem_before
                logger.info(f"💾 Memory after model load: {mem_after:.2f} MB (used: {mem_used:.2f} MB)")
            except:
                pass
                
            if self.pricing_api and self.pricing_api.is_loaded:
                # Clean up any previous model to prevent memory leaks
                if self.model and self.model != self.pricing_api:
                    del self.model
                    gc.collect()
                    
                self.model = self.pricing_api
                self.is_loaded = True
                
                # Calculate model file hash for integrity tracking
                self._calculate_model_hash(model_path)
                
                logger.info("✅ Model loaded successfully with version tracking")
                self._log_deployment_info()
                
                return True
            else:
                logger.error("❌ Model failed to load properly")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error loading model: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return False
        finally:
            with self._loading_lock:
                self.loading = False
            
    def _load_deployment_metadata(self):
        """Load deployment metadata if available"""
        try:
            # Try new metadata format first
            metadata_path = 'models/model_metadata.json'
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    self.deployment_metadata = json.load(f)
                self.model_version = self.deployment_metadata.get('model_version', 'unknown')
                logger.info(f"📋 Deployment metadata loaded - Version: {self.model_version}")
                return
                
            # Try .env.model format
            env_model_path = '.env.model'
            if os.path.exists(env_model_path):
                env_data = {}
                with open(env_model_path, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if '=' in line and not line.startswith('#'):
                            key, value = line.split('=', 1)
                            env_data[key] = value
                            
                self.model_version = env_data.get('MODEL_VERSION', 'unknown')
                self.deployment_metadata = {
                    'model_version': self.model_version,
                    'deployment_timestamp': env_data.get('DEPLOYMENT_TIMESTAMP'),
                    'validation_metrics': {
                        'validation_mae': env_data.get('VALIDATION_MAE'),
                        'validation_accuracy': env_data.get('VALIDATION_ACCURACY'),
                        'training_samples': env_data.get('TRAINING_SAMPLES')
                    },
                    'deployment_trigger': env_data.get('DEPLOYMENT_TRIGGER', 'manual'),
                    'scraper_repo_commit': env_data.get('SCRAPER_COMMIT')
                }
                logger.info(f"📋 Environment metadata loaded - Version: {self.model_version}")
                return
                
            # Fallback - no metadata available
            self.model_version = 'legacy'
            logger.info("⚠️ No deployment metadata found - using legacy mode")
            
        except Exception as e:
            logger.warning(f"⚠️ Could not load deployment metadata: {e}")
            self.model_version = 'unknown'
            
    def _calculate_model_hash(self, model_path: str):
        """Calculate SHA256 hash of model file for integrity verification"""
        try:
            with open(model_path, 'rb') as f:
                model_data = f.read()
                self._model_hash = hashlib.sha256(model_data).hexdigest()
                logger.info(f"🔐 Model hash calculated: {self._model_hash[:16]}...")
        except Exception as e:
            logger.warning(f"⚠️ Could not calculate model hash: {e}")
            self._model_hash = None
            
    def verify_model_integrity(self, model_path: str) -> bool:
        """Verify model file integrity against stored hash"""
        if not self._model_hash:
            logger.warning("⚠️ No stored hash available for verification")
            return True  # Allow loading if no hash stored
            
        try:
            with open(model_path, 'rb') as f:
                model_data = f.read()
                current_hash = hashlib.sha256(model_data).hexdigest()
                
            if current_hash == self._model_hash:
                logger.info("✅ Model integrity verified")
                return True
            else:
                logger.error(f"❌ Model integrity check failed! Expected: {self._model_hash[:16]}..., Got: {current_hash[:16]}...")
                return False
        except Exception as e:
            logger.error(f"❌ Model integrity verification failed: {e}")
            return False
            
    def _log_deployment_info(self):
        """Log deployment information"""
        logger.info("=" * 70)
        logger.info("📊 MODEL DEPLOYMENT INFORMATION")
        logger.info("=" * 70)
        logger.info(f"🏷️  Model Version: {self.model_version}")
        
        if self.deployment_metadata:
            deployment_time = self.deployment_metadata.get('deployment_timestamp', 'unknown')
            trigger = self.deployment_metadata.get('deployment_trigger', 'unknown')
            
            logger.info(f"🕐 Deployment Time: {deployment_time}")
            logger.info(f"🔄 Deployment Trigger: {trigger}")
            
            # Log validation metrics if available
            metrics = self.deployment_metadata.get('validation_metrics', {})
            if metrics:
                mae = metrics.get('validation_mae', 'unknown')
                accuracy = metrics.get('validation_accuracy', 'unknown')
                samples = metrics.get('training_samples', 'unknown')
                
                logger.info(f"📈 Validation MAE: {mae}")
                logger.info(f"🎯 Validation Accuracy: {accuracy}")
                logger.info(f"📊 Training Samples: {samples}")
                
            scraper_commit = self.deployment_metadata.get('scraper_repo_commit')
            if scraper_commit and scraper_commit != 'unknown':
                logger.info(f"🔗 Scraper Commit: {scraper_commit}")
                
        logger.info("✅ Model ready for production use")
        logger.info("=" * 70)
        
    def predict_all_services(self, pickup_lat: float, pickup_lng: float, 
                           dropoff_lat: float, dropoff_lng: float) -> Dict[str, float]:
        """
        Make predictions for all services using the loaded model.
        
        Args:
            pickup_lat: Pickup latitude
            pickup_lng: Pickup longitude  
            dropoff_lat: Dropoff latitude
            dropoff_lng: Dropoff longitude
            
        Returns:
            Dict[str, float]: Predictions for all services (PREMIER, SUV_PREMIER, UBERX, UBERXL)
        """
        if not self.is_loaded or not self.model:
            raise RuntimeError("Model not loaded")
            
        return self.model.predict_all_services(
            pickup_lat=pickup_lat,
            pickup_lng=pickup_lng,
            dropoff_lat=dropoff_lat,
            dropoff_lng=dropoff_lng
        )
        
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get comprehensive model information for API responses.
        
        Returns:
            Dict containing model info maintaining API contract compatibility
        """
        # Base info for API contract compatibility
        base_info = {
            'model_type': 'xgboost_miami_model',
            'accuracy': '88.22%'
        }
        
        # Enhanced info (internal use, not exposed via API to maintain contract)
        enhanced_info = {
            'version': self.model_version,
            'loaded': self.is_loaded,
            'load_attempt_time': self.load_attempt_time.isoformat() if self.load_attempt_time else None,
            'deployment_metadata': self.deployment_metadata,
            'has_version_tracking': self.deployment_metadata is not None,
            'legacy_mode': self.model_version in ['legacy', 'unknown']
        }
        
        return {
            'api_info': base_info,  # For API responses (contract compliance)
            'internal_info': enhanced_info  # For internal monitoring/debugging
        }
        
    def get_api_model_info(self) -> Dict[str, str]:
        """
        Get model info formatted for API responses (maintains contract).
        
        Returns:
            Dict with model_type and accuracy only (API contract compliance)
        """
        return {
            'model_type': 'xgboost_miami_model',
            'accuracy': '88.22%'
        }
        
    def is_model_loaded(self) -> bool:
        """Check if model is loaded and ready"""
        return self.is_loaded and self.model is not None
        
    def get_deployment_summary(self) -> Dict[str, Any]:
        """
        Get deployment summary for monitoring/debugging.
        
        Returns:
            Dict containing deployment information
        """
        if not self.deployment_metadata:
            return {
                'status': 'legacy_mode',
                'version': self.model_version,
                'message': 'No deployment metadata available'
            }
            
        return {
            'status': 'tracked_deployment',
            'version': self.model_version,
            'deployment_timestamp': self.deployment_metadata.get('deployment_timestamp'),
            'deployment_trigger': self.deployment_metadata.get('deployment_trigger'),
            'validation_metrics': self.deployment_metadata.get('validation_metrics'),
            'scraper_commit': self.deployment_metadata.get('scraper_repo_commit'),
            'model_loaded': self.is_loaded
        }

# Global model manager instance
model_manager = ModelManager()