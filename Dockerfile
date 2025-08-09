# Use Python 3.10 slim image for stable compatibility
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for better caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Validate model can load (fail fast if incompatible)
RUN python -c "from xgboost_pricing_api import XGBoostPricingAPI; \
    api = XGBoostPricingAPI('xgboost_miami_model.pkl'); \
    print('Model loaded:', api.is_loaded); \
    assert api.is_loaded, 'Model failed to load in container'; \
    print('✅ Model validation passed')"

# Expose port
EXPOSE 5000

# Use the clean production app
CMD ["gunicorn", "-c", "gunicorn.conf.py", "app:app"] 