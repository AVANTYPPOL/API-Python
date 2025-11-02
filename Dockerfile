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

# Validate hybrid model can load (fail fast if incompatible)
RUN python -c "from hybrid_pricing_api import HybridPricingAPI; \
    api = HybridPricingAPI('booking_fee_model.pkl'); \
    print('[CHECK] Hybrid model loaded:', api.is_loaded); \
    assert api.is_loaded, 'Hybrid model failed to load in container'; \
    print('[OK] Hybrid pricing model validation passed')"

# Expose port
EXPOSE 5000

# Use the clean production app
CMD ["gunicorn", "-c", "gunicorn.conf.py", "app:app"] 