# Gunicorn configuration file
import multiprocessing
import os

# Server socket
bind = f"0.0.0.0:{os.environ.get('PORT', '5000')}"
backlog = 2048

# Worker processes
# Reduce workers to conserve memory for model loading
workers = min(multiprocessing.cpu_count(), 2)  # Max 2 workers
worker_class = "sync"
worker_connections = 1000
timeout = 240  # Extended timeout for model + Google Maps API loading
keepalive = 2
max_requests = 1000
max_requests_jitter = 50

# Security
limit_request_line = 4094
limit_request_fields = 100
limit_request_field_size = 8190

# Logging
accesslog = "-"
errorlog = "-"
loglevel = "info"
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s'

# Process naming
proc_name = "rideshare_hybrid_pricing_api"

# Server mechanics
preload_app = False  # Changed to False to avoid model loading issues
daemon = False
pidfile = "/tmp/gunicorn.pid"
tmp_upload_dir = None

# Application - clean production app
wsgi_module = "app:app"

# SSL (if needed)
# keyfile = "/path/to/ssl/key.pem"
# certfile = "/path/to/ssl/cert.pem" 