# Contributing to Rideshare Pricing API

Thank you for your interest in contributing to the Rideshare Pricing API! This document provides guidelines for developers working on this project.

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- Git
- Google Cloud SDK (for deployments)
- Docker (optional, for containerization)

### Development Setup

1. **Fork and Clone**
   ```bash
   git clone https://github.com/your-org/rideshare-pricing-api.git
   cd rideshare-pricing-api
   ```

2. **Create Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set Up Environment Variables**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

5. **Run Tests**
   ```bash
   python -m pytest tests/
   ```

6. **Start Development Server**
   ```bash
   python app.py
   ```

## 🔄 Development Workflow

### Branch Strategy

- `main` - Production-ready code, auto-deploys to cloud
- `develop` - Development branch for feature integration
- `feature/*` - Feature branches
- `hotfix/*` - Critical bug fixes

### Workflow Steps

1. **Create Feature Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make Changes**
   - Write code
   - Add tests
   - Update documentation

3. **Test Your Changes**
   ```bash
   # Run all tests
   python -m pytest tests/
   
   # Test specific functionality
   python -m pytest tests/test_api.py::test_predict_endpoint
   
   # Run local server and test manually
   python app.py
   curl http://localhost:5000/health
   ```

4. **Commit Changes**
   ```bash
   git add .
   git commit -m "feat: add new pricing feature"
   ```

5. **Push and Create PR**
   ```bash
   git push origin feature/your-feature-name
   ```

### Commit Message Format

Use conventional commits:

```
type(scope): description

feat: add new feature
fix: fix bug
docs: update documentation
test: add tests
refactor: code refactoring
ci: update CI/CD
```

Examples:
- `feat(api): add batch prediction endpoint`
- `fix(model): correct distance calculation`
- `docs(readme): update API documentation`

## 🧪 Testing

### Running Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_api.py -v

# Run with coverage
python -m pytest tests/ --cov=. --cov-report=html
```

### Test Structure

```
tests/
├── test_api.py          # API endpoint tests
├── test_model.py        # ML model tests
├── test_integration.py  # Integration tests
└── fixtures/           # Test data
```

### Writing Tests

```python
import pytest
from app import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_health_endpoint(client):
    response = client.get('/health')
    assert response.status_code == 200
    assert response.json['status'] == 'healthy'
```

## 📝 Code Standards

### Python Code Style

- Follow PEP 8
- Use type hints where possible
- Maximum line length: 88 characters (Black formatter)
- Use descriptive variable names

### Code Formatting

```bash
# Format code with Black
black .

# Sort imports
isort .

# Lint code
flake8 .
```

### Documentation

- Update README.md for user-facing changes
- Add docstrings to all functions
- Update API documentation for endpoint changes

```python
def predict_price(pickup_lat: float, pickup_lng: float) -> dict:
    """
    Predict ride price based on pickup and dropoff coordinates.
    
    Args:
        pickup_lat: Pickup latitude (-90 to 90)
        pickup_lng: Pickup longitude (-180 to 180)
        
    Returns:
        dict: Price predictions for all service types
        
    Raises:
        ValueError: If coordinates are invalid
    """
    pass
```

## 🚀 Deployment

### Automatic Deployment

- Merging to `main` triggers automatic deployment
- GitHub Actions handles the build and deployment
- Monitor deployment status in GitHub Actions tab

### Manual Deployment

```bash
# Deploy to staging
gcloud run deploy rideshare-pricing-api-staging \
  --source . \
  --region us-central1

# Deploy to production
gcloud run deploy rideshare-pricing-api \
  --source . \
  --region us-central1
```

## 🔧 API Changes

### Adding New Endpoints

1. **Add endpoint to app.py**
   ```python
   @app.route('/new-endpoint', methods=['POST'])
   def new_endpoint():
       # Implementation
       pass
   ```

2. **Add tests**
   ```python
   def test_new_endpoint(client):
       response = client.post('/new-endpoint', json={...})
       assert response.status_code == 200
   ```

3. **Update documentation**
   - Add to README.md
   - Update API examples

### Modifying Model

1. **Update model file**
2. **Update tests**
3. **Update model version in API response**
4. **Test thoroughly before deployment**

## 🐛 Bug Reports

### Before Reporting

1. Check existing issues
2. Reproduce the bug
3. Test with latest version

### Bug Report Template

```markdown
## Bug Description
Clear description of the bug

## Steps to Reproduce
1. Step 1
2. Step 2
3. Step 3

## Expected Behavior
What should happen

## Actual Behavior
What actually happens

## Environment
- Python version:
- API version:
- Request details:
```

## 📊 Performance Guidelines

### API Response Times

- Health endpoint: < 100ms
- Prediction endpoint: < 3 seconds
- Batch endpoint: < 5 seconds

### Code Performance

- Use vectorized operations where possible
- Cache expensive calculations
- Monitor memory usage with large datasets

## 🔒 Security

### API Keys

- Never commit API keys to version control
- Use environment variables
- Rotate keys regularly

### Input Validation

- Validate all user inputs
- Sanitize data before processing
- Use proper error handling

## 📞 Getting Help

### Communication Channels

- **GitHub Issues**: Bug reports and feature requests
- **Team Chat**: Daily coordination
- **Code Reviews**: Technical discussions

### Code Review Process

1. Create pull request
2. Request review from team member
3. Address feedback
4. Merge when approved

## 🏆 Recognition

Contributors are recognized in:
- README.md contributors section
- Release notes
- Team meetings

---

Thank you for contributing to the Rideshare Pricing API! 🚗✨ 