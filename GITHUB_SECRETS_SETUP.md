# GitHub Secrets Setup Guide 🔐

This guide explains how to configure the required GitHub secrets for automatic deployment to Google Cloud Run.

## Required Secrets

You need to set up these 4 secrets in your GitHub repository:

### 1. `GCP_PROJECT_ID`
Your Google Cloud Project ID

### 2. `GCP_SA_KEY` 
Service Account JSON key (base64 encoded)

### 3. `GOOGLE_MAPS_API_KEY`
Google Maps API key for distance calculations

### 4. `WEATHER_API_KEY`
OpenWeatherMap API key for weather-based pricing

---

## Step-by-Step Setup

### Step 1: Create Google Cloud Service Account

1. **Go to Google Cloud Console**
   - Visit: https://console.cloud.google.com/
   - Select your project

2. **Navigate to IAM & Admin > Service Accounts**
   - Click "Create Service Account"

3. **Create Service Account**
   ```
   Name: github-actions-deploy
   Description: Service account for GitHub Actions deployment
   ```

4. **Grant Permissions**
   Add these roles:
   - `Cloud Run Admin`
   - `Cloud Build Service Account`
   - `Storage Admin`
   - `Service Account User`

5. **Create Key**
   - Click on the service account
   - Go to "Keys" tab
   - Click "Add Key" > "Create new key"
   - Choose JSON format
   - Download the JSON file

### Step 2: Prepare the Service Account Key

1. **Encode the JSON key to base64**
   
   **On macOS/Linux:**
   ```bash
   base64 -i path/to/your/service-account-key.json
   ```
   
   **On Windows (PowerShell):**
   ```powershell
   [Convert]::ToBase64String([IO.File]::ReadAllBytes("path\to\your\service-account-key.json"))
   ```

2. **Copy the entire base64 output** (it will be very long)

### Step 3: Get Your Project ID

1. **Find your Google Cloud Project ID**
   - In Google Cloud Console, look at the project selector
   - Or run: `gcloud config get-value project`

### Step 4: Get API Keys

#### Google Maps API Key

1. **Go to Google Cloud Console > APIs & Services > Credentials**
2. **Click "Create Credentials" > "API Key"**
3. **Restrict the key:**
   - Application restrictions: None (or IP addresses if you want)
   - API restrictions: Distance Matrix API, Directions API
4. **Copy the API key**

#### OpenWeatherMap API Key

1. **Sign up at:** https://openweathermap.org/api
2. **Go to your account > API keys**
3. **Copy the API key**

### Step 5: Add Secrets to GitHub

1. **Go to your GitHub repository**
2. **Navigate to Settings > Secrets and variables > Actions**
3. **Click "New repository secret" for each:**

   **Secret 1: GCP_PROJECT_ID**
   ```
   Name: GCP_PROJECT_ID
   Value: your-gcp-project-id
   ```

   **Secret 2: GCP_SA_KEY**
   ```
   Name: GCP_SA_KEY
   Value: [paste the entire base64 encoded JSON here]
   ```

   **Secret 3: GOOGLE_MAPS_API_KEY**
   ```
   Name: GOOGLE_MAPS_API_KEY
   Value: your-google-maps-api-key
   ```

   **Secret 4: WEATHER_API_KEY**
   ```
   Name: WEATHER_API_KEY
   Value: your-openweather-api-key
   ```

---

## Verification

### Test the Setup

1. **Make a small change to your code**
2. **Commit and push to main branch:**
   ```bash
   git add .
   git commit -m "Test CI/CD pipeline"
   git push origin main
   ```

3. **Check GitHub Actions:**
   - Go to your repo > Actions tab
   - Watch the deployment workflow run
   - Should see green checkmarks ✅

### Expected Workflow

When you push to `main`, the workflow will:

1. ✅ **Checkout code**
2. ✅ **Set up Python 3.9**
3. ✅ **Install dependencies**
4. ✅ **Run basic tests** (model loading)
5. ✅ **Authenticate to Google Cloud**
6. ✅ **Deploy to Cloud Run**
7. ✅ **Test the deployed API**
8. ✅ **Create deployment summary**

---

## Troubleshooting

### Common Issues

#### ❌ "Invalid service account key"
- Make sure the base64 encoding is correct
- Ensure no extra spaces or line breaks
- Try re-encoding the JSON file

#### ❌ "Permission denied"
- Check service account has correct roles
- Verify project ID is correct
- Ensure Cloud Run API is enabled

#### ❌ "API key invalid"
- Verify API keys are correct
- Check API key restrictions
- Ensure APIs are enabled in Google Cloud

#### ❌ "Resource already exists"
- The Cloud Run service already exists
- This is normal, deployment will update it

### Enable Required APIs

Make sure these APIs are enabled in Google Cloud:

```bash
gcloud services enable run.googleapis.com
gcloud services enable cloudbuild.googleapis.com
gcloud services enable containerregistry.googleapis.com
```

---

## Security Best Practices

### ✅ Do:
- Use service accounts with minimal required permissions
- Regularly rotate API keys
- Monitor deployment logs for issues
- Use repository secrets (not environment variables)

### ❌ Don't:
- Commit API keys to code
- Share service account keys
- Use overly broad permissions
- Store secrets in plain text

---

## Testing Your Deployment

After successful deployment, test your API:

```bash
# Test health endpoint
curl https://rideshare-pricing-api-721577626239.us-central1.run.app/health

# Test prediction endpoint
curl -X POST https://rideshare-pricing-api-721577626239.us-central1.run.app/predict \
  -H "Content-Type: application/json" \
  -d '{
    "pickup_latitude": 25.7617,
    "pickup_longitude": -80.1918,
    "dropoff_latitude": 25.7907,
    "dropoff_longitude": -80.1300,
    "service_type": "UberX"
  }'
```

---

## Next Steps

Once setup is complete:

1. ✅ **Push changes to main** - triggers automatic deployment
2. ✅ **Monitor GitHub Actions** - ensure deployments succeed
3. ✅ **Test API endpoints** - verify functionality
4. ✅ **Set up monitoring** - track API performance
5. ✅ **Add more tests** - improve CI/CD reliability

---

**🎉 Congratulations! Your CI/CD pipeline is now set up for automatic deployment to Google Cloud Run.**Test deployment with fixed GCP_SA_KEY
