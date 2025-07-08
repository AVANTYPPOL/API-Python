#!/bin/bash

# Ride Share Pricing API Deployment Script
# Supports Docker, Heroku, AWS ECS, and Google Cloud Run

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
APP_NAME="rideshare-pricing-api"
DOCKER_IMAGE="$APP_NAME:latest"
GCP_PROJECT_ID=""
AWS_REGION="us-east-1"
HEROKU_APP_NAME=""

# Helper functions
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

error() {
    echo -e "${RED}[ERROR] $1${NC}"
    exit 1
}

warn() {
    echo -e "${YELLOW}[WARNING] $1${NC}"
}

info() {
    echo -e "${BLUE}[INFO] $1${NC}"
}

# Check prerequisites
check_prerequisites() {
    log "Checking prerequisites..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        error "Docker is not installed"
    fi
    
    # Check if model files exist
    if [ ! -f "Dynamic Pricing/hybrid_uber_model.pkl" ] && [ ! -f "Dynamic Pricing/production_uber_model.pkl" ]; then
        error "No model files found. Please ensure model files are present in Dynamic Pricing directory."
    fi
    
    log "Prerequisites check passed"
}

# Build Docker image
build_docker() {
    log "Building Docker image..."
    docker build -t $DOCKER_IMAGE .
    log "Docker image built successfully"
}

# Test the API locally
test_api() {
    log "Testing API locally..."
    
    # Start container
    docker run -d --name ${APP_NAME}-test -p 5000:5000 $DOCKER_IMAGE
    
    # Wait for container to be ready
    sleep 10
    
    # Run tests
    python test_api.py
    
    # Cleanup
    docker stop ${APP_NAME}-test
    docker rm ${APP_NAME}-test
    
    log "API tests completed successfully"
}

# Deploy to Heroku
deploy_heroku() {
    log "Deploying to Heroku..."
    
    if [ -z "$HEROKU_APP_NAME" ]; then
        error "HEROKU_APP_NAME is not set"
    fi
    
    # Check if Heroku CLI is installed
    if ! command -v heroku &> /dev/null; then
        error "Heroku CLI is not installed"
    fi
    
    # Create Heroku app if it doesn't exist
    heroku apps:info $HEROKU_APP_NAME &> /dev/null || heroku create $HEROKU_APP_NAME
    
    # Set buildpack
    heroku buildpacks:set heroku/python -a $HEROKU_APP_NAME
    
    # Deploy
    git push heroku main
    
    log "Deployed to Heroku: https://$HEROKU_APP_NAME.herokuapp.com"
}

# Deploy to Google Cloud Run
deploy_gcp() {
    log "Deploying to Google Cloud Run..."
    
    if [ -z "$GCP_PROJECT_ID" ]; then
        error "GCP_PROJECT_ID is not set"
    fi
    
    # Check if gcloud CLI is installed
    if ! command -v gcloud &> /dev/null; then
        error "Google Cloud CLI is not installed"
    fi
    
    # Build and push to Google Container Registry
    gcloud builds submit --tag gcr.io/$GCP_PROJECT_ID/$APP_NAME
    
    # Deploy to Cloud Run
    gcloud run deploy $APP_NAME \
        --image gcr.io/$GCP_PROJECT_ID/$APP_NAME \
        --platform managed \
        --region us-central1 \
        --allow-unauthenticated \
        --memory 2Gi \
        --cpu 2 \
        --max-instances 10 \
        --port 5000
    
    log "Deployed to Google Cloud Run"
}

# Deploy to AWS ECS
deploy_aws() {
    log "Deploying to AWS ECS..."
    
    # Check if AWS CLI is installed
    if ! command -v aws &> /dev/null; then
        error "AWS CLI is not installed"
    fi
    
    # Get AWS account ID
    AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
    ECR_REPOSITORY="$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$APP_NAME"
    
    # Create ECR repository if it doesn't exist
    aws ecr describe-repositories --repository-names $APP_NAME --region $AWS_REGION &> /dev/null || \
    aws ecr create-repository --repository-name $APP_NAME --region $AWS_REGION
    
    # Login to ECR
    aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $ECR_REPOSITORY
    
    # Tag and push image
    docker tag $DOCKER_IMAGE $ECR_REPOSITORY:latest
    docker push $ECR_REPOSITORY:latest
    
    log "Image pushed to ECR: $ECR_REPOSITORY:latest"
    info "Please create an ECS service using the AWS Console or CLI"
}

# Deploy to local Docker
deploy_local() {
    log "Deploying locally with Docker Compose..."
    
    # Check if docker-compose is installed
    if ! command -v docker-compose &> /dev/null; then
        error "Docker Compose is not installed"
    fi
    
    # Start services
    docker-compose up -d
    
    # Wait for services to be ready
    sleep 15
    
    # Test the API
    python test_api.py
    
    log "Local deployment completed successfully"
    info "API is available at: http://localhost:5000"
}

# Main deployment function
main() {
    log "Starting deployment process..."
    
    # Parse command line arguments
    DEPLOYMENT_TARGET=${1:-"local"}
    
    case $DEPLOYMENT_TARGET in
        "local")
            check_prerequisites
            build_docker
            deploy_local
            ;;
        "heroku")
            check_prerequisites
            deploy_heroku
            ;;
        "gcp")
            check_prerequisites
            build_docker
            deploy_gcp
            ;;
        "aws")
            check_prerequisites
            build_docker
            deploy_aws
            ;;
        "test")
            check_prerequisites
            build_docker
            test_api
            ;;
        *)
            error "Invalid deployment target: $DEPLOYMENT_TARGET"
            echo "Usage: $0 [local|heroku|gcp|aws|test]"
            exit 1
            ;;
    esac
    
    log "Deployment process completed successfully!"
}

# Run main function
main "$@" 