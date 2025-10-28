#!/bin/bash
# SynFinance Deployment Script
# Deploy SynFinance to production environment

set -e  # Exit on error

# Configuration
APP_NAME="synfinance"
IMAGE_NAME="synfinance:latest"
CONTAINER_NAME="synfinance-prod"
API_PORT=8000
HEALTH_CHECK_URL="http://localhost:${API_PORT}/health"
MAX_RETRIES=30
RETRY_INTERVAL=2

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}SynFinance Production Deployment${NC}"
echo -e "${GREEN}========================================${NC}"

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo -e "${RED}Error: Docker is not running${NC}"
    exit 1
fi

# Pull latest image
echo -e "${YELLOW}Pulling latest Docker image...${NC}"
docker pull ${IMAGE_NAME} || {
    echo -e "${RED}Failed to pull Docker image${NC}"
    exit 1
}

# Stop existing container if running
if docker ps -a | grep -q ${CONTAINER_NAME}; then
    echo -e "${YELLOW}Stopping existing container...${NC}"
    docker stop ${CONTAINER_NAME} || true
    docker rm ${CONTAINER_NAME} || true
fi

# Create necessary directories
echo -e "${YELLOW}Creating necessary directories...${NC}"
mkdir -p ./data ./output ./logs

# Start new container
echo -e "${YELLOW}Starting new container...${NC}"
docker run -d \
    --name ${CONTAINER_NAME} \
    --restart unless-stopped \
    -p ${API_PORT}:8000 \
    -v $(pwd)/data:/app/data:ro \
    -v $(pwd)/output:/app/output \
    -v $(pwd)/logs:/app/logs \
    -e SYNFINANCE_ENV=production \
    -e PYTHONUNBUFFERED=1 \
    ${IMAGE_NAME}

# Wait for container to be ready
echo -e "${YELLOW}Waiting for container to be ready...${NC}"
RETRY_COUNT=0
while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    if curl -f ${HEALTH_CHECK_URL} > /dev/null 2>&1; then
        echo -e "${GREEN}Container is ready!${NC}"
        break
    fi
    
    RETRY_COUNT=$((RETRY_COUNT + 1))
    if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
        echo -e "${RED}Health check failed after ${MAX_RETRIES} retries${NC}"
        echo -e "${YELLOW}Container logs:${NC}"
        docker logs ${CONTAINER_NAME}
        exit 1
    fi
    
    echo -e "${YELLOW}Waiting... (${RETRY_COUNT}/${MAX_RETRIES})${NC}"
    sleep ${RETRY_INTERVAL}
done

# Show container info
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Deployment Successful!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "Container: ${GREEN}${CONTAINER_NAME}${NC}"
echo -e "Image: ${GREEN}${IMAGE_NAME}${NC}"
echo -e "API URL: ${GREEN}http://localhost:${API_PORT}${NC}"
echo -e "Health Check: ${GREEN}${HEALTH_CHECK_URL}${NC}"
echo -e "Swagger Docs: ${GREEN}http://localhost:${API_PORT}/docs${NC}"
echo ""
echo -e "To view logs: ${YELLOW}docker logs -f ${CONTAINER_NAME}${NC}"
echo -e "To stop: ${YELLOW}docker stop ${CONTAINER_NAME}${NC}"
echo ""
