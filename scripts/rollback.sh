#!/bin/bash
# SynFinance Rollback Script
# Rollback to previous deployment

set -e

# Configuration
CONTAINER_NAME="synfinance-prod"
BACKUP_CONTAINER="synfinance-backup"
IMAGE_NAME="synfinance:latest"
PREVIOUS_IMAGE="synfinance:previous"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${YELLOW}========================================${NC}"
echo -e "${YELLOW}SynFinance Deployment Rollback${NC}"
echo -e "${YELLOW}========================================${NC}"

# Check if backup container exists
if ! docker ps -a | grep -q ${BACKUP_CONTAINER}; then
    echo -e "${RED}Error: No backup container found${NC}"
    echo -e "${YELLOW}Cannot rollback without a previous deployment${NC}"
    exit 1
fi

# Stop current container
echo -e "${YELLOW}Stopping current container...${NC}"
docker stop ${CONTAINER_NAME} || true
docker rename ${CONTAINER_NAME} "${CONTAINER_NAME}-failed"

# Restore backup container
echo -e "${YELLOW}Restoring backup container...${NC}"
docker start ${BACKUP_CONTAINER}
docker rename ${BACKUP_CONTAINER} ${CONTAINER_NAME}

# Wait for health check
echo -e "${YELLOW}Waiting for service to be ready...${NC}"
sleep 5

if curl -f http://localhost:8000/health > /dev/null 2>&1; then
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}Rollback Successful!${NC}"
    echo -e "${GREEN}========================================${NC}"
    
    # Clean up failed container
    docker rm ${CONTAINER_NAME}-failed || true
else
    echo -e "${RED}Rollback failed! Health check unsuccessful${NC}"
    exit 1
fi
