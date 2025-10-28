#!/bin/bash
# SynFinance Health Check Script
# Verify deployment health and API functionality

set -e

# Configuration
API_URL="${API_URL:-http://localhost:8000}"
TIMEOUT=10

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${YELLOW}========================================${NC}"
echo -e "${YELLOW}SynFinance Health Check${NC}"
echo -e "${YELLOW}========================================${NC}"
echo ""

# Function to check endpoint
check_endpoint() {
    local endpoint=$1
    local name=$2
    
    echo -n "Checking ${name}... "
    
    if curl -f -s --max-time ${TIMEOUT} "${API_URL}${endpoint}" > /dev/null 2>&1; then
        echo -e "${GREEN}OK${NC}"
        return 0
    else
        echo -e "${RED}FAILED${NC}"
        return 1
    fi
}

# Health checks
FAILED=0

check_endpoint "/health" "Health Endpoint" || FAILED=$((FAILED + 1))
check_endpoint "/docs" "API Documentation" || FAILED=$((FAILED + 1))
check_endpoint "/openapi.json" "OpenAPI Schema" || FAILED=$((FAILED + 1))

# Check container status (if running in Docker)
if command -v docker &> /dev/null; then
    echo -n "Checking Docker container... "
    if docker ps | grep -q synfinance-prod; then
        echo -e "${GREEN}RUNNING${NC}"
    else
        echo -e "${RED}NOT RUNNING${NC}"
        FAILED=$((FAILED + 1))
    fi
fi

# Final summary
echo ""
echo -e "${YELLOW}========================================${NC}"
if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}All Health Checks PASSED${NC}"
    echo -e "${YELLOW}========================================${NC}"
    exit 0
else
    echo -e "${RED}${FAILED} Health Check(s) FAILED${NC}"
    echo -e "${YELLOW}========================================${NC}"
    exit 1
fi
