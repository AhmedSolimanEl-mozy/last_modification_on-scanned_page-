#!/bin/bash

# Arabic Financial RAG API - Phase 2 Deployment Script
# Automated deployment and testing for FastAPI backend

set -e

GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

echo "============================================================"
echo "Arabic Financial RAG API - Phase 2 Deployment"
echo "============================================================"
echo ""

# Check prerequisites
echo -e "${BLUE}[1/5] Checking prerequisites...${NC}"

# Check database
if ! docker compose -f database/docker-compose.yml ps | grep -q "Up"; then
    echo -e"${RED}✗ Database is not running${NC}"
    echo "Starting database..."
    cd database
    docker compose up -d
    cd ..
    sleep 5
fi

# Verify database has data
count=$(psql postgresql://arab_rag:arab_rag_pass_2024@localhost:5432/arab_rag_db -t -c "SELECT COUNT(*) FROM information_units;" 2>/dev/null || echo "0")
if [ "$count" -lt 400 ]; then
    echo -e "${RED}✗ Database has insufficient data ($count units)${NC}"
    echo "Please run Phase 1 ingestion first: python database/ingest.py"
    exit 1
fi

echo -e "${GREEN}✓ Database ready with $count units${NC}"

# Check Python dependencies
echo ""
echo -e "${BLUE}[2/5] Installing Python dependencies...${NC}"
pip install -q -r api/requirements.txt
echo -e "${GREEN}✓ Dependencies installed${NC}"

# Check .env file
echo ""
echo -e "${BLUE}[3/5] Checking configuration...${NC}"
if [ ! -f "api/.env" ]; then
    cp api/.env.example api/.env
    echo -e "${GREEN}✓ Created api/.env from template${NC}"
else
    echo -e "${GREEN}✓ api/.env exists${NC}"
fi

# Start API server in background
echo ""
echo -e "${BLUE}[4/5] Starting API server...${NC}"
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 &
API_PID=$!

# Wait for server to start
echo "Waiting for API to be ready..."
max_attempts=30
attempt=0
while [ $attempt -lt $max_attempts ]; do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo -e "${GREEN}✓ API server is ready (PID: $API_PID)${NC}"
        break
    fi
    attempt=$((attempt + 1))
    sleep 1
done

if [ $attempt -eq $max_attempts ]; then
    echo -e "${RED}✗ API failed to start${NC}"
    kill $API_PID 2>/dev/null || true
    exit 1
fi

# Run tests
echo ""
echo -e "${BLUE}[5/5] Running test suite...${NC}"
python api/test_api.py

# Summary
echo ""
echo "============================================================"
echo -e "${GREEN}✓ DEPLOYMENT COMPLETE${NC}"
echo "============================================================"
echo ""
echo "API is running at: http://localhost:8000"
echo "API PID: $API_PID"
echo ""
echo "Useful commands:"
echo "  - View logs: tail -f nohup.out"
echo "  - Stop API: kill $API_PID"
echo "  - Interactive docs: http://localhost:8000/docs"
echo "  - Test again: python api/test_api.py"
echo ""
echo "To stop the API, run: kill $API_PID"
echo ""

# Keep API running (user can Ctrl+C to stop)
echo "API is running. Press Ctrl+C to stop..."
wait $API_PID
