#!/bin/bash

# Arabic Financial RAG System - Phase 1 Deployment Script
# This script automates the complete deployment process

set -e  # Exit on error

echo "============================================================"
echo "Arabic Financial RAG System - Phase 1 Deployment"
echo "============================================================"
echo ""

# Color codes for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check prerequisites
echo -e "${BLUE}[1/5] Checking prerequisites...${NC}"

# Check Docker
if ! command -v docker &> /dev/null; then
    echo -e "${RED}✗ Docker not found. Please install Docker first.${NC}"
    echo "See database/DOCKER_SETUP.md for installation instructions."
    exit 1
fi
echo -e "${GREEN}✓ Docker found: $(docker --version)${NC}"

# Check Docker Compose
if ! docker compose version &> /dev/null; then
    echo -e "${RED}✗ Docker Compose not found.${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Docker Compose found${NC}"

# Check Python
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}✗ Python3 not found.${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Python found: $(python3 --version)${NC}"

# Check JSON file
if [ ! -f "final_json_18_pages.json" ]; then
    echo -e "${RED}✗ final_json_18_pages.json not found in current directory${NC}"
    exit 1
fi
echo -e "${GREEN}✓ JSON file found ($(wc -l < final_json_18_pages.json) lines)${NC}"
echo ""

# Start PostgreSQL
echo -e "${BLUE}[2/5] Starting PostgreSQL + pgvector...${NC}"
cd database

if docker compose ps | grep -q "Up"; then
    echo -e "${GREEN}✓ PostgreSQL already running${NC}"
else
    docker compose up -d
    echo "Waiting for database to be ready..."
    sleep 5
    
    # Wait for health check
    max_attempts=30
    attempt=0
    while [ $attempt -lt $max_attempts ]; do
        if docker compose exec postgres pg_isready -U arab_rag -d arab_rag_db &> /dev/null; then
            echo -e "${GREEN}✓ PostgreSQL is ready${NC}"
            break
        fi
        attempt=$((attempt + 1))
        sleep 1
    done
    
    if [ $attempt -eq $max_attempts ]; then
        echo -e "${RED}✗ Database failed to start${NC}"
        docker compose logs
        exit 1
    fi
fi

cd ..
echo ""

# Install dependencies
echo -e "${BLUE}[3/5] Installing Python dependencies...${NC}"

# Check if venv is active
if [ -z "$VIRTUAL_ENV" ]; then
    echo "Virtual environment not active. Activating..."
    source bin/activate
fi

pip install -q -r database/requirements_db.txt
echo -e "${GREEN}✓ Dependencies installed${NC}"
echo ""

# Run ingestion
echo -e "${BLUE}[4/5] Running ingestion pipeline...${NC}"
echo "This will:"
echo "  - Download BAAI/bge-m3 model (~2.3 GB, one-time)"
echo "  - Generate embeddings for 417 units"
echo "  - Insert data into PostgreSQL"
echo "  - Verify insertion"
echo ""
echo "Expected time: 2-5 minutes"
echo ""

python database/ingest.py

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Ingestion completed successfully${NC}"
else
    echo -e "${RED}✗ Ingestion failed${NC}"
    exit 1
fi
echo ""

# Run tests
echo -e "${BLUE}[5/5] Running test queries...${NC}"
python database/test_queries.py

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ All tests passed${NC}"
else
    echo -e "${RED}✗ Tests failed${NC}"
    exit 1
fi
echo ""

# Summary
echo "============================================================"
echo -e "${GREEN}✓ DEPLOYMENT COMPLETE${NC}"
echo "============================================================"
echo ""
echo "Your Arabic Financial RAG database is now ready!"
echo ""
echo "Database URL: postgresql://arab_rag:arab_rag_pass_2024@localhost:5432/arab_rag_db"
echo "Total units indexed: 417"
echo ""
echo "Next steps:"
echo "  - Connect to the database: psql postgresql://arab_rag:arab_rag_pass_2024@localhost:5432/arab_rag_db"
echo "  - Run custom queries: python database/test_queries.py"
echo "  - View logs: cd database && docker compose logs -f"
echo "  - Stop database: cd database && docker compose down"
echo ""
echo "Documentation:"
echo "  - Quick start: database/QUICKSTART.md"
echo "  - Full guide: database/README.md"
echo "  - Docker setup: database/DOCKER_SETUP.md"
echo "  - Visual guide: database/VISUAL_GUIDE.md"
echo ""
