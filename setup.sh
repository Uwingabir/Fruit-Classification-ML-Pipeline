#!/bin/bash

# Fruit Classification ML Pipeline - Setup Script
# This script sets up the complete environment for the project

set -e

echo "============================================="
echo "  Fruit Classification ML Pipeline Setup"
echo "============================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check Python version
echo -e "${YELLOW}[1/7] Checking Python version...${NC}"
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

if ! python3 -c 'import sys; exit(0 if sys.version_info >= (3, 10) else 1)'; then
    echo -e "${RED}Error: Python 3.10 or higher is required${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Python version OK${NC}"
echo ""

# Create virtual environment
echo -e "${YELLOW}[2/7] Creating virtual environment...${NC}"
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo -e "${GREEN}✓ Virtual environment created${NC}"
else
    echo -e "${GREEN}✓ Virtual environment already exists${NC}"
fi
echo ""

# Activate virtual environment
echo -e "${YELLOW}[3/7] Activating virtual environment...${NC}"
source venv/bin/activate
echo -e "${GREEN}✓ Virtual environment activated${NC}"
echo ""

# Upgrade pip
echo -e "${YELLOW}[4/7] Upgrading pip...${NC}"
pip install --upgrade pip setuptools wheel
echo -e "${GREEN}✓ Pip upgraded${NC}"
echo ""

# Install requirements
echo -e "${YELLOW}[5/7] Installing Python packages...${NC}"
pip install -r requirements.txt
echo -e "${GREEN}✓ Packages installed${NC}"
echo ""

# Create necessary directories
echo -e "${YELLOW}[6/7] Creating project directories...${NC}"
mkdir -p models
mkdir -p uploads
mkdir -p data/retrain
mkdir -p logs
echo -e "${GREEN}✓ Directories created${NC}"
echo ""

# Check dataset
echo -e "${YELLOW}[7/7] Checking dataset...${NC}"
if [ -d "archive (2)/dataset/train" ]; then
    train_count=$(find "archive (2)/dataset/train" -type f | wc -l)
    test_count=$(find "archive (2)/dataset/test" -type f | wc -l)
    echo "Training images: $train_count"
    echo "Test images: $test_count"
    echo -e "${GREEN}✓ Dataset found${NC}"
else
    echo -e "${RED}⚠ Warning: Dataset not found in 'archive (2)/dataset/'${NC}"
    echo "Please ensure your dataset is organized as:"
    echo "  archive (2)/dataset/train/{freshapples,freshbanana,...}"
    echo "  archive (2)/dataset/test/{freshapples,freshbanana,...}"
fi
echo ""

# Setup complete
echo "============================================="
echo -e "${GREEN}Setup Complete!${NC}"
echo "============================================="
echo ""
echo "Next steps:"
echo "1. Train the model:"
echo "   jupyter notebook notebook/fruit_classification.ipynb"
echo ""
echo "2. Run the API server:"
echo "   python app.py"
echo ""
echo "3. Or use Docker:"
echo "   docker-compose up -d"
echo ""
echo "4. Run load tests:"
echo "   locust -f locustfile.py --host=http://localhost:8000"
echo ""
echo "============================================="
