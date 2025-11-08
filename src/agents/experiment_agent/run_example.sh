#!/bin/bash

# Example script to run Research Experiment Agent
# This script demonstrates how to set up and run an experiment

set -e  # Exit on error

echo "========================================"
echo "Research Experiment Agent - Example Run"
echo "========================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Step 1: Check if Docker is running
echo -e "\n${YELLOW}[1/5]${NC} Checking Docker..."
if docker ps > /dev/null 2>&1; then
    echo -e "${GREEN}✓${NC} Docker is running"
else
    echo -e "${RED}✗${NC} Docker is not running or not accessible"
    echo "Please start Docker and try again"
    exit 1
fi

# Step 2: Check if container is running
echo -e "\n${YELLOW}[2/5]${NC} Checking Docker container..."
CONTAINER_NAME="${DOCKER_CONTAINER_NAME:-research-agent-container}"

if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo -e "${GREEN}✓${NC} Container '${CONTAINER_NAME}' is running"
else
    echo -e "${YELLOW}⚠${NC} Container '${CONTAINER_NAME}' is not running"
    echo "Available containers:"
    docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
fi

# Step 3: Check Python environment
echo -e "\n${YELLOW}[3/5]${NC} Checking Python environment..."
if python --version > /dev/null 2>&1; then
    PYTHON_VERSION=$(python --version)
    echo -e "${GREEN}✓${NC} ${PYTHON_VERSION}"
else
    echo -e "${RED}✗${NC} Python not found"
    exit 1
fi

# Step 4: Check required packages
echo -e "\n${YELLOW}[4/5]${NC} Checking required packages..."
REQUIRED_PACKAGES=("agents" "openai")
ALL_INSTALLED=true

for package in "${REQUIRED_PACKAGES[@]}"; do
    if python -c "import ${package}" 2>/dev/null; then
        echo -e "${GREEN}✓${NC} ${package} is installed"
    else
        echo -e "${RED}✗${NC} ${package} is NOT installed"
        ALL_INSTALLED=false
    fi
done

if [ "$ALL_INSTALLED" = false ]; then
    echo -e "\n${YELLOW}Installing missing packages...${NC}"
    pip install agents openai
fi

# Step 5: Print configuration
echo -e "\n${YELLOW}[5/5]${NC} Current configuration:"
python config.py

# Ask user to confirm
echo -e "\n${YELLOW}Ready to run experiment?${NC}"
echo "Input type: ${INPUT_TYPE:-idea}"
echo "Input path: ${INPUT_PATH:-using default from config}"
echo "Max iterations: ${MAX_ITERATIONS:-5}"
echo ""
read -p "Continue? (y/n) " -n 1 -r
echo

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted by user"
    exit 0
fi

# Run the experiment
echo -e "\n${GREEN}Starting experiment...${NC}\n"

python main.py \
    --input-type "${INPUT_TYPE:-idea}" \
    ${INPUT_PATH:+--input-path "$INPUT_PATH"} \
    --max-iterations "${MAX_ITERATIONS:-5}"

# Check exit status
if [ $? -eq 0 ]; then
    echo -e "\n${GREEN}✓ Experiment completed successfully!${NC}"
else
    echo -e "\n${RED}✗ Experiment failed${NC}"
    exit 1
fi

