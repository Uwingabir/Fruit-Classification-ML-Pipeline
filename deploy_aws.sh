#!/bin/bash

# Deploy to AWS EC2 - Quick deployment script

echo "========================================"
echo "  AWS EC2 Deployment Script"
echo "========================================"
echo ""

# Configuration
read -p "Enter your EC2 instance IP: " EC2_IP
read -p "Enter your SSH key path: " SSH_KEY
read -p "Enter SSH user (default: ec2-user): " SSH_USER
SSH_USER=${SSH_USER:-ec2-user}

echo ""
echo "Deploying to: $SSH_USER@$EC2_IP"
echo ""

# Copy files to EC2
echo "[1/4] Copying files to EC2 instance..."
rsync -avz -e "ssh -i $SSH_KEY" \
    --exclude 'venv' \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    --exclude '.git' \
    --exclude 'archive (2)' \
    . $SSH_USER@$EC2_IP:~/ML_Pipeline/

echo "✓ Files copied"
echo ""

# Install dependencies on EC2
echo "[2/4] Installing dependencies on EC2..."
ssh -i $SSH_KEY $SSH_USER@$EC2_IP << 'ENDSSH'
cd ~/ML_Pipeline

# Update system
sudo yum update -y

# Install Docker
if ! command -v docker &> /dev/null; then
    echo "Installing Docker..."
    sudo yum install docker -y
    sudo service docker start
    sudo usermod -a -G docker $USER
fi

# Install Docker Compose
if ! command -v docker-compose &> /dev/null; then
    echo "Installing Docker Compose..."
    sudo curl -L "https://github.com/docker/compose/releases/download/v2.20.0/docker-compose-$(uname -s)-$(uname -m)" \
        -o /usr/local/bin/docker-compose
    sudo chmod +x /usr/local/bin/docker-compose
fi

echo "✓ Dependencies installed"
ENDSSH

echo "✓ Dependencies installed"
echo ""

# Build and start containers
echo "[3/4] Building and starting Docker containers..."
ssh -i $SSH_KEY $SSH_USER@$EC2_IP << 'ENDSSH'
cd ~/ML_Pipeline

# Build and start
sudo docker-compose build
sudo docker-compose up -d

# Wait for services to be healthy
echo "Waiting for services to start..."
sleep 30

# Check status
sudo docker-compose ps

echo "✓ Containers started"
ENDSSH

echo "✓ Deployment complete"
echo ""

# Print access information
echo "========================================"
echo "  Deployment Successful!"
echo "========================================"
echo ""
echo "Access your application at:"
echo "  API: http://$EC2_IP:8000"
echo "  Load Balanced: http://$EC2_IP"
echo "  Grafana: http://$EC2_IP:3000"
echo "  Prometheus: http://$EC2_IP:9090"
echo ""
echo "API Documentation:"
echo "  http://$EC2_IP:8000/docs"
echo ""
echo "To check logs:"
echo "  ssh -i $SSH_KEY $SSH_USER@$EC2_IP 'cd ~/ML_Pipeline && sudo docker-compose logs -f'"
echo ""
echo "To stop services:"
echo "  ssh -i $SSH_KEY $SSH_USER@$EC2_IP 'cd ~/ML_Pipeline && sudo docker-compose down'"
echo ""
echo "========================================"
