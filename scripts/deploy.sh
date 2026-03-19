#!/bin/bash

# --- Configuration for output aesthetics ---
# Define ANSI color codes for success (Green) and No Color
GREEN='\033[0;32m'
NC='\033[0m' 

echo "--------------------------------------------------------"
echo "Starting deployment script..."

echo -e "\n🚀 Triggering remote deployment script on EC2 instance..."
ssh $sshKey -t $EC2_HOST 'cd ~/biasguard; git pull; docker stop $(docker ps -q); docker compose up --build -d'

# 3. Completion Message
# Use -e flag for echo to interpret the \n (newline) and ANSI colors
echo -e "\n${GREEN}✅ DEPLOYMENT COMPLETE! ✨ ALL SYSTEMS GO!${NC}"
echo "--------------------------------------------------------"
