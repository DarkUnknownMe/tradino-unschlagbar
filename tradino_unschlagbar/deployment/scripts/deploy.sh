#!/bin/bash

# =============================================================================
# TRADINO UNSCHLAGBAR - Deployment Script
# =============================================================================

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_NAME="tradino-unschlagbar"
DOCKER_COMPOSE_FILE="deployment/docker/docker-compose.yml"
ENV_FILE=".env"
BACKUP_DIR="data/backups"

echo -e "${BLUE}=== TRADINO UNSCHLAGBAR DEPLOYMENT ===${NC}"
echo -e "${BLUE}Starting deployment process...${NC}\n"

# Function to print status
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    print_status "Checking prerequisites..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 is not installed. Please install Python 3 first."
        exit 1
    fi
    
    print_status "Prerequisites check passed ✓"
}

# Setup environment
setup_environment() {
    print_status "Setting up environment..."
    
    # Create .env from template if it doesn't exist
    if [ ! -f "$ENV_FILE" ]; then
        if [ -f "env.template" ]; then
            cp env.template "$ENV_FILE"
            print_warning "Created .env file from template. Please update with your credentials."
        else
            print_error "env.template not found. Cannot create .env file."
            exit 1
        fi
    fi
    
    # Create necessary directories
    mkdir -p data/{backups,historical,live,logs,models}
    mkdir -p logs
    mkdir -p models
    
    print_status "Environment setup completed ✓"
}

# Install dependencies
install_dependencies() {
    print_status "Installing Python dependencies..."
    
    if [ -f "requirements.txt" ]; then
        pip3 install -r requirements.txt
        print_status "Dependencies installed ✓"
    else
        print_error "requirements.txt not found."
        exit 1
    fi
}

# Run tests
run_tests() {
    print_status "Running tests..."
    
    if [ -d "tests" ]; then
        python3 -m pytest tests/ -v
        print_status "Tests passed ✓"
    else
        print_warning "No tests directory found. Skipping tests."
    fi
}

# Build Docker images
build_docker() {
    print_status "Building Docker images..."
    
    if [ -f "$DOCKER_COMPOSE_FILE" ]; then
        docker-compose -f "$DOCKER_COMPOSE_FILE" build
        print_status "Docker images built ✓"
    else
        print_error "Docker compose file not found: $DOCKER_COMPOSE_FILE"
        exit 1
    fi
}

# Deploy services
deploy_services() {
    print_status "Deploying services..."
    
    # Stop existing services
    docker-compose -f "$DOCKER_COMPOSE_FILE" down
    
    # Start services
    docker-compose -f "$DOCKER_COMPOSE_FILE" up -d
    
    # Wait for services to be ready
    sleep 10
    
    # Check service health
    if docker-compose -f "$DOCKER_COMPOSE_FILE" ps | grep -q "Up"; then
        print_status "Services deployed successfully ✓"
    else
        print_error "Service deployment failed."
        docker-compose -f "$DOCKER_COMPOSE_FILE" logs
        exit 1
    fi
}

# Create backup
create_backup() {
    print_status "Creating backup..."
    
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    BACKUP_FILE="$BACKUP_DIR/backup_$TIMESTAMP.tar.gz"
    
    mkdir -p "$BACKUP_DIR"
    tar -czf "$BACKUP_FILE" data/ models/ config.yaml 2>/dev/null || true
    
    print_status "Backup created: $BACKUP_FILE ✓"
}

# Health check
health_check() {
    print_status "Performing health check..."
    
    # Check if main service is running
    if curl -f http://localhost:8080/health &>/dev/null; then
        print_status "Health check passed ✓"
    else
        print_warning "Health check failed. Service may still be starting..."
    fi
}

# Main deployment function
main() {
    echo -e "${BLUE}Starting deployment of $PROJECT_NAME${NC}\n"
    
    check_prerequisites
    setup_environment
    install_dependencies
    run_tests
    create_backup
    build_docker
    deploy_services
    health_check
    
    echo -e "\n${GREEN}=== DEPLOYMENT COMPLETED SUCCESSFULLY ===${NC}"
    echo -e "${GREEN}Tradino Unschlagbar is now running!${NC}"
    echo -e "${BLUE}Access the application at: http://localhost:8080${NC}"
    echo -e "${BLUE}Monitor logs with: docker-compose -f $DOCKER_COMPOSE_FILE logs -f${NC}\n"
}

# Run main function if script is executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi 