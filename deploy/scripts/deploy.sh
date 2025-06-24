#!/bin/bash

# =========================================================================
# TRADINO UNSCHLAGBAR - Deployment Script
# Automated deployment script for production and staging environments
# =========================================================================

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPLOY_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_ROOT="$(cd "${DEPLOY_DIR}/.." && pwd)"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

# Usage function
usage() {
    cat << EOF
Usage: $0 [OPTIONS] ENVIRONMENT

TRADINO Deployment Script

ENVIRONMENTS:
    development     Deploy to development environment
    staging         Deploy to staging environment
    production      Deploy to production environment

OPTIONS:
    -h, --help      Show this help message
    -v, --version   Show version information
    -c, --check     Check deployment prerequisites
    -b, --backup    Create backup before deployment
    -s, --skip-tests Skip pre-deployment tests
    --dry-run       Show what would be deployed without executing

EXAMPLES:
    $0 development
    $0 staging --backup
    $0 production --check --backup
    $0 staging --dry-run

EOF
}

# Version function
version() {
    echo "TRADINO Deployment Script v1.0.0"
}

# Check prerequisites
check_prerequisites() {
    log "Checking deployment prerequisites..."
    
    local requirements=(
        "docker:Docker"
        "kubectl:Kubernetes CLI"
        "helm:Helm package manager"
        "git:Git version control"
    )
    
    local missing=()
    
    for req in "${requirements[@]}"; do
        cmd="${req%%:*}"
        desc="${req##*:}"
        
        if ! command -v "$cmd" &> /dev/null; then
            missing+=("$desc ($cmd)")
        fi
    done
    
    if [ ${#missing[@]} -gt 0 ]; then
        error "Missing required tools:"
        printf '%s\n' "${missing[@]}"
        return 1
    fi
    
    success "All prerequisites satisfied"
}

# Validate environment
validate_environment() {
    local env="$1"
    
    case "$env" in
        development|staging|production)
            log "Deploying to $env environment"
            ;;
        *)
            error "Invalid environment: $env"
            usage
            exit 1
            ;;
    esac
}

# Load environment configuration
load_config() {
    local env="$1"
    
    # Set environment-specific variables
    case "$env" in
        development)
            export NAMESPACE="tradino-dev"
            export DOMAIN="dev.nobelbrett.com"
            export REPLICAS="1"
            export RESOURCES_LIMITS_CPU="1"
            export RESOURCES_LIMITS_MEMORY="2Gi"
            ;;
        staging)
            export NAMESPACE="tradino-staging"
            export DOMAIN="staging.nobelbrett.com"
            export REPLICAS="2"
            export RESOURCES_LIMITS_CPU="2"
            export RESOURCES_LIMITS_MEMORY="4Gi"
            ;;
        production)
            export NAMESPACE="tradino-production"
            export DOMAIN="nobelbrett.com"
            export REPLICAS="3"
            export RESOURCES_LIMITS_CPU="4"
            export RESOURCES_LIMITS_MEMORY="8Gi"
            ;;
    esac
    
    # Load environment-specific configuration file
    local config_file="${DEPLOY_DIR}/config/${env}.env"
    if [ -f "$config_file" ]; then
        log "Loading configuration from $config_file"
        source "$config_file"
    fi
}

# Create namespace
create_namespace() {
    local env="$1"
    
    log "Creating namespace $NAMESPACE..."
    
    # Apply namespace configuration
    envsubst < "${DEPLOY_DIR}/kubernetes/namespace.yaml" | kubectl apply -f -
    
    success "Namespace $NAMESPACE created/updated"
}

# Deploy secrets
deploy_secrets() {
    local env="$1"
    
    log "Deploying secrets for $env environment..."
    
    # Check if secrets file exists
    local secrets_file="${DEPLOY_DIR}/secrets/${env}-secrets.yaml"
    if [ ! -f "$secrets_file" ]; then
        warning "Secrets file not found: $secrets_file"
        warning "Please create secrets manually or use secret management tool"
        return 0
    fi
    
    kubectl apply -f "$secrets_file" -n "$NAMESPACE"
    success "Secrets deployed"
}

# Deploy configmaps
deploy_configmaps() {
    log "Deploying configuration..."
    
    envsubst < "${DEPLOY_DIR}/kubernetes/configmap.yaml" | kubectl apply -f - -n "$NAMESPACE"
    success "ConfigMaps deployed"
}

# Deploy application
deploy_application() {
    local env="$1"
    
    log "Deploying TRADINO application..."
    
    # Deploy persistent volumes
    if [ -f "${DEPLOY_DIR}/kubernetes/pvc.yaml" ]; then
        envsubst < "${DEPLOY_DIR}/kubernetes/pvc.yaml" | kubectl apply -f - -n "$NAMESPACE"
    fi
    
    # Deploy application
    envsubst < "${DEPLOY_DIR}/kubernetes/deployment.yaml" | kubectl apply -f - -n "$NAMESPACE"
    
    # Deploy services
    envsubst < "${DEPLOY_DIR}/kubernetes/services.yaml" | kubectl apply -f - -n "$NAMESPACE"
    
    # Wait for deployment to be ready
    log "Waiting for deployment to be ready..."
    kubectl rollout status deployment/tradino-app -n "$NAMESPACE" --timeout=600s
    
    success "Application deployed successfully"
}

# Deploy monitoring
deploy_monitoring() {
    local env="$1"
    
    if [ "$env" = "development" ]; then
        log "Skipping monitoring deployment for development environment"
        return 0
    fi
    
    log "Deploying monitoring stack..."
    
    # Deploy Prometheus
    if [ -f "${DEPLOY_DIR}/monitoring/prometheus-deployment.yaml" ]; then
        envsubst < "${DEPLOY_DIR}/monitoring/prometheus-deployment.yaml" | kubectl apply -f - -n "$NAMESPACE"
    fi
    
    # Deploy Grafana
    if [ -f "${DEPLOY_DIR}/monitoring/grafana-deployment.yaml" ]; then
        envsubst < "${DEPLOY_DIR}/monitoring/grafana-deployment.yaml" | kubectl apply -f - -n "$NAMESPACE"
    fi
    
    success "Monitoring stack deployed"
}

# Health checks
health_check() {
    log "Performing health checks..."
    
    # Wait for pods to be ready
    kubectl wait --for=condition=Ready pod -l app=tradino -n "$NAMESPACE" --timeout=300s
    
    # Get service endpoint
    local service_ip
    service_ip=$(kubectl get svc tradino-app-service -n "$NAMESPACE" -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || echo "")
    
    if [ -z "$service_ip" ]; then
        service_ip=$(kubectl get svc tradino-app-service -n "$NAMESPACE" -o jsonpath='{.spec.clusterIP}')
    fi
    
    # Test health endpoint
    log "Testing health endpoint at $service_ip:8000/health"
    
    local retries=10
    local count=0
    
    while [ $count -lt $retries ]; do
        if kubectl exec -n "$NAMESPACE" deployment/tradino-app -- curl -f http://localhost:8000/health; then
            success "Health check passed"
            return 0
        fi
        
        count=$((count + 1))
        log "Health check attempt $count/$retries failed, retrying in 10 seconds..."
        sleep 10
    done
    
    error "Health check failed after $retries attempts"
    return 1
}

# Create backup
create_backup() {
    local env="$1"
    
    log "Creating backup for $env environment..."
    
    local backup_name="tradino-backup-$(date +%Y%m%d-%H%M%S)"
    
    # Database backup
    kubectl exec -n "$NAMESPACE" deployment/tradino-postgres -- pg_dump -U postgres tradino_db > "${backup_name}.sql"
    
    # Application data backup
    kubectl exec -n "$NAMESPACE" deployment/tradino-app -- tar -czf "/tmp/${backup_name}.tar.gz" /app/data
    kubectl cp "$NAMESPACE/$(kubectl get pod -n "$NAMESPACE" -l app=tradino,component=app -o jsonpath='{.items[0].metadata.name}'):/tmp/${backup_name}.tar.gz" "${backup_name}.tar.gz"
    
    success "Backup created: $backup_name"
}

# Run tests
run_tests() {
    log "Running pre-deployment tests..."
    
    # Validate Kubernetes manifests
    kubectl apply --dry-run=client -f "${DEPLOY_DIR}/kubernetes/" -n "$NAMESPACE" > /dev/null
    
    # Check Docker image availability
    if [ -n "${DOCKER_IMAGE:-}" ]; then
        docker pull "$DOCKER_IMAGE" > /dev/null
    fi
    
    success "Pre-deployment tests passed"
}

# Main deployment function
main() {
    local environment=""
    local check_only=false
    local create_backup_flag=false
    local skip_tests=false
    local dry_run=false
    
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                usage
                exit 0
                ;;
            -v|--version)
                version
                exit 0
                ;;
            -c|--check)
                check_only=true
                shift
                ;;
            -b|--backup)
                create_backup_flag=true
                shift
                ;;
            -s|--skip-tests)
                skip_tests=true
                shift
                ;;
            --dry-run)
                dry_run=true
                shift
                ;;
            -*)
                error "Unknown option $1"
                usage
                exit 1
                ;;
            *)
                if [ -z "$environment" ]; then
                    environment="$1"
                else
                    error "Multiple environments specified"
                    usage
                    exit 1
                fi
                shift
                ;;
        esac
    done
    
    # Validate arguments
    if [ -z "$environment" ] && [ "$check_only" = false ]; then
        error "Environment is required"
        usage
        exit 1
    fi
    
    # Check prerequisites
    check_prerequisites || exit 1
    
    if [ "$check_only" = true ]; then
        success "Prerequisites check completed"
        exit 0
    fi
    
    # Validate environment
    validate_environment "$environment"
    
    # Load configuration
    load_config "$environment"
    
    if [ "$dry_run" = true ]; then
        log "DRY RUN: Would deploy to $environment with the following configuration:"
        echo "  Namespace: $NAMESPACE"
        echo "  Domain: $DOMAIN"
        echo "  Replicas: $REPLICAS"
        echo "  CPU Limit: $RESOURCES_LIMITS_CPU"
        echo "  Memory Limit: $RESOURCES_LIMITS_MEMORY"
        exit 0
    fi
    
    log "Starting deployment to $environment environment..."
    
    # Create backup if requested
    if [ "$create_backup_flag" = true ]; then
        create_backup "$environment"
    fi
    
    # Run tests unless skipped
    if [ "$skip_tests" = false ]; then
        run_tests
    fi
    
    # Deploy components
    create_namespace "$environment"
    deploy_secrets "$environment"
    deploy_configmaps
    deploy_application "$environment"
    deploy_monitoring "$environment"
    
    # Health check
    health_check
    
    success "Deployment to $environment completed successfully!"
    log "Application URL: https://$DOMAIN"
    
    if [ "$environment" != "development" ]; then
        log "Monitoring URL: https://monitoring.$DOMAIN"
    fi
}

# Run main function
main "$@" 