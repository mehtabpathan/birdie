#!/bin/bash

# 🚀 BIRCH Clustering System - Production Deployment Script
# This script automates the complete deployment process

set -euo pipefail  # Exit on error, undefined vars, pipe failures

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DOCKER_REGISTRY="${DOCKER_REGISTRY:-}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
ENVIRONMENT="${ENVIRONMENT:-production}"
HEALTH_CHECK_TIMEOUT=300  # 5 minutes
BACKUP_BEFORE_DEPLOY="${BACKUP_BEFORE_DEPLOY:-true}"

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Error handling
cleanup() {
    local exit_code=$?
    if [ $exit_code -ne 0 ]; then
        log_error "Deployment failed with exit code $exit_code"
        log_info "Rolling back changes..."
        rollback_deployment
    fi
    exit $exit_code
}

trap cleanup EXIT

# Help function
show_help() {
    cat << EOF
BIRCH Clustering System - Deployment Script

Usage: $0 [OPTIONS]

Options:
    -e, --environment ENV    Deployment environment (dev|staging|production) [default: production]
    -t, --tag TAG           Docker image tag [default: latest]
    -r, --registry URL      Docker registry URL
    -s, --skip-backup       Skip backup before deployment
    -f, --force             Force deployment without confirmation
    -h, --help              Show this help message

Environment Variables:
    DOCKER_REGISTRY         Docker registry URL
    IMAGE_TAG              Docker image tag
    ENVIRONMENT            Deployment environment
    BACKUP_BEFORE_DEPLOY   Whether to backup before deploy (true/false)

Examples:
    $0                                          # Deploy to production with latest tag
    $0 -e staging -t v1.2.3                   # Deploy to staging with specific tag
    $0 -r myregistry.com -t latest -f          # Deploy with custom registry, force mode
EOF
}

# Parse command line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -e|--environment)
                ENVIRONMENT="$2"
                shift 2
                ;;
            -t|--tag)
                IMAGE_TAG="$2"
                shift 2
                ;;
            -r|--registry)
                DOCKER_REGISTRY="$2"
                shift 2
                ;;
            -s|--skip-backup)
                BACKUP_BEFORE_DEPLOY="false"
                shift
                ;;
            -f|--force)
                FORCE_DEPLOY="true"
                shift
                ;;
            -h|--help)
                show_help
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done
}

# Validate environment
validate_environment() {
    log_info "Validating deployment environment..."
    
    # Check required tools
    local required_tools=("docker" "docker-compose" "aws" "curl" "jq")
    for tool in "${required_tools[@]}"; do
        if ! command -v "$tool" &> /dev/null; then
            log_error "Required tool '$tool' is not installed"
            exit 1
        fi
    done
    
    # Validate environment name
    if [[ ! "$ENVIRONMENT" =~ ^(dev|staging|production)$ ]]; then
        log_error "Invalid environment: $ENVIRONMENT. Must be dev, staging, or production"
        exit 1
    fi
    
    # Check environment file
    local env_file="$PROJECT_ROOT/.env.$ENVIRONMENT"
    if [[ ! -f "$env_file" ]]; then
        log_error "Environment file not found: $env_file"
        exit 1
    fi
    
    # Load environment variables
    set -a
    source "$env_file"
    set +a
    
    # Validate required environment variables
    local required_vars=("S3_BUCKET_NAME" "AWS_ACCESS_KEY_ID" "AWS_SECRET_ACCESS_KEY")
    for var in "${required_vars[@]}"; do
        if [[ -z "${!var:-}" ]]; then
            log_error "Required environment variable '$var' is not set"
            exit 1
        fi
    done
    
    log_success "Environment validation completed"
}

# Check AWS connectivity
check_aws_connectivity() {
    log_info "Checking AWS connectivity..."
    
    # Test S3 access
    if ! aws s3 ls "s3://$S3_BUCKET_NAME" &> /dev/null; then
        log_error "Cannot access S3 bucket: $S3_BUCKET_NAME"
        exit 1
    fi
    
    log_success "AWS connectivity verified"
}

# Backup current state
backup_current_state() {
    if [[ "$BACKUP_BEFORE_DEPLOY" == "true" ]]; then
        log_info "Creating backup before deployment..."
        
        # Check if API is running
        if curl -s -f "http://localhost:8000/health" &> /dev/null; then
            # Trigger backup via API
            if curl -s -X POST "http://localhost:8000/backup" | jq -e '.success' &> /dev/null; then
                log_success "Backup completed successfully"
            else
                log_warning "Backup via API failed, continuing with deployment"
            fi
        else
            log_warning "API not accessible, skipping backup"
        fi
    else
        log_info "Skipping backup (disabled)"
    fi
}

# Build Docker images
build_images() {
    log_info "Building Docker images..."
    
    cd "$PROJECT_ROOT"
    
    # Build main application image
    local image_name="birch-clustering"
    if [[ -n "$DOCKER_REGISTRY" ]]; then
        image_name="$DOCKER_REGISTRY/$image_name"
    fi
    
    log_info "Building image: $image_name:$IMAGE_TAG"
    docker build -t "$image_name:$IMAGE_TAG" .
    
    # Tag as latest if not already
    if [[ "$IMAGE_TAG" != "latest" ]]; then
        docker tag "$image_name:$IMAGE_TAG" "$image_name:latest"
    fi
    
    # Push to registry if specified
    if [[ -n "$DOCKER_REGISTRY" ]]; then
        log_info "Pushing image to registry..."
        docker push "$image_name:$IMAGE_TAG"
        if [[ "$IMAGE_TAG" != "latest" ]]; then
            docker push "$image_name:latest"
        fi
    fi
    
    log_success "Docker images built successfully"
}

# Deploy services
deploy_services() {
    log_info "Deploying services..."
    
    cd "$PROJECT_ROOT"
    
    # Update docker-compose with new image tag
    export IMAGE_TAG
    export DOCKER_REGISTRY
    
    # Stop existing services
    log_info "Stopping existing services..."
    docker-compose down --remove-orphans
    
    # Pull latest images
    if [[ -n "$DOCKER_REGISTRY" ]]; then
        log_info "Pulling latest images..."
        docker-compose pull
    fi
    
    # Start services
    log_info "Starting services..."
    docker-compose up -d
    
    log_success "Services deployed successfully"
}

# Health checks
perform_health_checks() {
    log_info "Performing health checks..."
    
    local start_time=$(date +%s)
    local timeout=$HEALTH_CHECK_TIMEOUT
    
    # Wait for API to be ready
    log_info "Waiting for API to be ready..."
    while true; do
        local current_time=$(date +%s)
        local elapsed=$((current_time - start_time))
        
        if [[ $elapsed -gt $timeout ]]; then
            log_error "Health check timeout after ${timeout}s"
            return 1
        fi
        
        if curl -s -f "http://localhost:8000/health" &> /dev/null; then
            log_success "API health check passed"
            break
        fi
        
        log_info "Waiting for API... (${elapsed}s/${timeout}s)"
        sleep 5
    done
    
    # Detailed health check
    log_info "Running detailed health checks..."
    local health_response=$(curl -s "http://localhost:8000/health")
    
    if echo "$health_response" | jq -e '.status == "healthy"' &> /dev/null; then
        log_success "Detailed health check passed"
    else
        log_error "Detailed health check failed: $health_response"
        return 1
    fi
    
    # Check system stats
    log_info "Checking system statistics..."
    local stats_response=$(curl -s "http://localhost:8000/stats")
    
    if echo "$stats_response" | jq -e '.pipeline_status == "running"' &> /dev/null; then
        log_success "System statistics check passed"
    else
        log_warning "System statistics check failed, but continuing deployment"
    fi
    
    # Check all services
    log_info "Checking all services..."
    local failed_services=()
    
    while IFS= read -r service; do
        if ! docker-compose ps "$service" | grep -q "Up"; then
            failed_services+=("$service")
        fi
    done < <(docker-compose config --services)
    
    if [[ ${#failed_services[@]} -eq 0 ]]; then
        log_success "All services are running"
    else
        log_error "Failed services: ${failed_services[*]}"
        return 1
    fi
    
    log_success "All health checks passed"
}

# Rollback deployment
rollback_deployment() {
    log_warning "Rolling back deployment..."
    
    cd "$PROJECT_ROOT"
    
    # Stop current services
    docker-compose down
    
    # Restore from backup if available
    if [[ "$BACKUP_BEFORE_DEPLOY" == "true" ]]; then
        log_info "Attempting to restore from backup..."
        if curl -s -X POST "http://localhost:8000/restore" | jq -e '.success' &> /dev/null; then
            log_success "Restored from backup"
        else
            log_error "Failed to restore from backup"
        fi
    fi
    
    log_warning "Rollback completed"
}

# Post-deployment tasks
post_deployment_tasks() {
    log_info "Running post-deployment tasks..."
    
    # Clean up old Docker images
    log_info "Cleaning up old Docker images..."
    docker image prune -f
    
    # Update monitoring dashboards
    if command -v grafana-cli &> /dev/null; then
        log_info "Updating Grafana dashboards..."
        # Add dashboard update logic here
    fi
    
    # Send deployment notification
    if [[ -n "${SLACK_WEBHOOK_URL:-}" ]]; then
        log_info "Sending deployment notification..."
        curl -X POST -H 'Content-type: application/json' \
            --data "{\"text\":\"🚀 BIRCH Clustering deployed to $ENVIRONMENT (tag: $IMAGE_TAG)\"}" \
            "$SLACK_WEBHOOK_URL" || log_warning "Failed to send Slack notification"
    fi
    
    log_success "Post-deployment tasks completed"
}

# Generate deployment report
generate_deployment_report() {
    log_info "Generating deployment report..."
    
    local report_file="$PROJECT_ROOT/deployment-report-$(date +%Y%m%d-%H%M%S).json"
    
    cat > "$report_file" << EOF
{
    "deployment": {
        "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
        "environment": "$ENVIRONMENT",
        "image_tag": "$IMAGE_TAG",
        "registry": "$DOCKER_REGISTRY",
        "deployed_by": "$(whoami)",
        "git_commit": "$(git rev-parse HEAD 2>/dev/null || echo 'unknown')",
        "git_branch": "$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo 'unknown')"
    },
    "services": $(docker-compose ps --format json | jq -s '.'),
    "system_stats": $(curl -s "http://localhost:8000/stats" || echo '{}'),
    "health_check": $(curl -s "http://localhost:8000/health" || echo '{}')
}
EOF
    
    log_success "Deployment report saved to: $report_file"
}

# Main deployment function
main() {
    log_info "Starting BIRCH Clustering System deployment..."
    log_info "Environment: $ENVIRONMENT"
    log_info "Image Tag: $IMAGE_TAG"
    log_info "Registry: ${DOCKER_REGISTRY:-'local'}"
    
    # Confirmation prompt (unless forced)
    if [[ "${FORCE_DEPLOY:-false}" != "true" ]]; then
        echo
        read -p "Continue with deployment? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            log_info "Deployment cancelled by user"
            exit 0
        fi
    fi
    
    # Deployment steps
    validate_environment
    check_aws_connectivity
    backup_current_state
    build_images
    deploy_services
    perform_health_checks
    post_deployment_tasks
    generate_deployment_report
    
    log_success "🎉 Deployment completed successfully!"
    log_info "API is available at: http://localhost:8000"
    log_info "Grafana dashboard: http://localhost:3000"
    log_info "Prometheus metrics: http://localhost:9090"
    
    # Display service status
    echo
    log_info "Service Status:"
    docker-compose ps
}

# Script entry point
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    parse_args "$@"
    main
fi 