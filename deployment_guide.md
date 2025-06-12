# 🚀 Production Deployment Guide
## BIRCH Tweet Clustering System

This guide provides comprehensive instructions for deploying the BIRCH clustering system to production with S3 state persistence, monitoring, and high availability.

## 📋 Table of Contents

1. [Prerequisites](#prerequisites)
2. [AWS S3 Setup](#aws-s3-setup)
3. [Local Development Setup](#local-development-setup)
4. [Production Deployment](#production-deployment)
5. [Configuration Management](#configuration-management)
6. [Monitoring & Observability](#monitoring--observability)
7. [Backup & Recovery](#backup--recovery)
8. [Scaling & Performance](#scaling--performance)
9. [Security Best Practices](#security-best-practices)
10. [Troubleshooting](#troubleshooting)

## 🔧 Prerequisites

### System Requirements
- **CPU**: 4+ cores recommended
- **RAM**: 8GB+ (16GB recommended for large datasets)
- **Storage**: 50GB+ SSD
- **Network**: Stable internet connection for S3 access

### Software Requirements
- Docker & Docker Compose
- AWS CLI (configured)
- Git
- Python 3.11+ (for local development)

### AWS Services
- **S3 Bucket** for state persistence
- **IAM User** with S3 permissions
- **CloudWatch** (optional, for enhanced monitoring)

## ☁️ AWS S3 Setup

### 1. Create S3 Bucket

```bash
# Create bucket (replace with your bucket name)
aws s3 mb s3://your-birch-clustering-bucket --region us-east-1

# Enable versioning for backup safety
aws s3api put-bucket-versioning \
    --bucket your-birch-clustering-bucket \
    --versioning-configuration Status=Enabled

# Set lifecycle policy for cost optimization
aws s3api put-bucket-lifecycle-configuration \
    --bucket your-birch-clustering-bucket \
    --lifecycle-configuration file://s3-lifecycle.json
```

### 2. Create IAM Policy

Create `birch-s3-policy.json`:
```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "s3:GetObject",
                "s3:PutObject",
                "s3:DeleteObject",
                "s3:ListBucket"
            ],
            "Resource": [
                "arn:aws:s3:::your-birch-clustering-bucket",
                "arn:aws:s3:::your-birch-clustering-bucket/*"
            ]
        }
    ]
}
```

### 3. Create IAM User

```bash
# Create IAM user
aws iam create-user --user-name birch-clustering-user

# Attach policy
aws iam put-user-policy \
    --user-name birch-clustering-user \
    --policy-name BirchS3Access \
    --policy-document file://birch-s3-policy.json

# Create access keys
aws iam create-access-key --user-name birch-clustering-user
```

## 🛠️ Local Development Setup

### 1. Clone and Setup

```bash
# Clone repository
git clone <your-repo-url>
cd birdie

# Copy environment template
cp .env.production .env

# Edit configuration
nano .env
```

### 2. Configure Environment

Edit `.env` with your actual values:
```bash
# AWS Configuration
S3_BUCKET_NAME=your-birch-clustering-bucket
AWS_ACCESS_KEY_ID=your-access-key
AWS_SECRET_ACCESS_KEY=your-secret-key

# Other configurations...
```

### 3. Test Locally

```bash
# Install dependencies
pip install -r requirements.txt
pip install -r production-requirements.txt

# Run locally
python production_api.py

# Test API
curl http://localhost:8000/health
```

## 🚀 Production Deployment

### Option 1: Docker Compose (Recommended)

```bash
# Build and start services
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f birch-api
```

### Option 2: Kubernetes Deployment

Create `k8s-deployment.yaml`:
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: birch-clustering-api
spec:
  replicas: 2
  selector:
    matchLabels:
      app: birch-clustering-api
  template:
    metadata:
      labels:
        app: birch-clustering-api
    spec:
      containers:
      - name: api
        image: your-registry/birch-clustering:latest
        ports:
        - containerPort: 8000
        env:
        - name: S3_BUCKET_NAME
          valueFrom:
            secretKeyRef:
              name: birch-secrets
              key: s3-bucket-name
        # ... other env vars
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 60
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
```

### Option 3: AWS ECS/Fargate

Create `task-definition.json`:
```json
{
    "family": "birch-clustering",
    "networkMode": "awsvpc",
    "requiresCompatibilities": ["FARGATE"],
    "cpu": "2048",
    "memory": "4096",
    "executionRoleArn": "arn:aws:iam::account:role/ecsTaskExecutionRole",
    "taskRoleArn": "arn:aws:iam::account:role/birch-task-role",
    "containerDefinitions": [
        {
            "name": "birch-api",
            "image": "your-registry/birch-clustering:latest",
            "portMappings": [
                {
                    "containerPort": 8000,
                    "protocol": "tcp"
                }
            ],
            "environment": [
                {
                    "name": "S3_BUCKET_NAME",
                    "value": "your-birch-clustering-bucket"
                }
            ],
            "secrets": [
                {
                    "name": "AWS_ACCESS_KEY_ID",
                    "valueFrom": "arn:aws:secretsmanager:region:account:secret:birch-aws-creds"
                }
            ],
            "healthCheck": {
                "command": ["CMD-SHELL", "curl -f http://localhost:8000/health || exit 1"],
                "interval": 30,
                "timeout": 5,
                "retries": 3,
                "startPeriod": 60
            },
            "logConfiguration": {
                "logDriver": "awslogs",
                "options": {
                    "awslogs-group": "/ecs/birch-clustering",
                    "awslogs-region": "us-east-1",
                    "awslogs-stream-prefix": "ecs"
                }
            }
        }
    ]
}
```

## ⚙️ Configuration Management

### Environment-Specific Configs

```bash
# Development
cp .env.production .env.dev

# Staging
cp .env.production .env.staging

# Production
cp .env.production .env.prod
```

### Configuration Validation

```python
# Add to production_api.py startup
def validate_config():
    required_vars = [
        'S3_BUCKET_NAME',
        'AWS_ACCESS_KEY_ID', 
        'AWS_SECRET_ACCESS_KEY'
    ]
    
    missing = [var for var in required_vars if not os.getenv(var)]
    if missing:
        raise ValueError(f"Missing required environment variables: {missing}")
```

### Dynamic Configuration Updates

```bash
# Update configuration without restart
curl -X POST http://localhost:8000/config/reload \
    -H "Content-Type: application/json" \
    -d '{"similarity_threshold": 0.8}'
```

## 📊 Monitoring & Observability

### 1. Health Checks

```bash
# Basic health check
curl http://localhost:8000/health

# Detailed system stats
curl http://localhost:8000/stats
```

### 2. Prometheus Metrics

Access metrics at: `http://localhost:9090`

Key metrics to monitor:
- `birch_tweets_processed_total`
- `birch_clusters_active`
- `birch_backup_success_total`
- `birch_api_request_duration_seconds`

### 3. Grafana Dashboards

Access Grafana at: `http://localhost:3000`

Import the provided dashboard:
```bash
# Import dashboard
curl -X POST http://admin:admin@localhost:3000/api/dashboards/db \
    -H "Content-Type: application/json" \
    -d @grafana/birch-dashboard.json
```

### 4. Log Aggregation

```bash
# View aggregated logs
docker-compose logs -f fluentd

# Search logs
curl "http://localhost:24224/logs?query=ERROR&limit=100"
```

### 5. Alerting Rules

Create `prometheus-alerts.yml`:
```yaml
groups:
- name: birch-clustering
  rules:
  - alert: BirchAPIDown
    expr: up{job="birch-api"} == 0
    for: 1m
    labels:
      severity: critical
    annotations:
      summary: "BIRCH API is down"
      
  - alert: HighErrorRate
    expr: rate(birch_api_errors_total[5m]) > 0.1
    for: 2m
    labels:
      severity: warning
    annotations:
      summary: "High error rate detected"
      
  - alert: BackupFailed
    expr: increase(birch_backup_failures_total[1h]) > 0
    for: 0m
    labels:
      severity: critical
    annotations:
      summary: "S3 backup failed"
```

## 💾 Backup & Recovery

### 1. Automated Backups

Backups are automatically triggered:
- Every hour (configurable)
- After processing 1000+ tweets
- Before system shutdown
- On manual trigger via API

### 2. Manual Backup

```bash
# Trigger backup via API
curl -X POST http://localhost:8000/backup

# Check backup status
curl http://localhost:8000/stats | jq '.last_backup'
```

### 3. Restore from Backup

```bash
# Restore latest backup
curl -X POST http://localhost:8000/restore

# Restore specific backup (if implemented)
curl -X POST http://localhost:8000/restore \
    -H "Content-Type: application/json" \
    -d '{"timestamp": "20241201_143022"}'
```

### 4. Backup Verification

```bash
# List backups in S3
aws s3 ls s3://your-birch-clustering-bucket/birch-clustering/state/ --recursive

# Verify backup integrity
python scripts/verify_backup.py --timestamp 20241201_143022
```

## 📈 Scaling & Performance

### 1. Horizontal Scaling

```yaml
# docker-compose.yml
services:
  birch-api:
    deploy:
      replicas: 3
    # ... other config
```

### 2. Load Balancing

```nginx
# nginx.conf
upstream birch_backend {
    server birch-api-1:8000;
    server birch-api-2:8000;
    server birch-api-3:8000;
}

server {
    listen 80;
    location / {
        proxy_pass http://birch_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

### 3. Performance Tuning

```bash
# Optimize for high throughput
export BATCH_SIZE=500
export BIRCH_THRESHOLD=0.2
export MAX_TWEET_EMBEDDINGS=100000

# Optimize for memory efficiency
export BATCH_SIZE=50
export CLUSTER_EXPIRY_HOURS=24
export MAX_CLUSTERS_IN_MEMORY=5000
```

### 4. Caching Strategy

```python
# Add Redis caching for embeddings
@cache(ttl=3600)  # 1 hour cache
async def get_tweet_embedding(text: str):
    return embedding_model.encode([text])[0]
```

## 🔒 Security Best Practices

### 1. Environment Security

```bash
# Use secrets management
export AWS_ACCESS_KEY_ID=$(aws secretsmanager get-secret-value \
    --secret-id birch/aws-credentials \
    --query SecretString --output text | jq -r .access_key_id)
```

### 2. Network Security

```yaml
# docker-compose.yml - Internal network
networks:
  birch-internal:
    driver: bridge
    internal: true
```

### 3. API Security

```python
# Add authentication middleware
from fastapi.security import HTTPBearer
from fastapi import Depends

security = HTTPBearer()

@app.post("/tweets")
async def add_tweet(tweet: TweetInput, token: str = Depends(security)):
    # Validate token
    if not validate_token(token):
        raise HTTPException(401, "Invalid token")
```

### 4. Data Encryption

```python
# Encrypt sensitive data before S3 upload
import cryptography.fernet

def encrypt_state_data(data: dict) -> bytes:
    key = os.getenv("ENCRYPTION_KEY").encode()
    f = Fernet(key)
    return f.encrypt(json.dumps(data).encode())
```

## 🔧 Troubleshooting

### Common Issues

#### 1. S3 Connection Issues
```bash
# Test S3 connectivity
aws s3 ls s3://your-birch-clustering-bucket

# Check IAM permissions
aws iam simulate-principal-policy \
    --policy-source-arn arn:aws:iam::account:user/birch-clustering-user \
    --action-names s3:GetObject \
    --resource-arns arn:aws:s3:::your-birch-clustering-bucket/*
```

#### 2. Memory Issues
```bash
# Monitor memory usage
docker stats birch-api

# Adjust memory limits
docker-compose up -d --scale birch-api=1 \
    --memory=8g birch-api
```

#### 3. Clustering Performance
```bash
# Check clustering stats
curl http://localhost:8000/stats | jq '.clustering_stats'

# Adjust BIRCH parameters
export BIRCH_THRESHOLD=0.4  # Increase for fewer, larger clusters
export BIRCH_BRANCHING_FACTOR=100  # Increase for better performance
```

#### 4. Backup Failures
```bash
# Check S3 permissions
aws s3api head-bucket --bucket your-birch-clustering-bucket

# Verify backup logs
docker-compose logs birch-api | grep -i backup

# Manual backup test
curl -X POST http://localhost:8000/backup -v
```

### Debug Mode

```bash
# Enable debug logging
export LOG_LEVEL=debug

# Run with debug
docker-compose -f docker-compose.yml -f docker-compose.debug.yml up
```

### Performance Profiling

```python
# Add profiling endpoint
@app.get("/debug/profile")
async def profile_system():
    import cProfile
    import pstats
    
    profiler = cProfile.Profile()
    profiler.enable()
    
    # Run some operations
    stats = pipeline.get_pipeline_stats()
    
    profiler.disable()
    stats = pstats.Stats(profiler)
    
    return {"profile_data": stats.get_stats_profile()}
```

## 🚀 Deployment Checklist

### Pre-Deployment
- [ ] AWS S3 bucket created and configured
- [ ] IAM user with proper permissions
- [ ] Environment variables configured
- [ ] SSL certificates obtained (if using HTTPS)
- [ ] Monitoring dashboards configured
- [ ] Backup strategy tested

### Deployment
- [ ] Docker images built and pushed
- [ ] Services deployed and healthy
- [ ] Health checks passing
- [ ] Load balancer configured
- [ ] DNS records updated
- [ ] SSL/TLS configured

### Post-Deployment
- [ ] Monitoring alerts configured
- [ ] Backup schedule verified
- [ ] Performance baseline established
- [ ] Documentation updated
- [ ] Team trained on operations

### Production Readiness
- [ ] Load testing completed
- [ ] Disaster recovery tested
- [ ] Security audit passed
- [ ] Compliance requirements met
- [ ] Runbooks created

---

## 📞 Support

For issues and questions:
- Check the [troubleshooting section](#troubleshooting)
- Review application logs
- Monitor system metrics
- Contact the development team

## 📚 Additional Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [AWS S3 Best Practices](https://docs.aws.amazon.com/s3/latest/userguide/best-practices.html)
- [Docker Production Guide](https://docs.docker.com/config/containers/resource_constraints/)
- [Prometheus Monitoring](https://prometheus.io/docs/guides/go-application/)

---

**Happy Clustering! 🎯** 