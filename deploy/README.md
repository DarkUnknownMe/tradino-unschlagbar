# 🚀 TRADINO UNSCHLAGBAR - Production Deployment Guide

Vollständige Anleitung für die skalierbare, sichere Production-Deployment-Lösung des TRADINO Advanced AI Trading Systems.

## 📋 Übersicht

Diese Deployment-Lösung bietet:

- **🐳 Containerisierung** mit Multi-Stage Docker Builds
- **☸️ Kubernetes Orchestrierung** mit High Availability
- **🔄 CI/CD Pipeline** mit GitHub Actions
- **📊 Monitoring & Alerting** mit Prometheus/Grafana
- **🔒 Security Best Practices** mit Secret Management
- **💾 Automatische Backups** und Disaster Recovery
- **🌐 SSL/TLS Termination** mit automatischen Zertifikaten

## 🏗️ Architektur

```
┌─────────────────────┐    ┌─────────────────────┐    ┌─────────────────────┐
│     Internet        │────│   Load Balancer     │────│    Nginx Proxy     │
│                     │    │   (AWS ALB/NLB)     │    │   (Reverse Proxy)  │
└─────────────────────┘    └─────────────────────┘    └─────────────────────┘
                                      │                           │
                           ┌──────────▼──────────┐     ┌─────────▼─────────┐
                           │   Kubernetes        │     │   SSL/TLS         │
                           │   Ingress          │     │   Termination     │
                           └─────────────────────┘     └───────────────────┘
                                      │
                    ┌─────────────────▼─────────────────┐
                    │         TRADINO Application       │
                    │     (Multiple Replicas)          │
                    └─────────────────┬─────────────────┘
                                      │
        ┌─────────────────────────────┼─────────────────────────────┐
        │                             │                             │
┌───────▼───────┐           ┌─────────▼─────────┐           ┌─────▼─────┐
│   PostgreSQL  │           │      Redis        │           │ Monitoring│
│   Database    │           │      Cache        │           │ Stack     │
└───────────────┘           └───────────────────┘           └───────────┘
```

## 🛠️ Voraussetzungen

### System Requirements

- **Kubernetes Cluster** (v1.24+)
- **Docker** (v20.10+)
- **kubectl** (v1.24+)
- **Helm** (v3.0+) - Optional
- **Git** (v2.0+)

### Infrastructure Requirements

#### Development Environment
- **CPU**: 2 vCPU
- **Memory**: 4 GB RAM
- **Storage**: 20 GB SSD
- **Nodes**: 1

#### Staging Environment
- **CPU**: 4 vCPU
- **Memory**: 8 GB RAM
- **Storage**: 50 GB SSD
- **Nodes**: 2

#### Production Environment
- **CPU**: 8+ vCPU
- **Memory**: 16+ GB RAM
- **Storage**: 100+ GB SSD
- **Nodes**: 3+ (Multi-AZ)

### Cloud Provider Setup

#### AWS EKS (Recommended)
```bash
# Install eksctl
curl --silent --location "https://github.com/weaveworks/eksctl/releases/latest/download/eksctl_$(uname -s)_amd64.tar.gz" | tar xz -C /tmp
sudo mv /tmp/eksctl /usr/local/bin

# Create EKS cluster
eksctl create cluster --name tradino-prod --version 1.28 --region eu-central-1 --nodegroup-name tradino-nodes --node-type m5.large --nodes 3 --nodes-min 1 --nodes-max 10 --managed
```

#### Azure AKS
```bash
# Create resource group
az group create --name tradino-rg --location eastus

# Create AKS cluster
az aks create --resource-group tradino-rg --name tradino-aks --node-count 3 --enable-addons monitoring --generate-ssh-keys
```

#### Google GKE
```bash
# Create GKE cluster
gcloud container clusters create tradino-gke --num-nodes=3 --zone=europe-west1-a --machine-type=n1-standard-2
```

## 📦 Installation & Setup

### 1. Repository Setup

```bash
# Clone repository
git clone https://github.com/your-org/tradino-unschlagbar.git
cd tradino-unschlagbar

# Make deployment script executable
chmod +x deploy/scripts/deploy.sh
```

### 2. Environment Configuration

Erstelle Environment-spezifische Konfigurationsdateien:

```bash
# Development
cp deploy/config/development.env.example deploy/config/development.env

# Staging
cp deploy/config/staging.env.example deploy/config/staging.env

# Production
cp deploy/config/production.env.example deploy/config/production.env
```

Bearbeite die Konfigurationsdateien mit deinen spezifischen Werten:

```bash
# deploy/config/production.env
DOMAIN=nobelbrett.com
POSTGRES_DB=tradino_prod
REDIS_PASSWORD=your-secure-redis-password
BITGET_API_KEY=your-bitget-api-key
BITGET_SECRET=your-bitget-secret
TELEGRAM_BOT_TOKEN=your-telegram-bot-token
```

### 3. Secrets Management

#### Kubernetes Secrets
```bash
# Create secrets for production
kubectl create secret generic tradino-secrets \
  --from-literal=postgres-password=your-db-password \
  --from-literal=redis-password=your-redis-password \
  --from-literal=bitget-api-key=your-api-key \
  --from-literal=bitget-secret=your-api-secret \
  --from-literal=telegram-bot-token=your-bot-token \
  --namespace=tradino-production
```

#### External Secret Management (Recommended)
Für Production wird empfohlen, externe Secret Manager zu verwenden:

- **AWS Secrets Manager**
- **Azure Key Vault**
- **Google Secret Manager**
- **HashiCorp Vault**

### 4. SSL/TLS Zertifikate

#### Let's Encrypt (Automatisch)
```bash
# Install cert-manager
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.13.0/cert-manager.yaml

# Apply ClusterIssuer
kubectl apply -f deploy/ssl/cluster-issuer.yaml
```

#### Eigene Zertifikate
```bash
# Erstelle TLS Secret
kubectl create secret tls tradino-tls-secret \
  --cert=path/to/cert.pem \
  --key=path/to/key.pem \
  --namespace=tradino-production
```

## 🚀 Deployment

### Quick Start

```bash
# Voraussetzungen prüfen
./deploy/scripts/deploy.sh --check

# Development Deployment
./deploy/scripts/deploy.sh development

# Staging Deployment mit Backup
./deploy/scripts/deploy.sh staging --backup

# Production Deployment
./deploy/scripts/deploy.sh production --check --backup
```

### Manual Deployment Steps

#### 1. Namespace und Basis-Setup
```bash
kubectl apply -f deploy/kubernetes/namespace.yaml
kubectl apply -f deploy/kubernetes/configmap.yaml
```

#### 2. Persistent Volumes
```bash
kubectl apply -f deploy/kubernetes/pvc.yaml
```

#### 3. Secrets
```bash
kubectl apply -f deploy/secrets/production-secrets.yaml
```

#### 4. Application Deployment
```bash
kubectl apply -f deploy/kubernetes/deployment.yaml
kubectl apply -f deploy/kubernetes/services.yaml
```

#### 5. Ingress & Load Balancer
```bash
kubectl apply -f deploy/kubernetes/ingress.yaml
```

#### 6. Monitoring Stack
```bash
kubectl apply -f deploy/monitoring/
```

### Docker Compose Deployment (Entwicklung)

```bash
# Development
docker-compose up -d

# Production
docker-compose -f docker-compose.prod.yml up -d
```

## 📊 Monitoring & Observability

### Zugiff auf Monitoring Tools

- **Grafana**: https://monitoring.nobelbrett.com/grafana
- **Prometheus**: https://monitoring.nobelbrett.com/prometheus
- **Kibana**: https://monitoring.nobelbrett.com/kibana (falls ELK Stack)

### Wichtige Dashboards

1. **Trading Performance Dashboard**
   - PnL Tracking
   - Trade Volume
   - Win/Loss Ratio
   - Risk Metrics

2. **System Health Dashboard**
   - Application Metrics
   - Database Performance
   - Cache Hit Rates
   - API Response Times

3. **Infrastructure Dashboard**
   - Kubernetes Cluster Status
   - Node Resources
   - Network Traffic
   - Storage Usage

### Alerting Setup

Alerts werden automatisch an folgende Kanäle gesendet:

- **Slack**: #tradino-alerts
- **Email**: admin@nobelbrett.com
- **PagerDuty**: Für kritische Alerts
- **Telegram**: Bot Notifications

## 🔒 Sicherheit

### Security Best Practices

1. **Container Security**
   - Non-root User ausführung
   - Read-only Root Filesystem
   - Security Context Policies
   - Regular Image Scanning

2. **Network Security**
   - Network Policies
   - Ingress Controller mit WAF
   - mTLS zwischen Services
   - VPC/Subnet Isolation

3. **Secret Management**
   - Externe Secret Stores
   - Automatic Secret Rotation
   - Encrypted Secrets at Rest
   - RBAC für Secret Access

4. **Access Control**
   - RBAC Policies
   - Service Accounts
   - Pod Security Standards
   - Audit Logging

### Security Scans

```bash
# Container Vulnerability Scan
trivy image tradino:latest

# Kubernetes Security Scan
kube-score score deploy/kubernetes/

# SAST Code Analysis
bandit -r tradino_unschlagbar/
```

## 💾 Backup & Recovery

### Automatische Backups

Backups werden automatisch erstellt:

- **Database**: Täglich um 02:00 UTC
- **Application Data**: Täglich um 03:00 UTC
- **Configuration**: Bei jedem Deployment
- **Retention**: 30 Tage

### Manual Backup

```bash
# Database Backup
kubectl exec -n tradino-production deployment/tradino-postgres -- pg_dump -U postgres tradino_db > backup.sql

# Application Data Backup
kubectl exec -n tradino-production deployment/tradino-app -- tar -czf /tmp/app-backup.tar.gz /app/data
```

### Disaster Recovery

```bash
# Database Restore
kubectl exec -i -n tradino-production deployment/tradino-postgres -- psql -U postgres tradino_db < backup.sql

# Application Data Restore
kubectl cp app-backup.tar.gz tradino-production/tradino-app-pod:/tmp/
kubectl exec -n tradino-production deployment/tradino-app -- tar -xzf /tmp/app-backup.tar.gz -C /
```

## 🔧 CI/CD Pipeline

### GitHub Actions Workflow

Die CI/CD Pipeline umfasst:

1. **Code Quality Checks**
   - Linting (flake8, black)
   - Type Checking (mypy)
   - Security Scanning (bandit)

2. **Testing**
   - Unit Tests
   - Integration Tests
   - E2E Tests

3. **Build & Push**
   - Docker Image Build
   - Multi-platform Support
   - Image Security Scan

4. **Deployment**
   - Staging Deployment (develop branch)
   - Production Deployment (tags)
   - Health Checks
   - Notifications

### Manual Trigger

```bash
# Trigger CI/CD Pipeline
gh workflow run "TRADINO CI/CD Pipeline" --ref main
```

## 🐛 Troubleshooting

### Häufige Probleme

#### 1. Pod startet nicht
```bash
# Check Pod Status
kubectl get pods -n tradino-production

# Check Events
kubectl describe pod tradino-app-xxx -n tradino-production

# Check Logs
kubectl logs tradino-app-xxx -n tradino-production
```

#### 2. Database Connection Error
```bash
# Check Database Pod
kubectl get pods -l component=postgres -n tradino-production

# Test Connection
kubectl exec -it deployment/tradino-postgres -n tradino-production -- psql -U postgres -d tradino_db
```

#### 3. External API nicht erreichbar
```bash
# Check Network Policies
kubectl get networkpolicies -n tradino-production

# Test External Connectivity
kubectl exec -it deployment/tradino-app -n tradino-production -- curl -v https://api.bitget.com
```

### Logs und Debugging

```bash
# Application Logs
kubectl logs -f deployment/tradino-app -n tradino-production

# All Components Logs
kubectl logs -f -l app=tradino -n tradino-production --all-containers=true

# Previous Container Logs
kubectl logs deployment/tradino-app -n tradino-production --previous
```

### Performance Tuning

```bash
# Check Resource Usage
kubectl top pods -n tradino-production
kubectl top nodes

# Scale Application
kubectl scale deployment tradino-app --replicas=5 -n tradino-production

# Update Resource Limits
kubectl patch deployment tradino-app -n tradino-production -p '{"spec":{"template":{"spec":{"containers":[{"name":"tradino-app","resources":{"limits":{"cpu":"4","memory":"8Gi"}}}]}}}}'
```

## 📈 Scaling

### Horizontal Pod Autoscaler

```bash
# Setup HPA
kubectl autoscale deployment tradino-app --cpu-percent=70 --min=2 --max=10 -n tradino-production

# Check HPA Status
kubectl get hpa -n tradino-production
```

### Vertical Pod Autoscaler

```bash
# Install VPA
kubectl apply -f https://github.com/kubernetes/autoscaler/releases/latest/download/vpa-v1.0.0.yaml

# Apply VPA
kubectl apply -f deploy/kubernetes/vpa.yaml
```

### Cluster Autoscaler

```bash
# AWS EKS
kubectl apply -f deploy/kubernetes/cluster-autoscaler-aws.yaml

# Configure Node Groups
eksctl create nodegroup --cluster=tradino-prod --name=tradino-spot --spot --instance-types=m5.large,m5.xlarge --nodes-min=0 --nodes-max=10
```

## 🔄 Updates & Maintenance

### Rolling Updates

```bash
# Update Image Version
kubectl set image deployment/tradino-app tradino-app=tradino:v2.0.0 -n tradino-production

# Check Rollout Status
kubectl rollout status deployment/tradino-app -n tradino-production

# Rollback if needed
kubectl rollout undo deployment/tradino-app -n tradino-production
```

### Maintenance Mode

```bash
# Enable Maintenance Mode
kubectl patch deployment tradino-app -n tradino-production -p '{"spec":{"template":{"metadata":{"annotations":{"maintenance.tradino.io/enabled":"true"}}}}}'

# Scale down to 0
kubectl scale deployment tradino-app --replicas=0 -n tradino-production
```

## 📞 Support & Contacts

### Technical Support
- **Email**: support@nobelbrett.com
- **Slack**: #tradino-support
- **Emergency**: +49-xxx-xxx-xxxx

### Documentation
- **Technical Docs**: https://docs.nobelbrett.com
- **API Reference**: https://api.nobelbrett.com/docs
- **Status Page**: https://status.nobelbrett.com

### Contributing
- **GitHub**: https://github.com/your-org/tradino-unschlagbar
- **Issues**: https://github.com/your-org/tradino-unschlagbar/issues
- **Pull Requests**: Welcome!

---

**© 2024 TRADINO UNSCHLAGBAR - Advanced AI Trading System** 