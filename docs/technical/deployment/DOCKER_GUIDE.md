# Docker Deployment Guide

**Version:** 0.6.6  
**Last Updated:** October 28, 2025

This guide covers Docker containerization for SynFinance, including building images, running containers, and deploying to production.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Prerequisites](#prerequisites)
3. [Docker Images](#docker-images)
4. [Building Images](#building-images)
5. [Running Containers](#running-containers)
6. [Docker Compose](#docker-compose)
7. [Configuration](#configuration)
8. [Production Deployment](#production-deployment)
9. [Troubleshooting](#troubleshooting)
10. [Best Practices](#best-practices)

---

## Quick Start

### Pull and Run (Recommended for Users)

```bash
# Pull latest image from registry
docker pull ghcr.io/synfinance/synfinance:latest

# Run container
docker run -d \
  --name synfinance \
  -p 8000:8000 \
  ghcr.io/synfinance/synfinance:latest

# Check health
curl http://localhost:8000/health

# View API docs
open http://localhost:8000/docs
```

### Build and Run (For Development)

```bash
# Build image
docker build -t synfinance:latest .

# Run container
docker run -d --name synfinance -p 8000:8000 synfinance:latest

# View logs
docker logs -f synfinance
```

---

## Prerequisites

### Required Software

- **Docker Desktop** (Windows/Mac) or **Docker Engine** (Linux)
  - Version: 20.10+
  - Download: https://www.docker.com/products/docker-desktop

- **Docker Compose** (optional, for multi-container setups)
  - Version: 2.0+
  - Included with Docker Desktop

### System Requirements

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| CPU | 2 cores | 4+ cores |
| RAM | 2 GB | 4+ GB |
| Disk Space | 2 GB | 5+ GB |
| OS | Windows 10+, macOS 10.15+, Linux kernel 3.10+ | Latest |

### Verify Installation

```bash
# Check Docker version
docker --version
# Output: Docker version 24.0.x

# Check Docker Compose version
docker-compose --version
# Output: Docker Compose version 2.x.x

# Test Docker
docker run hello-world
```

---

## Docker Images

SynFinance provides two Docker images:

### 1. Production Image (`Dockerfile`)

**Purpose:** Optimized for production deployments

**Features:**
- Multi-stage build for minimal size
- Python 3.12-slim base image
- Non-root user (`synfinance`)
- Health check enabled
- Security hardened
- Size: ~400-500 MB

**When to use:**
- Production deployments
- Cloud hosting (AWS, Azure, GCP)
- Kubernetes/container orchestration
- Performance-critical environments

### 2. Development Image (`Dockerfile.dev`)

**Purpose:** Feature-rich environment for development

**Features:**
- Python 3.12 full image
- Jupyter Lab included
- Development tools (black, flake8, pylint)
- Debugging tools (ipdb, line_profiler)
- Volume mounts for live code editing
- Size: ~1-1.5 GB

**When to use:**
- Local development
- Data exploration with Jupyter
- Debugging and profiling
- Testing new features

---

## Building Images

### Production Image

```bash
# Basic build
docker build -t synfinance:latest .

# Build with specific version tag
docker build -t synfinance:0.6.6 .

# Build for multi-platform (ARM64 + AMD64)
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  -t synfinance:latest \
  --push .

# Build with build arguments
docker build \
  --build-arg DEBIAN_FRONTEND=noninteractive \
  -t synfinance:latest .
```

### Development Image

```bash
# Build development image
docker build -f Dockerfile.dev -t synfinance:dev .

# With custom requirements
docker build \
  -f Dockerfile.dev \
  --build-arg REQUIREMENTS_FILE=requirements-dev.txt \
  -t synfinance:dev .
```

### Build Options

| Flag | Description | Example |
|------|-------------|---------|
| `-t, --tag` | Name and tag | `-t synfinance:v1.0` |
| `-f, --file` | Dockerfile path | `-f Dockerfile.dev` |
| `--no-cache` | Build without cache | `--no-cache` |
| `--build-arg` | Set build arguments | `--build-arg VERSION=1.0` |
| `--platform` | Target platform | `--platform linux/amd64` |

---

## Running Containers

### Basic Container

```bash
# Run in foreground
docker run --rm -it -p 8000:8000 synfinance:latest

# Run in background (detached)
docker run -d --name synfinance -p 8000:8000 synfinance:latest

# Run with auto-restart
docker run -d \
  --name synfinance \
  --restart unless-stopped \
  -p 8000:8000 \
  synfinance:latest
```

### With Volume Mounts

```bash
# Mount data and output directories
docker run -d \
  --name synfinance \
  -p 8000:8000 \
  -v $(pwd)/data:/app/data:ro \
  -v $(pwd)/output:/app/output \
  -v $(pwd)/logs:/app/logs \
  synfinance:latest
```

### With Environment Variables

```bash
# Set environment variables
docker run -d \
  --name synfinance \
  -p 8000:8000 \
  -e SYNFINANCE_ENV=production \
  -e LOG_LEVEL=INFO \
  -e PYTHONUNBUFFERED=1 \
  synfinance:latest

# From environment file
docker run -d \
  --name synfinance \
  -p 8000:8000 \
  --env-file .env.production \
  synfinance:latest
```

### Resource Limits

```bash
# Limit CPU and memory
docker run -d \
  --name synfinance \
  -p 8000:8000 \
  --cpus=2.0 \
  --memory=4g \
  --memory-swap=4g \
  synfinance:latest
```

### Development Container

```bash
# Run Jupyter Lab (development image)
docker run -d \
  --name synfinance-dev \
  -p 8888:8888 \
  -p 8000:8000 \
  -v $(pwd):/app \
  synfinance:dev

# Access Jupyter at http://localhost:8888
```

---

## Docker Compose

### Basic Usage

```bash
# Start all services
docker-compose up -d

# Start specific service
docker-compose up api

# Stop all services
docker-compose down

# View logs
docker-compose logs -f api

# Restart services
docker-compose restart api
```

### Service Profiles

SynFinance supports optional service profiles:

```bash
# Start with monitoring (Prometheus + Grafana)
docker-compose --profile monitoring up -d

# Start with database
docker-compose --profile database up -d

# Start with caching
docker-compose --profile cache up -d

# Start everything
docker-compose --profile monitoring --profile database --profile cache up -d
```

### Available Services

| Service | Port | Profile | Description |
|---------|------|---------|-------------|
| api | 8000 | default | Main SynFinance API |
| prometheus | 9090 | monitoring | Metrics collection |
| grafana | 3000 | monitoring | Dashboards |
| postgres | 5432 | database | PostgreSQL database |
| redis | 6379 | cache | Redis cache |

### Custom Configuration

Create `docker-compose.override.yml` for local customization:

```yaml
version: '3.8'

services:
  api:
    environment:
      - LOG_LEVEL=DEBUG
    volumes:
      - ./custom-data:/app/data:ro
```

---

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SYNFINANCE_ENV` | `production` | Environment (production/development) |
| `PYTHONUNBUFFERED` | `1` | Disable Python output buffering |
| `LOG_LEVEL` | `INFO` | Logging level (DEBUG/INFO/WARNING/ERROR) |
| `API_HOST` | `0.0.0.0` | API bind address |
| `API_PORT` | `8000` | API port |
| `WORKERS` | `4` | Number of Uvicorn workers |

### Volume Mounts

Recommended volume mounts:

```bash
-v ./data:/app/data:ro        # Input data (read-only)
-v ./output:/app/output        # Generated outputs
-v ./logs:/app/logs            # Application logs
-v ./config:/app/config:ro     # Configuration files (read-only)
```

### Port Mappings

```bash
-p 8000:8000   # API (required)
-p 8888:8888   # Jupyter Lab (development only)
-p 9090:9090   # Prometheus (monitoring)
-p 3000:3000   # Grafana (monitoring)
```

---

## Production Deployment

### Using Deployment Script

```bash
# Deploy to production
chmod +x scripts/deploy.sh
./scripts/deploy.sh

# Check deployment health
chmod +x scripts/health_check.sh
./scripts/health_check.sh

# Rollback if needed
chmod +x scripts/rollback.sh
./scripts/rollback.sh
```

### Manual Deployment Steps

```bash
# 1. Pull latest image
docker pull ghcr.io/synfinance/synfinance:latest

# 2. Stop old container (if exists)
docker stop synfinance-prod || true
docker rm synfinance-prod || true

# 3. Start new container
docker run -d \
  --name synfinance-prod \
  --restart unless-stopped \
  -p 8000:8000 \
  -v $(pwd)/data:/app/data:ro \
  -v $(pwd)/output:/app/output \
  -v $(pwd)/logs:/app/logs \
  -e SYNFINANCE_ENV=production \
  -e LOG_LEVEL=INFO \
  --cpus=2.0 \
  --memory=4g \
  ghcr.io/synfinance/synfinance:latest

# 4. Wait for health check
sleep 10
curl http://localhost:8000/health

# 5. Verify deployment
curl http://localhost:8000/docs
```

### Cloud Platforms

#### AWS ECS

```bash
# Push to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <account-id>.dkr.ecr.us-east-1.amazonaws.com
docker tag synfinance:latest <account-id>.dkr.ecr.us-east-1.amazonaws.com/synfinance:latest
docker push <account-id>.dkr.ecr.us-east-1.amazonaws.com/synfinance:latest
```

#### Google Cloud Run

```bash
# Deploy to Cloud Run
gcloud run deploy synfinance \
  --image ghcr.io/synfinance/synfinance:latest \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

#### Azure Container Instances

```bash
# Deploy to Azure
az container create \
  --resource-group synfinance-rg \
  --name synfinance \
  --image ghcr.io/synfinance/synfinance:latest \
  --cpu 2 \
  --memory 4 \
  --ports 8000
```

---

## Troubleshooting

### Common Issues

#### Container Won't Start

```bash
# Check logs
docker logs synfinance

# Check container status
docker ps -a | grep synfinance

# Inspect container
docker inspect synfinance

# Try running interactively
docker run --rm -it synfinance:latest /bin/bash
```

#### Health Check Failing

```bash
# Check health status
docker inspect --format='{{.State.Health.Status}}' synfinance

# View health check logs
docker inspect --format='{{range .State.Health.Log}}{{.Output}}{{end}}' synfinance

# Manual health check
docker exec synfinance curl -f http://localhost:8000/health
```

#### Port Already in Use

```bash
# Find process using port
lsof -i :8000  # Linux/Mac
netstat -ano | findstr :8000  # Windows

# Use different port
docker run -p 8080:8000 synfinance:latest
```

#### Permission Denied

```bash
# Linux: Add user to docker group
sudo usermod -aG docker $USER
newgrp docker

# Or run with sudo
sudo docker run ...
```

#### Image Too Large

```bash
# Check image size
docker images synfinance

# Remove unused layers
docker system prune

# Use multi-stage build (already implemented)
# Minimize dependencies in requirements.txt
```

---

## Best Practices

### Security

1. **Run as non-root user** (already configured)
2. **Use read-only mounts** for input data
3. **Scan images** for vulnerabilities:
   ```bash
   docker scan synfinance:latest
   ```
4. **Keep images updated**:
   ```bash
   docker pull ghcr.io/synfinance/synfinance:latest
   ```
5. **Use secrets** for sensitive data:
   ```bash
   docker secret create db_password ./password.txt
   ```

### Performance

1. **Use multi-stage builds** (already implemented)
2. **Leverage layer caching**:
   ```bash
   # Copy requirements first (cached layer)
   COPY requirements.txt .
   RUN pip install -r requirements.txt
   # Then copy code (changes frequently)
   COPY . .
   ```
3. **Minimize image size**:
   - Use slim base images
   - Remove build dependencies
   - Clean package manager cache
4. **Resource limits**:
   ```bash
   --cpus=2.0 --memory=4g
   ```

### Monitoring

1. **Enable health checks** (already configured)
2. **Collect logs**:
   ```bash
   docker logs -f synfinance > synfinance.log
   ```
3. **Use monitoring stack**:
   ```bash
   docker-compose --profile monitoring up -d
   ```
4. **Track metrics**:
   - Prometheus: http://localhost:9090
   - Grafana: http://localhost:3000

### Maintenance

1. **Regular updates**:
   ```bash
   # Pull latest
   docker pull ghcr.io/synfinance/synfinance:latest
   # Restart with new image
   docker-compose up -d
   ```

2. **Cleanup old images**:
   ```bash
   docker image prune
   docker system prune
   ```

3. **Backup volumes**:
   ```bash
   docker run --rm -v synfinance-data:/data -v $(pwd):/backup ubuntu tar czf /backup/data-backup.tar.gz /data
   ```

4. **Monitor disk usage**:
   ```bash
   docker system df
   ```

---

## Additional Resources

- **Docker Documentation:** https://docs.docker.com/
- **Docker Compose Reference:** https://docs.docker.com/compose/
- **Best Practices:** https://docs.docker.com/develop/dev-best-practices/
- **SynFinance CI/CD Guide:** [CICD_GUIDE.md](./CICD_GUIDE.md)
- **SynFinance API Documentation:** http://localhost:8000/docs

---

## Support

For issues or questions:
1. Check logs: `docker logs synfinance`
2. Review health status: `curl http://localhost:8000/health`
3. Consult troubleshooting section above
4. Check GitHub Issues
5. Contact development team

---

**Last Updated:** October 28, 2025  
**Version:** 0.6.6  
**Docker Image:** ghcr.io/synfinance/synfinance:latest
