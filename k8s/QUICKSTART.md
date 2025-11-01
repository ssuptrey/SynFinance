# Quick Start Guide - SynFinance Kubernetes Deployment

## Local Development (5 Minutes)

### Prerequisites
- Minikube or Kind installed
- kubectl installed
- Docker image built: `synfinance:2.15.0`

### Deploy in 4 Commands

```bash
# 1. Start Minikube
minikube start --memory=8192 --cpus=4
minikube addons enable ingress metrics-server

# 2. Load Docker image
minikube image load synfinance:2.15.0

# 3. Deploy all resources
kubectl apply -k k8s/base

# 4. Port forward and test
kubectl port-forward -n synfinance svc/synfinance-api 8000:8000
curl http://localhost:8000/health
```

## Production Deployment with Helm

```bash
# 1. Create secrets
kubectl create secret generic synfinance-secrets \
  --from-literal=DATABASE_URL="$DATABASE_URL" \
  --from-literal=REDIS_URL="$REDIS_URL" \
  --from-literal=SECRET_KEY="$(openssl rand -hex 32)" \
  --from-literal=JWT_SECRET="$(openssl rand -hex 32)" \
  -n synfinance

# 2. Deploy with Helm
helm upgrade --install synfinance ./helm/synfinance \
  --namespace synfinance \
  --create-namespace \
  --values helm/synfinance/values-prod.yaml \
  --wait

# 3. Verify
helm test synfinance -n synfinance
kubectl get all -n synfinance
```

## Useful Commands

```bash
# View all resources
kubectl get all -n synfinance

# View logs
kubectl logs -n synfinance -l app.kubernetes.io/name=synfinance-api -f

# Check pod status
kubectl get pods -n synfinance -o wide

# Execute shell in pod
kubectl exec -it POD_NAME -n synfinance -- /bin/sh

# Port forward
kubectl port-forward -n synfinance svc/synfinance-api 8000:8000

# Scale deployment
kubectl scale deployment/synfinance-api --replicas=5 -n synfinance

# Restart deployment
kubectl rollout restart deployment/synfinance-api -n synfinance

# Rollback deployment
kubectl rollout undo deployment/synfinance-api -n synfinance
```

## Troubleshooting

### Pods not starting?
```bash
kubectl describe pod POD_NAME -n synfinance
kubectl logs POD_NAME -n synfinance
```

### Can't pull image?
```bash
# For Minikube
minikube image load synfinance:2.15.0

# For Kind
kind load docker-image synfinance:2.15.0
```

### Database connection failed?
```bash
kubectl logs -n synfinance postgres-0
kubectl exec -it postgres-0 -n synfinance -- psql -U synfinance_trey -d synfinance
```

## Full Documentation

See [k8s/README.md](./README.md) for complete documentation.
