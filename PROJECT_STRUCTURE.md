# SynFinance Project Structure

Complete directory structure and file organization for the SynFinance synthetic transaction data generator.

**Last Updated**: November 2, 2025
**Version**: 2.16.0 (Production/Stable)
**Status**: Week 9 Complete - Production Infrastructure & DevOps

## Project Overview

SynFinance is an enterprise-grade, cloud-native Python-based synthetic financial transaction data generator designed for the Indian market. It creates realistic customer profiles and transactions with advanced behavioral patterns, temporal dynamics, geographic consistency, merchant ecosystems, fraud detection, anomaly detection, ML features, database integration, resilience patterns, professional CLI tools, **GraphQL/WebSocket APIs**, **ensemble ML models**, **multi-tenancy**, **comprehensive API versioning**, and **production infrastructure** with Docker, Kubernetes, CI/CD, observability, and Istio service mesh.

**Key Features**:
- 15 fraud pattern types with ML optimization
- 69 combined ML features + Ensemble models (96.9% accuracy)
- GraphQL API with DataLoader optimization
- WebSocket real-time updates (10K connections, <10ms latency)
- Multi-tenancy with RBAC (1000+ tenants)
- API versioning with migration strategies
- SQLAlchemy 2.0 database integration with PostgreSQL
- Professional CLI with 20+ commands (Click + Rich UI)
- Resilience framework (Circuit Breaker, Retry, Rate Limiter, Health Checker)
- **Docker containerization (5.16GB optimized image)**
- **Kubernetes + Helm (multi-environment deployment)**
- **CI/CD with GitHub Actions + ArgoCD GitOps**
- **Comprehensive observability (Prometheus, Grafana, Loki, Jaeger, 15+ metrics)**
- **Istio service mesh (mTLS, canary deployments, circuit breakers)**
- Prometheus monitoring + Grafana dashboards
- Multi-environment configuration management
- Automated quality assurance framework
- Structured logging and distributed tracing
- 967+ tests passing (97.5% success rate)

---

## Directory Structure

```
SynFinance/
├── src/                                # Source code (28,000+ lines)
│   ├── __init__.py
│   ├── app.py                          # Streamlit application
│   ├── config.py                       # Configuration settings
│   ├── constants.py                    # Global constants
│   ├── customer_generator.py          # Customer profile generation
│   ├── customer_profile.py            # Customer profile class
│   ├── data_generator.py              # Main data generation orchestrator
│   │
│   ├── analytics/                      # Advanced analytics (Week 6)
│   │   ├── __init__.py
│   │   ├── correlation_analyzer.py     # Correlation analysis
│   │   ├── feature_importance_analyzer.py  # Feature importance
│   │   ├── model_performance_analyzer.py   # Model metrics
│   │   └── statistical_tests.py        # Statistical testing
│   │
│   ├── api/                            # FastAPI server & APIs (Week 6, 8)
│   │   ├── __init__.py
│   │   ├── app.py                      # API server
│   │   ├── client.py                   # API client
│   │   ├── schemas.py                  # Pydantic schemas
│   │   │
│   │   ├── graphql/                    # GraphQL API (Week 8 Day 1 - ~1,500 lines)
│   │   │   ├── __init__.py
│   │   │   ├── schema.py               # GraphQL schema definition
│   │   │   ├── types.py                # GraphQL types (Transaction, Customer, etc.)
│   │   │   ├── dataloaders.py          # DataLoader for N+1 optimization
│   │   │   └── resolvers/
│   │   │       ├── __init__.py
│   │   │       ├── queries.py          # Query resolvers
│   │   │       ├── mutations.py        # Mutation resolvers
│   │   │       └── subscriptions.py    # Subscription resolvers
│   │   │
│   │   ├── tenants/                    # Multi-tenancy API (Week 8 Day 4)
│   │   │   ├── __init__.py
│   │   │   ├── routes.py               # Tenant management endpoints (11)
│   │   │   └── schemas.py              # Pydantic schemas for tenants
│   │   │
│   │   └── versioning/                 # API Versioning (Week 8 Day 5 - ~2,300 lines)
│   │       ├── __init__.py
│   │       ├── registry.py             # Version registry (~295 lines)
│   │       ├── negotiation.py          # Version detection (~213 lines)
│   │       ├── deprecation.py          # Deprecation management (~250 lines)
│   │       ├── compatibility.py        # Transformation layer (~328 lines)
│   │       ├── middleware.py           # Version middleware (~95 lines)
│   │       ├── router.py               # Versioned routers (~117 lines)
│   │       └── migration.py            # Migration tools (~435 lines)
│   │
│   ├── cli/                            # CLI tools (Week 7 Day 6 - 882 lines)
│   │   ├── __init__.py
│   │   ├── main_cli.py                 # Main CLI entry point (50 lines)
│   │   ├── generate_commands.py        # Data generation commands (240 lines)
│   │   ├── model_commands.py           # ML model commands (220 lines)
│   │   ├── database_commands.py        # Database commands (150 lines)
│   │   └── system_commands.py          # System commands (150 lines)
│   │
│   ├── config/                         # Configuration (Week 7 Day 2)
│   │   ├── __init__.py
│   │   ├── config_manager.py           # Configuration management
│   │   ├── environment.py              # Environment settings
│   │   ├── schema.py                   # Config schema validation
│   │   └── validator.py                # Config validators
│   │
│   ├── database/                       # Database layer (Week 7 Day 5 - 1,577 lines)
│   │   ├── __init__.py
│   │   ├── models.py                   # SQLAlchemy 2.0 models (620 lines)
│   │   ├── db_manager.py               # Database manager (580 lines)
│   │   └── repositories.py             # Repository pattern (580 lines)
│   │
│   ├── generators/                     # Specialized generators
│   │   ├── __init__.py
│   │   ├── advanced_schema_generator.py    # Advanced schema features
│   │   ├── anomaly_patterns.py         # Anomaly detection (Week 5)
│   │   ├── fraud_patterns.py           # Fraud detection (Week 4)
│   │   ├── geographic_generator.py     # Geographic patterns
│   │   ├── merchant_generator.py       # Merchant ecosystem
│   │   ├── temporal_generator.py       # Temporal patterns
│   │   └── transaction_core.py         # Core transaction logic
│   │
│   ├── ml/                             # ML framework (Week 5-6, 8)
│   │   ├── __init__.py
│   │   ├── anomaly_features.py         # Anomaly ML features
│   │   ├── combined_features.py        # Combined features (69 total)
│   │   ├── dataset_preparation.py      # Dataset prep pipeline
│   │   ├── feature_generator.py        # Feature engineering
│   │   ├── fraud_features.py           # Fraud ML features
│   │   ├── model_optimization.py       # Hyperparameter tuning
│   │   │
│   │   ├── base_model.py               # Base model interface (Week 8 Day 3)
│   │   ├── models/                     # ML Models (Week 8 Day 3 - ~800 lines)
│   │   │   ├── __init__.py
│   │   │   ├── random_forest.py        # RandomForest (95.40% accuracy)
│   │   │   └── xgboost_model.py        # XGBoost (96.90% accuracy)
│   │   │
│   │   └── ensemble/                   # Ensemble Models (Week 8 Day 3 - ~1,000 lines)
│   │       ├── __init__.py
│   │       └── voting.py               # Voting ensemble (soft/hard)
│   │
│   ├── models/                         # Data models
│   │   ├── __init__.py
│   │   └── transaction.py              # Transaction data model
│   │
│   ├── monitoring/                     # Monitoring (Week 7 Day 1 - 4,500 lines)
│   │   ├── __init__.py
│   │   ├── metrics.py                  # Metrics definitions
│   │   ├── prometheus_collector.py     # Prometheus integration
│   │   ├── grafana_dashboard.py        # Grafana dashboards
│   │   └── alerts.py                   # Alert rules
│   │
│   ├── observability/                  # Observability (Week 7 Day 4 - 2,216 lines)
│   │   ├── __init__.py
│   │   ├── logger.py                   # Structured logging
│   │   ├── tracer.py                   # Distributed tracing
│   │   ├── context.py                  # Context management
│   │   └── formatter.py                # Log formatters
│   │
│   ├── performance/                    # Performance (Week 6 Days 1-2)
│   │   ├── __init__.py
│   │   ├── parallel_generator.py       # Parallel generation (45K txn/sec)
│   │   ├── streaming_generator.py      # Streaming generation
│   │   ├── cache.py                    # LRU caching
│   │   └── benchmarks.py               # Performance benchmarking
│   │
│   ├── qa/                             # Quality Assurance (Week 7 Day 3 - 3,473 lines)
│   │   ├── __init__.py
│   │   ├── test_framework.py           # Testing framework
│   │   ├── data_validator.py           # Data validation
│   │   ├── quality_metrics.py          # Quality metrics
│   │   └── test_runner.py              # Test runner
│   │
│   ├── resilience/                     # Resilience Framework (Week 7 Day 7 - 1,441 lines)
│   │   ├── __init__.py
│   │   ├── circuit_breaker.py          # Circuit Breaker pattern (368 lines)
│   │   ├── retry_handler.py            # Retry with exponential backoff (280 lines)
│   │   ├── rate_limiter.py             # Token bucket rate limiter (330 lines)
│   │   └── health_checker.py           # Kubernetes-style health probes (390 lines)
│   │
│   ├── tenancy/                        # Multi-tenancy (Week 8 Day 4 - ~2,500 lines)
│   │   ├── __init__.py
│   │   ├── models.py                   # Tenant, TenantUser, TenantQuota models
│   │   ├── context.py                  # ContextVar-based isolation
│   │   ├── permissions.py              # 30+ granular permissions
│   │   ├── rbac.py                     # RBAC manager (5 roles)
│   │   ├── middleware.py               # Tenant isolation middleware
│   │   └── quotas.py                   # Resource quota management
│   │
│   ├── utils/                          # Utility modules
│   │   ├── __init__.py
│   │   ├── geographic_data.py          # City/region data
│   │   ├── indian_data.py              # Indian market data
│   │   └── merchant_data.py            # Merchant data
│   │
│   └── websocket/                      # WebSocket Real-time (Week 8 Day 2 - ~1,400 lines)
│       ├── __init__.py
│       ├── manager.py                  # Connection manager (10K connections)
│       ├── handlers.py                 # Message handlers
│       ├── events.py                   # 9 event types
│       └── subscriptions.py            # Topic subscriptions
│
├── tests/                              # Test suite (967+ tests passing)
│   ├── __init__.py
│   ├── README.md                       # Test documentation
│   │
│   ├── analytics/                      # Analytics tests (Week 6)
│   │   ├── __init__.py
│   │   ├── test_correlation.py         # Correlation tests
│   │   ├── test_feature_importance.py  # Feature importance tests
│   │   ├── test_model_performance.py   # Model performance tests
│   │   └── test_statistical_tests.py   # Statistical tests
│   │
│   ├── api/                            # API tests (Week 6, 8)
│   │   ├── __init__.py
│   │   ├── test_api_endpoints.py
│   │   ├── test_api_client.py
│   │   ├── test_api_integration.py
│   │   ├── test_graphql.py             # 23 GraphQL tests (Week 8 Day 1)
│   │   ├── test_websocket.py           # 20 WebSocket tests (Week 8 Day 2)
│   │   └── test_versioning.py          # 26 API versioning tests (Week 8 Day 5)
│   │
│   ├── cli/                            # CLI tests (Week 7 Day 7)
│   │   ├── __init__.py
│   │   └── test_cli_commands.py        # 30 tests (13 passing, 17 import mocking limitations)
│   │
│   ├── config/                         # Configuration tests (Week 7 Day 2)
│   │   ├── __init__.py
│   │   └── test_config_*.py            # 42 tests
│   │
│   ├── database/                       # Database tests (Week 7 Day 5)
│   │   ├── __init__.py
│   │   └── test_database.py            # 14 tests (models, manager, repositories)
│   │
│   ├── deployment/                     # Deployment tests (Week 6)
│   │   ├── __init__.py
│   │   └── test_docker.py              # 18 tests (4 passing, 14 require Docker)
│   │
│   ├── generators/                     # Generator tests
│   │   ├── __init__.py
│   │   ├── test_advanced_schema.py     # 30 tests
│   │   ├── test_anomaly_patterns.py    # 30 tests
│   │   ├── test_fraud_patterns.py      # 100 tests
│   │   ├── test_geographic_patterns.py # 15 tests
│   │   ├── test_merchant_ecosystem.py  # 54 tests
│   │   └── test_temporal_patterns.py   # 18 tests
│   │
│   ├── integration/                    # Integration tests
│   │   ├── __init__.py
│   │   └── test_customer_integration.py  # 14 tests
│   │
│   ├── ml/                             # ML tests (Week 5, 8)
│   │   ├── __init__.py
│   │   ├── test_anomaly_features.py    # 23 tests
│   │   ├── test_combined_features.py   # 21 tests
│   │   ├── test_fraud_features.py
│   │   ├── test_dataset_preparation.py
│   │   └── test_ml_models.py           # 25 ensemble ML tests (Week 8 Day 3)
│   │
│   ├── monitoring/                     # Monitoring tests (Week 7 Day 1)
│   │   ├── __init__.py
│   │   └── test_monitoring_*.py        # 85 tests
│   │
│   ├── observability/                  # Observability tests (Week 7 Day 4)
│   │   ├── __init__.py
│   │   └── test_observability_*.py     # 31 tests
│   │
│   ├── performance/                    # Performance tests (Week 6)
│   │   ├── __init__.py
│   │   ├── test_parallel.py
│   │   ├── test_streaming.py
│   │   └── test_cache.py
│   │
│   ├── qa/                             # QA tests (Week 7 Day 3)
│   │   ├── __init__.py
│   │   └── test_qa_*.py                # 74 tests
│   │
│   ├── tenancy/                        # Multi-tenancy tests (Week 8 Day 4)
│   │   ├── __init__.py
│   │   ├── test_middleware.py          # 15 tenant middleware tests
│   │   ├── test_rbac.py                # 20 RBAC tests
│   │   ├── test_tenant_context.py      # 19 context tests
│   │   └── test_tenant_models.py       # 18 model tests
│   │
│   ├── test_resilience.py              # Resilience tests (Week 7 Day 7)
│   │                                   # 33 tests (Circuit Breaker, Retry, Rate Limiter, Health Checker)
│   │
│   ├── deployment/                     # Deployment tests (Week 9)
│   │   ├── __init__.py
│   │   ├── test_docker.py              # 18 tests (4 passing, 14 require Docker)
│   │   └── test_kubernetes.py          # Kubernetes manifest tests
│   │
│   └── unit/                           # Unit tests
│       ├── __init__.py
│       └── test_data_quality.py        # 13 tests
│
├── .github/                            # GitHub Actions (Week 9 Day 3)
│   └── workflows/
│       ├── ci-build-push.yml           # Docker build, scan, push (186 lines)
│       └── ci-manifest.yml             # Manifest validation (80 lines)
│
├── migrations/                         # Alembic migrations (Week 7 Day 5)
│   ├── env.py                          # Migration environment
│   ├── README                          # Migration documentation
│   ├── script.py.mako                  # Migration template
│   └── versions/                       # Migration versions
│
├── config/                             # Configuration files (Week 7 Day 2)
│   ├── default.yaml                    # Base configuration
│   ├── development.yaml                # Dev environment
│   ├── production.yaml                 # Production environment
│   ├── staging.yaml                    # Staging environment
│   ├── test.yaml                       # Test environment
│   └── schema.json                     # Config schema validation
│
├── monitoring/                         # Monitoring configuration (Week 7 Day 1, Week 9 Day 4)
│   ├── grafana/
│   │   └── dashboards/                 # Grafana dashboard JSON
│   │       ├── synfinance_dashboard.json
│   │       ├── application-overview.json   # Week 9: 6 panels (300 lines)
│   │       └── fraud-analytics.json        # Week 9: 3 panels (80 lines)
│   └── prometheus/
│       └── prometheus.yml              # Prometheus configuration
│
├── k8s/                                # Kubernetes manifests (Week 9 Day 2 - ~1,000 lines)
│   ├── README.md                       # Kubernetes overview
│   ├── QUICKSTART.md                   # Quick deployment guide
│   ├── DEPLOYMENT_CHECKLIST.md         # Pre-deployment verification
│   │
│   ├── base/                           # Base manifests (10 files, ~600 lines)
│   │   ├── namespace.yaml              # Namespace definitions
│   │   ├── api-deployment.yaml         # API deployment (3 replicas, rolling updates)
│   │   ├── postgres-statefulset.yaml   # PostgreSQL StatefulSet (10Gi PVC)
│   │   ├── redis-statefulset.yaml      # Redis StatefulSet (5Gi PVC)
│   │   ├── configmap.yaml              # ConfigMap for env vars
│   │   ├── secrets.yaml                # Secrets (base64 encoded)
│   │   ├── ingress.yaml                # Ingress with TLS
│   │   ├── hpa.yaml                    # HorizontalPodAutoscaler (3-10 replicas)
│   │   ├── rbac.yaml                   # RBAC (ServiceAccount, Role, RoleBinding)
│   │   ├── storage-class.yaml          # StorageClass for PVCs
│   │   ├── resource-limits.yaml        # LimitRange
│   │   └── kustomization.yaml          # Kustomize base
│   │
│   ├── overlays/                       # Environment-specific overlays
│   │   ├── production/
│   │   │   ├── kustomization.yaml
│   │   │   ├── patches.yaml            # Production patches
│   │   │   └── argocd-app.yaml         # ArgoCD application (Week 9 Day 3)
│   │   ├── staging/
│   │   │   ├── kustomization.yaml
│   │   │   ├── patches.yaml
│   │   │   └── argocd-app.yaml
│   │   └── development/
│   │       ├── kustomization.yaml
│   │       └── patches.yaml
│   │
│   └── istio/                          # Istio service mesh (Week 9 Day 5 - ~900 lines, 7 files)
│       ├── INSTALL.md                  # Installation guide (400 lines)
│       ├── gateway.yaml                # Ingress gateway (60 lines)
│       ├── virtualservice.yaml         # Traffic routing (180 lines)
│       ├── destinationrule.yaml        # Load balancing, circuit breakers (200 lines)
│       ├── peer-authentication.yaml    # mTLS configuration (40 lines)
│       └── authorization-policy.yaml   # Zero-trust policies (220 lines)
│
├── helm/                               # Helm chart (Week 9 Day 2 - ~400 lines)
│   └── synfinance/
│       ├── Chart.yaml                  # Chart metadata (v0.1.0)
│       ├── values.yaml                 # Default values
│       ├── values-prod.yaml            # Production values
│       ├── values-staging.yaml         # Staging values
│       ├── values-dev.yaml             # Development values
│       └── templates/                  # Helm templates (15 files)
│           ├── deployment.yaml
│           ├── service.yaml
│           ├── ingress.yaml
│           ├── configmap.yaml
│           ├── secrets.yaml
│           ├── hpa.yaml
│           ├── pdb.yaml                # PodDisruptionBudget
│           ├── serviceaccount.yaml
│           ├── role.yaml
│           ├── rolebinding.yaml
│           ├── postgres-statefulset.yaml
│           ├── redis-statefulset.yaml
│           ├── persistentvolumeclaim.yaml
│           ├── NOTES.txt
│           └── _helpers.tpl            # Template helpers
│
├── scripts/                            # Utility scripts (Week 9 Day 3)
│   ├── ci/
│   │   ├── scan_image.sh               # Trivy scanning wrapper
│   │   └── deploy_argocd.sh            # ArgoCD sync helper
│   └── ...                             # Other utility scripts
│
├── Dockerfile                          # Production Docker image (Week 9 Day 1 - 231 lines)
├── docker-compose.yml                  # Docker Compose (Week 9 Day 1 - 3 services)
├── docker-compose.dev.yml              # Development compose
├── .dockerignore                       # Docker build context exclusions
│
├── deploy/                             # Legacy deployment files (pre-Week 9)
│   ├── docker/
│   │   ├── Dockerfile                  # Production Docker image
│   │   ├── Dockerfile.dev              # Development Docker image
│   │   └── docker-compose.yml          # Docker Compose orchestration
│   └── kubernetes/                     # Kubernetes manifests
│       ├── deployment.yaml
│       ├── service.yaml
│       ├── ingress.yaml
│       ├── configmap.yaml
│       └── secrets.yaml
│
├── examples/                           # Example scripts
│   ├── README.md                       # Examples documentation
│   ├── analyze_anomaly_patterns.py
│   ├── analyze_fraud_patterns.py
│   ├── api_demo.py
│   ├── api_integration_example.py      # Week 6 API integration (380 lines)
│   ├── batch_processing_example.py     # Week 6 batch processing (430 lines)
│   ├── complete_ml_pipeline.py         # Week 6 complete pipeline (850 lines)
│   ├── demo_all_fraud_patterns.py
│   ├── demo_analytics_dashboard.py
│   ├── demo_api_versioning.py          # Week 8 Day 5 API versioning demo (411 lines)
│   ├── demo_ensemble_models.py         # Week 8 Day 3 ensemble ML demo
│   ├── demo_geographic_patterns.py
│   ├── demo_merchant_ecosystem.py
│   ├── demo_observability.py
│   ├── demo_qa_framework.py
│   ├── demo_tenancy.py                 # Week 8 Day 4 multi-tenancy demo
│   ├── fraud_detection_tutorial.ipynb  # Jupyter notebook tutorial
│   ├── fraud_detection_tutorial.py
│   ├── generate_anomaly_dataset.py
│   ├── generate_anomaly_ml_features.py
│   ├── generate_combined_features.py
│   ├── generate_fraud_training_data.py
│   ├── monitoring_demo.py
│   ├── optimize_fraud_models.py
│   ├── performance_demo.py
│   ├── real_time_monitoring.py         # Week 6 real-time monitoring (570 lines)
│   ├── run_customer_test.py
│   └── train_fraud_detector.py
│
├── scripts/                            # Utility scripts
│   ├── README.md                       # Scripts documentation
│   ├── analyze_correlations.py
│   ├── analyze_variance.py
│   ├── deploy.sh                       # Deployment script
│   ├── generate_week3_dataset.py
│   ├── health_check.sh                 # Health check script
│   ├── refactor_script.py             # Code refactoring utility
│   ├── rollback.sh                    # Rollback script
│   ├── run.bat                        # Windows runner
│   ├── run.sh                         # Unix/Linux runner
│   └── validate_data_quality.py
│
├── docs/                               # Documentation (50+ documents)
│   ├── INDEX.md                       # Documentation index
│   ├── ORGANIZATION.md                # Documentation organization
│   ├── RECOVERY_REPORT_OCT21.md       # Recovery documentation
│   ├── STRUCTURE.md                   # Structure documentation
│   │
│   ├── guides/                        # User guides
│   │   ├── CI_CD_SETUP.md             # Week 9 Day 3: CI/CD configuration (300+ lines)
│   │   ├── INTEGRATION_GUIDE.md       # API integration guide
│   │   ├── OBSERVABILITY_GUIDE.md     # Week 9 Day 4: Metrics, logging, tracing (400+ lines)
│   │   ├── QUICK_REFERENCE.md         # Quick reference
│   │   ├── QUICKSTART.md              # 5-minute quickstart
│   │   ├── ROLLBACK_RUNBOOK.md        # Week 9 Day 3: Incident response (200+ lines)
│   │   ├── SERVICE_MESH_GUIDE.md      # Week 9 Day 5: Istio service mesh (1000+ lines)
│   │   └── WEEK1_GUIDE.md             # Week 1 tutorial
│   │
│   ├── technical/                     # Technical documentation
│   │   ├── ARCHITECTURE.md            # System architecture
│   │   ├── CHANGES.md                 # Change log
│   │   ├── CUSTOMER_SCHEMA.md         # Customer schema reference
│   │   ├── DESIGN_GUIDE.md            # Design patterns
│   │   ├── FIELD_REFERENCE.md         # Field reference (50 fields)
│   │   │
│   │   ├── deployment/                # Deployment documentation (Week 6)
│   │   │   └── PRODUCTION_GUIDE.md    # Production deployment guide (1,100 lines)
│   │   │
│   │   ├── fraud/                     # Fraud documentation (Week 4)
│   │   │   ├── FRAUD_PATTERNS.md
│   │   │   └── FRAUD_TECHNICAL.md
│   │   │
│   │   └── ml/                        # ML documentation (Week 5)
│   │       ├── ML_FEATURES.md
│   │       └── ANOMALY_FEATURES.md
│   │
│   ├── progress/                      # Progress reports
│   │   ├── README.md                  # Progress documentation index
│   │   ├── WEEK1_COMPLETION_SUMMARY.md
│   │   ├── WEEK2_DAY1-2_SUMMARY.md
│   │   ├── WEEK2_DAY3-4_SUMMARY.md
│   │   ├── WEEK2_DAY5-7_SUMMARY.md
│   │   ├── WEEK3_DAY1_COMPLETE.md
│   │   ├── WEEK3_DAY2-3_ANALYSIS.md
│   │   ├── WEEK3_DAY2-3_COMPLETE.md
│   │   ├── WEEK4_DAY1-2_COMPLETE.md
│   │   ├── WEEK4_DAY3-4_COMPLETE.md
│   │   ├── WEEK5_DAY1-7_COMPLETE.md
│   │   ├── WEEK6_DAY1-2_COMPLETE.md
│   │   ├── WEEK6_DAY3-4_COMPLETE.md
│   │   ├── WEEK6_DAY5_COMPLETE.md
│   │   ├── WEEK6_DAY6_COMPLETE.md
│   │   ├── WEEK6_DAY7_COMPLETE.md
│   │   ├── WEEK7_DAY5_COMPLETE.md     # Database integration (Week 7)
│   │   ├── WEEK7_DAY6_COMPLETE.md     # CLI tools (Week 7)
│   │   ├── WEEK7_COMPLETE.md          # Week 7 comprehensive summary
│   │   │
│   │   ├── week8/                     # Week 8 progress
│   │   │   ├── README.md              # Week 8 overview
│   │   │   ├── day1_complete.md       # GraphQL API (12.2 KB)
│   │   │   ├── day2_complete.md       # WebSocket (14.5 KB)
│   │   │   ├── day3_complete.md       # Ensemble ML (13.6 KB)
│   │   │   ├── day4_complete.md       # Multi-tenancy (11.5 KB)
│   │   │   ├── day5_complete.md       # API Versioning (13.8 KB)
│   │   │   ├── WEEK8_COMPLETION_SUMMARY.md   # Complete week summary
│   │   │   └── DEPENDENCIES_INSTALLED.md     # Dependency guide
│   │   │
│   │   └── week9/                     # Week 9 progress (NEW)
│   │       ├── day1_complete.md       # Docker containerization
│   │       ├── day2_complete.md       # Kubernetes & Helm
│   │       ├── day3_complete.md       # CI/CD & GitOps
│   │       ├── day4_complete.md       # Observability stack
│   │       ├── day4_plan.md           # Observability planning
│   │       ├── day5_complete.md       # Service mesh
│   │       ├── day5_plan.md           # Service mesh planning
│   │       └── WEEK9_COMPLETE.md      # Complete week summary (~850 lines)
│   │
│   └── planning/                      # Planning documents
│       ├── ROADMAP.md                 # 12-week roadmap
│       ├── BUSINESS_PLAN.md           # Business strategy
│       ├── ASSESSMENT_SUMMARY.md
│       └── WEEK7_DETAILED_PLAN.md
│
├── data/                               # Sample data (gitignored)
├── output/                             # Generated outputs (gitignored)
│   ├── analytics/                     # Analytics output
│   ├── combined_features/             # Combined features
│   └── qa_demo/                       # QA demo output
│
├── models/                             # Trained ML models (gitignored)
│
├── alembic.ini                         # Alembic configuration (Week 7)
├── docker-compose.yml                  # Docker Compose (Week 6)
├── Dockerfile                          # Production Dockerfile (Week 6)
├── Dockerfile.dev                      # Development Dockerfile (Week 6)
├── .dockerignore                       # Docker ignore file
├── .github/                            # GitHub Actions CI/CD (Week 6)
│   └── workflows/
│       ├── test.yml                    # Test workflow
│       ├── build.yml                   # Build workflow
│       └── deploy.yml                  # Deploy workflow
│
├── requirements.txt                    # Python dependencies
├── setup.py                            # Package setup
├── setup.cfg                           # Setup configuration
├── pyproject.toml                      # Project configuration
├── pytest.ini                          # Pytest configuration
├── MANIFEST.in                         # Package manifest
├── README.md                           # Project README
├── LICENSE                             # License file
├── CHANGELOG.md                        # Change log
├── CONTRIBUTING.md                     # Contribution guidelines
├── PROJECT_STRUCTURE.md                # This file
└── DOCUMENTATION_COMPLETE.md           # Documentation status
│   │   ├── ASSESSMENT_SUMMARY.md      # (empty)
│   │   ├── BUSINESS_PLAN.md           # (empty)
│   │   └── ROADMAP.md                 # (empty)
│   │
│   └── archive/                       # Archived documentation
│       ├── README.md                  # (empty)
│       ├── PROJECT_STRUCTURE.md       # Old structure (29KB)
│       ├── PROJECT_VALIDATION.md      # (empty)
│       ├── README.md                  # (empty)
│       └── REFACTORING_COMPLETE.md    # (empty)
│
├── data/                              # Data files (empty)
├── output/                            # Generated output
│   └── customer_validation_stats.json
│
├── .venv/                             # Virtual environment
├── __pycache__/                       # Python cache
│
├── CONTRIBUTING.md                    # Contribution guidelines
├── DOCUMENTATION_COMPLETE.md          # Documentation status
├── LICENSE                            # MIT License
├── PROJECT_STRUCTURE.md               # This file
├── README.md                          # Project README
└── requirements.txt                   # Python dependencies
```

---

## Key Files

### Source Code

#### Main Entry Points
- **`src/app.py`** - Main application entry point (Streamlit UI)
- **`src/data_generator.py`** - High-level data generation API
- **`src/config.py`** - Global configuration settings

#### Customer Generation
- **`src/customer_profile.py`** - CustomerProfile class (23 fields, 5 enums)
- **`src/customer_generator.py`** - Customer generation logic

#### Transaction Generation
- **`src/generators/transaction_core.py`** - Core transaction generation
- **`src/generators/temporal_generator.py`** - Temporal patterns (Week 2 Day 1-2)
- **`src/generators/geographic_generator.py`** - Geographic patterns (Week 2 Day 3-4)
- **`src/generators/merchant_generator.py`** - Merchant ecosystem (Week 2 Day 5-7)
- **`src/generators/advanced_schema_generator.py`** - Advanced features (Week 3)

#### Data & Utilities
- **`src/utils/indian_data.py`** - Indian market data (names, occupations, etc.)
- **`src/utils/geographic_data.py`** - City/region data (20 cities, 3 tiers)
- **`src/utils/merchant_data.py`** - Merchant data (40+ chains, categories)
- **`src/models/transaction.py`** - Transaction data model

### Tests

#### Integration Tests
- **`tests/integration/test_customer_integration.py`** - Week 1 integration (14 tests)

#### Generator Tests
- **`tests/generators/test_temporal_patterns.py`** - Week 2 Day 1-2 (18 tests)
- **`tests/generators/test_geographic_patterns.py`** - Week 2 Day 3-4 (15 tests)
- **`tests/generators/test_merchant_ecosystem.py`** - Week 2 Day 5-7 (21 tests)

**Total:** 68 tests, 68 passing (100%)

### Documentation

#### Getting Started
- **`docs/guides/QUICKSTART.md`** - 5-minute quickstart guide
- **`docs/guides/INTEGRATION_GUIDE.md`** - API integration guide
- **`docs/guides/QUICK_REFERENCE.md`** - Quick reference with code snippets

#### Technical
- **`docs/technical/ARCHITECTURE.md`** - System architecture overview
- **`docs/technical/CUSTOMER_SCHEMA.md`** - Complete customer schema reference
- **`docs/technical/WEEK*_SUMMARY.md`** - Weekly implementation summaries

#### Progress
- **`docs/progress/WEEK*_COMPLETE.md`** - Detailed weekly progress reports
- **`docs/progress/README.md`** - Progress documentation index

---

## File Statistics

### Source Code
- **Lines of Code:** ~8,500 (excluding tests)
- **Python Files:** 18
- **Modules:** 4 main (customer, transaction, generators, utils)

### Tests
- **Test Files:** 7
- **Test Cases:** 68 (100% passing)
- **Lines of Test Code:** ~4,200

### Documentation
- **Markdown Files:** 30
- **Total Documentation:** 152+ KB
- **With Content:** 19 files (63%)
- **Empty/Placeholder:** 11 files (37%)

---

## Data Flow

```
┌─────────────────────────────────────────────────────────┐
│                     User Application                     │
└──────────────────┬──────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────┐
│              data_generator.generate_dataset()          │
└──────────┬────────────────────────┬─────────────────────┘
           │                        │
           ▼                        ▼
┌──────────────────────┐  ┌────────────────────────────────┐
│ CustomerGenerator    │  │  TransactionGenerator          │
│ .generate_customers()│  │  .generate_transactions()      │
└──────────┬───────────┘  └────────┬───────────────────────┘
           │                       │
           │                       ▼
           │              ┌────────────────────────────────┐
           │              │ TemporalPatternGenerator       │
           │              │ .apply_temporal_multiplier()   │
           │              └────────┬───────────────────────┘
           │                       │
           │                       ▼
           │              ┌────────────────────────────────┐
           │              │ GeographicPatternGenerator     │
           │              │ .select_transaction_city()     │
           │              └────────┬───────────────────────┘
           │                       │
           │                       ▼
           │              ┌────────────────────────────────┐
           │              │ MerchantEcosystemGenerator     │
           │              │ .select_merchant()             │
           │              └────────┬───────────────────────┘
           │                       │
           ▼                       ▼
┌──────────────────────────────────────────────────────────┐
│              Customer Profiles + Transactions            │
│                 (pandas DataFrame)                       │
└──────────────────────────────────────────────────────────┘
```

---

## Module Dependencies

```
data_generator
├── customer_generator
│   ├── customer_profile
│   └── utils/indian_data
│
└── generators/transaction_core
    ├── generators/temporal_generator
    │   └── utils/indian_data (festivals)
    │
    ├── generators/geographic_generator
    │   └── utils/geographic_data
    │
    ├── generators/merchant_generator
    │   └── utils/merchant_data
    │
    └── generators/advanced_schema_generator
        └── models/transaction
```

---

## Configuration

### Environment Variables
```bash
PYTHONPATH=e:\SynFinance\src
```

### Dependencies (requirements.txt)
```
streamlit>=1.28.0
pandas>=2.0.0
faker>=20.0.0
numpy>=1.24.0
xlsxwriter>=3.1.0
pytest>=7.4.0 (dev)
```

---

## Version History

### v0.3.0 (Week 2 Complete - October 2025)
- ✅ Temporal patterns (18 tests)
- ✅ Geographic patterns (15 tests)
- ✅ Merchant ecosystem (21 tests)
- ✅ Advanced schema features
- ✅ 68/68 tests passing

### v0.2.0 (Week 1 Complete - October 2025)
- ✅ Customer profile generation (23 fields)
- ✅ Transaction integration (14 tests)
- ✅ Indian market patterns
- ✅ Data validation

### v0.1.0 (Initial Release)
- Basic customer and transaction generation
- Streamlit UI
- CSV/Excel export

---

## Future Additions

### Planned Features
- Unit test suite (`tests/unit/`)
- Planning documentation (`docs/planning/`)
- Additional example scripts
- Performance optimization tools
- Data visualization utilities

### Planned Documentation
- `docs/guides/WEEK1_GUIDE.md` - Detailed Week 1 tutorial
- `docs/technical/CHANGES.md` - Complete change log
- `docs/technical/DESIGN_GUIDE.md` - Design patterns guide
- `docs/planning/ROADMAP.md` - Product roadmap
- `docs/planning/BUSINESS_PLAN.md` - Market strategy

---

## Contributing

When adding new files:
1. Follow the established directory structure
2. Update this document with new file locations
3. Add appropriate documentation in `docs/`
4. Write tests in `tests/` matching the file structure
5. Update `README.md` if adding user-facing features

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

---

**Last Updated:** October 21, 2025  
**Project Status:** Active Development  
**Test Coverage:** 68/68 tests passing (100%)
