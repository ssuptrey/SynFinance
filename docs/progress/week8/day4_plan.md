## Week 8 Day 4: Multi-tenancy & Enterprise Features - Implementation Plan

**Date:** November 1, 2025  
**Status:** In Progress  
**Estimated Duration:** 6-8 hours

---

### Overview

Implement comprehensive multi-tenant architecture with tenant isolation, RBAC (Role-Based Access Control), resource quotas, and tenant management APIs. This enables the SynFinance platform to serve multiple organizations with complete data segregation and security.

---

### Goals

1. Implement multi-tenant database architecture
2. Build tenant isolation middleware
3. Add tenant context management
4. Implement RBAC system with permissions
5. Create tenant management API
6. Add resource quotas and rate limiting
7. Build tenant-specific configuration
8. Implement tenant analytics and monitoring

---

### Architecture Decisions

#### Multi-tenancy Strategy

**Chosen Approach: Shared Database with Tenant ID Column**

Pros:
- Cost-effective (single database)
- Easy to maintain and backup
- Efficient resource utilization
- Simpler deployment

Cons:
- Requires strict tenant isolation in queries
- Potential for data leakage if misconfigured
- Noisy neighbor issues possible

**Alternative Considered: Schema-per-tenant**
- More isolation but higher complexity
- Deferred for future if needed

#### Tenant Context

- Thread-local context storage
- Middleware injection for all requests
- GraphQL context propagation
- WebSocket connection-level tenant binding

#### RBAC Model

Roles:
- tenant_admin: Full tenant management
- analyst: Read-only access to analytics
- operator: Transaction and fraud management
- auditor: Read-only audit access
- api_user: Programmatic API access

Permissions:
- transactions.read, transactions.create, transactions.update
- fraud.detect, fraud.review, fraud.override
- analytics.view, analytics.export
- settings.read, settings.update
- users.read, users.create, users.update, users.delete

---

### Implementation Steps

#### Step 1: Database Schema for Multi-tenancy (1 hour)

**Files to Create/Modify:**
- `src/database/models.py` - Add tenant tables and relationships
- `migrations/versions/` - Alembic migration for tenant schema

**Tasks:**
1. Create `tenants` table with metadata
2. Create `tenant_users` table for user-tenant association
3. Create `tenant_roles` and `tenant_permissions` tables
4. Add `tenant_id` column to existing tables (transactions, customers, merchants)
5. Add database constraints for tenant isolation
6. Create indexes for performance

**Schema:**
```sql
-- Tenants table
CREATE TABLE tenants (
    id UUID PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    slug VARCHAR(100) UNIQUE NOT NULL,
    status VARCHAR(50) DEFAULT 'active',
    plan VARCHAR(50) DEFAULT 'free',
    max_transactions INTEGER DEFAULT 10000,
    max_users INTEGER DEFAULT 5,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    metadata JSONB
);

-- Tenant users and roles
CREATE TABLE tenant_users (
    id UUID PRIMARY KEY,
    tenant_id UUID REFERENCES tenants(id),
    user_id UUID NOT NULL,
    email VARCHAR(255) NOT NULL,
    role VARCHAR(50) NOT NULL,
    permissions JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Add tenant_id to existing tables
ALTER TABLE transactions ADD COLUMN tenant_id UUID REFERENCES tenants(id);
ALTER TABLE customers ADD COLUMN tenant_id UUID REFERENCES tenants(id);
ALTER TABLE merchants ADD COLUMN tenant_id UUID REFERENCES tenants(id);
```

#### Step 2: Tenant Context Management (1 hour)

**Files to Create:**
- `src/tenancy/context.py` - Tenant context storage and retrieval
- `src/tenancy/models.py` - Tenant dataclasses and models

**Tasks:**
1. Implement thread-local tenant context storage
2. Create tenant context manager
3. Build tenant resolver from request headers/JWT
4. Add tenant validation and activation check

**Key Components:**
```python
class TenantContext:
    """Thread-local tenant context"""
    
    def set_current_tenant(tenant_id: str)
    def get_current_tenant() -> Optional[str]
    def clear_tenant()
    def require_tenant() -> str
```

#### Step 3: Tenant Isolation Middleware (1.5 hours)

**Files to Create:**
- `src/tenancy/middleware.py` - Tenant isolation middleware
- `src/tenancy/decorators.py` - Tenant requirement decorators

**Tasks:**
1. Create FastAPI middleware for tenant extraction
2. Implement tenant validation and activation check
3. Add request context injection
4. Build tenant isolation for database queries
5. Create decorators for tenant-required endpoints

**Middleware Flow:**
1. Extract tenant from header (X-Tenant-ID) or JWT claim
2. Validate tenant exists and is active
3. Set tenant context for request
4. Inject into request state
5. Clear context after request

#### Step 4: RBAC System (2 hours)

**Files to Create:**
- `src/tenancy/rbac.py` - Role-based access control
- `src/tenancy/permissions.py` - Permission definitions and checks

**Tasks:**
1. Define permission constants and categories
2. Implement role definitions with default permissions
3. Build permission checker functions
4. Create RBAC decorators for endpoints
5. Add permission inheritance and composition
6. Implement user permission caching

**Permission System:**
```python
class Permission(Enum):
    TRANSACTIONS_READ = "transactions.read"
    TRANSACTIONS_CREATE = "transactions.create"
    FRAUD_DETECT = "fraud.detect"
    ANALYTICS_VIEW = "analytics.view"
    SETTINGS_UPDATE = "settings.update"

def require_permission(permission: Permission):
    """Decorator to require specific permission"""
    pass

def has_permission(user_id: str, permission: Permission) -> bool:
    """Check if user has permission"""
    pass
```

#### Step 5: Tenant Management API (1.5 hours)

**Files to Create:**
- `src/api/tenants/routes.py` - Tenant CRUD endpoints
- `src/api/tenants/schemas.py` - Pydantic models for tenants

**Tasks:**
1. Create tenant registration endpoint
2. Build tenant update endpoint
3. Add tenant deactivation/deletion
4. Implement tenant user management
5. Add tenant configuration endpoints
6. Build tenant analytics endpoints

**Endpoints:**
```
POST   /api/v1/tenants              - Create tenant
GET    /api/v1/tenants/{id}         - Get tenant details
PATCH  /api/v1/tenants/{id}         - Update tenant
DELETE /api/v1/tenants/{id}         - Deactivate tenant
GET    /api/v1/tenants/{id}/users   - List tenant users
POST   /api/v1/tenants/{id}/users   - Add user to tenant
GET    /api/v1/tenants/{id}/stats   - Tenant statistics
```

#### Step 6: Resource Quotas and Rate Limiting (1 hour)

**Files to Create:**
- `src/tenancy/quotas.py` - Resource quota management
- `src/tenancy/rate_limits.py` - Tenant-specific rate limiting

**Tasks:**
1. Implement quota tracking (transactions, API calls, storage)
2. Add quota validation before operations
3. Build tenant-specific rate limiters
4. Create quota exceeded error handling
5. Add quota usage endpoints

**Quota Types:**
- Max transactions per month
- Max API calls per day
- Max users per tenant
- Max storage (MB)
- Custom feature flags

#### Step 7: Tenant-Scoped Database Queries (1 hour)

**Files to Modify:**
- `src/database/repositories.py` - Add tenant filtering
- `src/api/graphql/resolvers/queries.py` - Add tenant context

**Tasks:**
1. Update all repository methods with tenant filtering
2. Add automatic tenant_id injection in creates
3. Build tenant-scoped query helpers
4. Update GraphQL resolvers with tenant context
5. Add tenant isolation tests

#### Step 8: Testing (1 hour)

**Files to Create:**
- `tests/tenancy/test_tenant_context.py`
- `tests/tenancy/test_middleware.py`
- `tests/tenancy/test_rbac.py`
- `tests/tenancy/test_quotas.py`
- `tests/api/test_tenant_routes.py`

**Test Coverage:**
1. Tenant context management (5 tests)
2. Middleware tenant extraction (5 tests)
3. RBAC permission checks (10 tests)
4. Quota enforcement (5 tests)
5. Tenant API endpoints (8 tests)
6. Tenant isolation (7 tests)
7. Multi-tenant data segregation (5 tests)

**Target: 45+ tests**

---

### Database Migration

**Migration Steps:**
1. Create new tenant tables
2. Add tenant_id columns to existing tables
3. Create default "system" tenant
4. Migrate existing data to system tenant
5. Add NOT NULL constraint after migration
6. Create indexes for performance

**Backward Compatibility:**
- System tenant for existing data
- Null tenant_id allowed initially
- Gradual migration path

---

### Security Considerations

1. **Tenant Isolation:**
   - All queries must filter by tenant_id
   - No cross-tenant data access
   - Audit logs for tenant access

2. **Authentication:**
   - JWT tokens include tenant_id claim
   - Validate tenant ownership
   - Prevent tenant ID spoofing

3. **Authorization:**
   - RBAC enforced at API level
   - Permission checks before operations
   - Audit trail for permission changes

4. **Data Protection:**
   - Tenant data encryption at rest
   - Separate backup per tenant (optional)
   - GDPR compliance for tenant deletion

---

### Performance Optimization

1. **Indexing:**
   - Composite indexes on (tenant_id, id)
   - Covering indexes for common queries

2. **Caching:**
   - Tenant metadata cached
   - Permission cache per user
   - Quota cache with TTL

3. **Connection Pooling:**
   - Tenant-aware connection pools
   - Statement caching per tenant

---

### Configuration

**Tenant Plans:**
```yaml
plans:
  free:
    max_transactions: 10000
    max_users: 5
    max_api_calls: 1000
    features: ["basic_fraud_detection"]
  
  professional:
    max_transactions: 100000
    max_users: 25
    max_api_calls: 10000
    features: ["basic_fraud_detection", "advanced_ml", "real_time_alerts"]
  
  enterprise:
    max_transactions: -1  # unlimited
    max_users: -1
    max_api_calls: 100000
    features: ["all"]
```

---

### Monitoring and Analytics

1. **Tenant Metrics:**
   - Transaction volume per tenant
   - API usage per tenant
   - Fraud detection rate per tenant
   - Resource consumption per tenant

2. **Alerts:**
   - Quota approaching threshold
   - Unusual activity patterns
   - Performance degradation

3. **Dashboards:**
   - Tenant health overview
   - Resource utilization
   - Cost allocation

---

### Success Criteria

- [ ] All tenant tables created and migrated
- [ ] Tenant context working in all API calls
- [ ] RBAC system enforcing permissions
- [ ] Resource quotas enforced
- [ ] Tenant management API functional
- [ ] 45+ tests passing with 100% coverage
- [ ] No cross-tenant data leakage
- [ ] Documentation complete
- [ ] Demo script working

---

### Timeline

| Task | Duration | Status |
|------|----------|--------|
| Database schema | 1 hour | Not Started |
| Tenant context | 1 hour | Not Started |
| Middleware | 1.5 hours | Not Started |
| RBAC system | 2 hours | Not Started |
| Management API | 1.5 hours | Not Started |
| Quotas | 1 hour | Not Started |
| Query scoping | 1 hour | Not Started |
| Testing | 1 hour | Not Started |

**Total Estimated Time:** 6-8 hours

---

### Next Steps After Completion

Day 5 will focus on:
1. API versioning system
2. Backward compatibility
3. Deprecation management
4. Migration tools
5. Version documentation
