"""
Comprehensive Kubernetes deployment tests for SynFinance.

Tests cover:
- YAML manifest validation
- Kubernetes resource deployment
- Health checks and probes
- Autoscaling functionality
- Network policies
- Persistent storage
- Security configurations
"""

import pytest
import yaml
import subprocess
import time
import requests
from pathlib import Path
from typing import Dict, List, Any


class TestKubernetesManifests:
    """Test all Kubernetes YAML manifests for validity."""
    
    @pytest.fixture
    def k8s_base_path(self) -> Path:
        """Get the base path for Kubernetes manifests."""
        return Path(__file__).parent.parent.parent / "k8s" / "base"
    
    def test_namespace_yaml_valid(self, k8s_base_path: Path):
        """Test that namespace.yaml is valid YAML."""
        manifest_path = k8s_base_path / "namespace.yaml"
        assert manifest_path.exists(), "namespace.yaml not found"
        
        with open(manifest_path, 'r') as f:
            manifests = list(yaml.safe_load_all(f))
        
        assert len(manifests) > 0, "No manifests found in namespace.yaml"
        
        namespace = manifests[0]
        assert namespace['kind'] == 'Namespace'
        assert namespace['metadata']['name'] == 'synfinance'
        assert 'labels' in namespace['metadata']
        assert namespace['metadata']['labels']['environment'] == 'production'
    
    def test_configmap_yaml_valid(self, k8s_base_path: Path):
        """Test that configmap.yaml is valid YAML with required config."""
        manifest_path = k8s_base_path / "configmap.yaml"
        assert manifest_path.exists(), "configmap.yaml not found"
        
        with open(manifest_path, 'r') as f:
            manifests = list(yaml.safe_load_all(f))
        
        # Find the main synfinance-config ConfigMap
        config_map = next((m for m in manifests if m.get('metadata', {}).get('name') == 'synfinance-config'), None)
        assert config_map is not None, "synfinance-config ConfigMap not found"
        
        # Verify required configuration keys
        required_keys = [
            'API_HOST', 'API_PORT', 'LOG_LEVEL', 'ENVIRONMENT',
            'ENABLE_GRAPHQL', 'ENABLE_WEBSOCKET', 'ENABLE_MULTITENANCY'
        ]
        for key in required_keys:
            assert key in config_map['data'], f"Required config key '{key}' not found"
    
    def test_secrets_yaml_valid(self, k8s_base_path: Path):
        """Test that secrets.yaml is valid YAML."""
        manifest_path = k8s_base_path / "secrets.yaml"
        assert manifest_path.exists(), "secrets.yaml not found"
        
        with open(manifest_path, 'r') as f:
            manifests = list(yaml.safe_load_all(f))
        
        # Should have synfinance-secrets, postgres-secrets, and redis-secrets
        secret_names = [m.get('metadata', {}).get('name') for m in manifests]
        assert 'synfinance-secrets' in secret_names
        assert 'postgres-secrets' in secret_names
        assert 'redis-secrets' in secret_names
        
        # Verify synfinance-secrets has required keys
        synfinance_secret = next((m for m in manifests if m.get('metadata', {}).get('name') == 'synfinance-secrets'), None)
        required_keys = ['DATABASE_URL', 'REDIS_URL', 'SECRET_KEY', 'JWT_SECRET']
        for key in required_keys:
            assert key in synfinance_secret['stringData'], f"Required secret key '{key}' not found"
    
    def test_api_deployment_yaml_valid(self, k8s_base_path: Path):
        """Test that api-deployment.yaml is valid with proper configuration."""
        manifest_path = k8s_base_path / "api-deployment.yaml"
        assert manifest_path.exists(), "api-deployment.yaml not found"
        
        with open(manifest_path, 'r') as f:
            manifests = list(yaml.safe_load_all(f))
        
        # Find Deployment and Service
        deployment = next((m for m in manifests if m.get('kind') == 'Deployment'), None)
        service = next((m for m in manifests if m.get('kind') == 'Service'), None)
        
        assert deployment is not None, "Deployment not found"
        assert service is not None, "Service not found"
        
        # Verify Deployment configuration
        spec = deployment['spec']
        assert spec['replicas'] == 3, "Expected 3 replicas for HA"
        
        # Verify rolling update strategy
        assert spec['strategy']['type'] == 'RollingUpdate'
        assert spec['strategy']['rollingUpdate']['maxUnavailable'] == 0
        
        # Verify container configuration
        container = spec['template']['spec']['containers'][0]
        assert container['name'] == 'api'
        assert container['image'] == 'synfinance:2.15.0'
        
        # Verify resource limits are set
        assert 'resources' in container
        assert 'requests' in container['resources']
        assert 'limits' in container['resources']
        
        # Verify probes are configured
        assert 'livenessProbe' in container
        assert 'readinessProbe' in container
        assert 'startupProbe' in container
        
        # Verify security context
        assert 'securityContext' in spec['template']['spec']
        assert spec['template']['spec']['securityContext']['runAsNonRoot'] is True
    
    def test_postgres_statefulset_yaml_valid(self, k8s_base_path: Path):
        """Test that postgres-statefulset.yaml is valid."""
        manifest_path = k8s_base_path / "postgres-statefulset.yaml"
        assert manifest_path.exists(), "postgres-statefulset.yaml not found"
        
        with open(manifest_path, 'r') as f:
            manifests = list(yaml.safe_load_all(f))
        
        # Find StatefulSet
        statefulset = next((m for m in manifests if m.get('kind') == 'StatefulSet'), None)
        assert statefulset is not None, "StatefulSet not found"
        
        # Verify StatefulSet configuration
        assert statefulset['spec']['serviceName'] == 'postgres'
        assert statefulset['spec']['replicas'] == 1
        
        # Verify volume claim templates
        assert 'volumeClaimTemplates' in statefulset['spec']
        assert len(statefulset['spec']['volumeClaimTemplates']) > 0
        
        pvc = statefulset['spec']['volumeClaimTemplates'][0]
        assert pvc['spec']['accessModes'] == ['ReadWriteOnce']
        assert '20Gi' in pvc['spec']['resources']['requests']['storage']
    
    def test_redis_statefulset_yaml_valid(self, k8s_base_path: Path):
        """Test that redis-statefulset.yaml is valid."""
        manifest_path = k8s_base_path / "redis-statefulset.yaml"
        assert manifest_path.exists(), "redis-statefulset.yaml not found"
        
        with open(manifest_path, 'r') as f:
            manifests = list(yaml.safe_load_all(f))
        
        # Find StatefulSet
        statefulset = next((m for m in manifests if m.get('kind') == 'StatefulSet'), None)
        assert statefulset is not None, "StatefulSet not found"
        
        # Verify configuration
        assert statefulset['spec']['serviceName'] == 'redis'
        assert 'volumeClaimTemplates' in statefulset['spec']
    
    def test_hpa_yaml_valid(self, k8s_base_path: Path):
        """Test that hpa.yaml is valid with proper autoscaling configuration."""
        manifest_path = k8s_base_path / "hpa.yaml"
        assert manifest_path.exists(), "hpa.yaml not found"
        
        with open(manifest_path, 'r') as f:
            manifests = list(yaml.safe_load_all(f))
        
        # Find HPA
        hpa = next((m for m in manifests if m.get('kind') == 'HorizontalPodAutoscaler'), None)
        assert hpa is not None, "HPA not found"
        
        # Verify HPA configuration
        spec = hpa['spec']
        assert spec['minReplicas'] == 3
        assert spec['maxReplicas'] == 20
        
        # Verify metrics
        assert 'metrics' in spec
        assert len(spec['metrics']) >= 2  # CPU and Memory
        
        # Verify behavior
        assert 'behavior' in spec
        assert 'scaleUp' in spec['behavior']
        assert 'scaleDown' in spec['behavior']
        
        # Find PodDisruptionBudgets
        pdbs = [m for m in manifests if m.get('kind') == 'PodDisruptionBudget']
        assert len(pdbs) >= 3, "Expected PDBs for API, PostgreSQL, and Redis"
    
    def test_ingress_yaml_valid(self, k8s_base_path: Path):
        """Test that ingress.yaml is valid with proper configuration."""
        manifest_path = k8s_base_path / "ingress.yaml"
        assert manifest_path.exists(), "ingress.yaml not found"
        
        with open(manifest_path, 'r') as f:
            manifests = list(yaml.safe_load_all(f))
        
        # Find Ingress
        ingress = next((m for m in manifests if m.get('kind') == 'Ingress'), None)
        assert ingress is not None, "Ingress not found"
        
        # Verify Ingress configuration
        assert ingress['spec']['ingressClassName'] == 'nginx'
        assert 'tls' in ingress['spec']
        assert 'rules' in ingress['spec']
        
        # Verify security annotations
        annotations = ingress['metadata']['annotations']
        assert 'nginx.ingress.kubernetes.io/ssl-redirect' in annotations
        assert annotations['nginx.ingress.kubernetes.io/ssl-redirect'] == 'true'
        
        # Find NetworkPolicies
        network_policies = [m for m in manifests if m.get('kind') == 'NetworkPolicy']
        assert len(network_policies) >= 3, "Expected NetworkPolicies for API, PostgreSQL, and Redis"
    
    def test_resource_limits_yaml_valid(self, k8s_base_path: Path):
        """Test that resource-limits.yaml is valid."""
        manifest_path = k8s_base_path / "resource-limits.yaml"
        assert manifest_path.exists(), "resource-limits.yaml not found"
        
        with open(manifest_path, 'r') as f:
            manifests = list(yaml.safe_load_all(f))
        
        # Find ResourceQuota
        quota = next((m for m in manifests if m.get('kind') == 'ResourceQuota'), None)
        assert quota is not None, "ResourceQuota not found"
        
        # Verify ResourceQuota
        assert 'hard' in quota['spec']
        hard_limits = quota['spec']['hard']
        assert 'requests.cpu' in hard_limits
        assert 'requests.memory' in hard_limits
        assert 'limits.cpu' in hard_limits
        assert 'limits.memory' in hard_limits
        
        # Find LimitRange
        limit_range = next((m for m in manifests if m.get('kind') == 'LimitRange'), None)
        assert limit_range is not None, "LimitRange not found"
    
    def test_rbac_yaml_valid(self, k8s_base_path: Path):
        """Test that rbac.yaml is valid with proper permissions."""
        manifest_path = k8s_base_path / "rbac.yaml"
        assert manifest_path.exists(), "rbac.yaml not found"
        
        with open(manifest_path, 'r') as f:
            manifests = list(yaml.safe_load_all(f))
        
        # Verify ServiceAccounts
        service_accounts = [m for m in manifests if m.get('kind') == 'ServiceAccount']
        assert len(service_accounts) >= 3, "Expected ServiceAccounts for API, PostgreSQL, and Redis"
        
        # Verify all ServiceAccounts have automountServiceAccountToken = false
        for sa in service_accounts:
            assert sa.get('automountServiceAccountToken') is False, "ServiceAccount should not auto-mount tokens"
        
        # Verify Role exists
        role = next((m for m in manifests if m.get('kind') == 'Role'), None)
        assert role is not None, "Role not found"
        
        # Verify RoleBinding exists
        role_binding = next((m for m in manifests if m.get('kind') == 'RoleBinding'), None)
        assert role_binding is not None, "RoleBinding not found"


@pytest.mark.integration
@pytest.mark.skipif(
    subprocess.run(['kubectl', 'version', '--client'], capture_output=True).returncode != 0,
    reason="kubectl not available"
)
class TestKubernetesDeployment:
    """Integration tests for Kubernetes deployment."""
    
    @pytest.fixture(scope="class")
    def k8s_namespace(self):
        """Ensure namespace exists for testing."""
        return "synfinance"
    
    def run_kubectl(self, args: List[str]) -> Dict[str, Any]:
        """Run kubectl command and return JSON output."""
        result = subprocess.run(
            ['kubectl'] + args + ['-o', 'json'],
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            pytest.fail(f"kubectl command failed: {result.stderr}")
        
        return yaml.safe_load(result.stdout) if result.stdout else {}
    
    def test_namespace_exists(self, k8s_namespace: str):
        """Test that the namespace was created successfully."""
        namespaces = self.run_kubectl(['get', 'namespace'])
        namespace_names = [ns['metadata']['name'] for ns in namespaces.get('items', [])]
        assert k8s_namespace in namespace_names, f"Namespace {k8s_namespace} not found"
    
    def test_configmaps_created(self, k8s_namespace: str):
        """Test that all ConfigMaps were created."""
        configmaps = self.run_kubectl(['get', 'configmap', '-n', k8s_namespace])
        cm_names = [cm['metadata']['name'] for cm in configmaps.get('items', [])]
        
        assert 'synfinance-config' in cm_names
        assert 'postgres-config' in cm_names
        assert 'redis-config' in cm_names
    
    def test_secrets_created(self, k8s_namespace: str):
        """Test that all Secrets were created."""
        secrets = self.run_kubectl(['get', 'secret', '-n', k8s_namespace])
        secret_names = [s['metadata']['name'] for s in secrets.get('items', [])]
        
        assert 'synfinance-secrets' in secret_names
        assert 'postgres-secrets' in secret_names
        assert 'redis-secrets' in secret_names
    
    def test_services_created(self, k8s_namespace: str):
        """Test that all Services were created."""
        services = self.run_kubectl(['get', 'service', '-n', k8s_namespace])
        service_names = [s['metadata']['name'] for s in services.get('items', [])]
        
        assert 'synfinance-api' in service_names
        assert 'postgres' in service_names
        assert 'redis' in service_names
    
    def test_deployments_created(self, k8s_namespace: str):
        """Test that Deployment was created."""
        deployments = self.run_kubectl(['get', 'deployment', '-n', k8s_namespace])
        deployment_names = [d['metadata']['name'] for d in deployments.get('items', [])]
        
        assert 'synfinance-api' in deployment_names
    
    def test_statefulsets_created(self, k8s_namespace: str):
        """Test that StatefulSets were created."""
        statefulsets = self.run_kubectl(['get', 'statefulset', '-n', k8s_namespace])
        ss_names = [ss['metadata']['name'] for ss in statefulsets.get('items', [])]
        
        assert 'postgres' in ss_names
        assert 'redis' in ss_names
    
    def test_pods_running(self, k8s_namespace: str):
        """Test that all pods are in Running state."""
        # Wait for pods to start (max 5 minutes)
        max_wait = 300
        start_time = time.time()
        
        while time.time() - start_time < max_wait:
            pods = self.run_kubectl(['get', 'pod', '-n', k8s_namespace])
            
            if not pods.get('items'):
                time.sleep(5)
                continue
            
            all_running = all(
                pod['status'].get('phase') == 'Running'
                for pod in pods['items']
            )
            
            if all_running:
                return
            
            time.sleep(5)
        
        pytest.fail("Pods did not reach Running state within timeout")
    
    def test_health_checks_passing(self, k8s_namespace: str):
        """Test that all pods pass health checks."""
        pods = self.run_kubectl(['get', 'pod', '-n', k8s_namespace])
        
        for pod in pods.get('items', []):
            pod_name = pod['metadata']['name']
            
            # Check container statuses
            container_statuses = pod['status'].get('containerStatuses', [])
            for container in container_statuses:
                assert container.get('ready') is True, \
                    f"Container {container['name']} in pod {pod_name} is not ready"
    
    def test_api_endpoint_accessible(self, k8s_namespace: str):
        """Test that API endpoint is accessible via port-forward."""
        # This test would require port-forwarding setup
        # Skipping in automated tests
        pytest.skip("Requires manual port-forward setup")
    
    def test_hpa_created(self, k8s_namespace: str):
        """Test that HorizontalPodAutoscaler was created."""
        hpas = self.run_kubectl(['get', 'hpa', '-n', k8s_namespace])
        hpa_names = [hpa['metadata']['name'] for hpa in hpas.get('items', [])]
        
        assert 'synfinance-api-hpa' in hpa_names
    
    def test_pdb_created(self, k8s_namespace: str):
        """Test that PodDisruptionBudgets were created."""
        pdbs = self.run_kubectl(['get', 'pdb', '-n', k8s_namespace])
        pdb_names = [pdb['metadata']['name'] for pdb in pdbs.get('items', [])]
        
        assert 'synfinance-api-pdb' in pdb_names
        assert 'postgres-pdb' in pdb_names
        assert 'redis-pdb' in pdb_names
    
    def test_persistent_volumes_bound(self, k8s_namespace: str):
        """Test that PersistentVolumeClaims are bound."""
        pvcs = self.run_kubectl(['get', 'pvc', '-n', k8s_namespace])
        
        for pvc in pvcs.get('items', []):
            assert pvc['status']['phase'] == 'Bound', \
                f"PVC {pvc['metadata']['name']} is not bound"
    
    def test_resource_quotas_applied(self, k8s_namespace: str):
        """Test that ResourceQuota was applied."""
        quotas = self.run_kubectl(['get', 'resourcequota', '-n', k8s_namespace])
        quota_names = [q['metadata']['name'] for q in quotas.get('items', [])]
        
        assert 'synfinance-resource-quota' in quota_names
    
    def test_network_policies_created(self, k8s_namespace: str):
        """Test that NetworkPolicies were created."""
        netpols = self.run_kubectl(['get', 'networkpolicy', '-n', k8s_namespace])
        netpol_names = [np['metadata']['name'] for np in netpols.get('items', [])]
        
        assert 'synfinance-api-network-policy' in netpol_names
        assert 'postgres-network-policy' in netpol_names
        assert 'redis-network-policy' in netpol_names


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
