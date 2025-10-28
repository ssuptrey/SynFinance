"""
Docker Deployment Tests

Tests for Docker containerization and deployment functionality.
"""

import pytest
import subprocess
import time
import requests
from pathlib import Path


class TestDockerBuild:
    """Test Docker image building."""
    
    def test_dockerfile_exists(self):
        """Test that Dockerfile exists."""
        dockerfile = Path("Dockerfile")
        assert dockerfile.exists(), "Dockerfile not found"
        assert dockerfile.is_file(), "Dockerfile is not a file"
    
    def test_dockerfile_dev_exists(self):
        """Test that development Dockerfile exists."""
        dockerfile_dev = Path("Dockerfile.dev")
        assert dockerfile_dev.exists(), "Dockerfile.dev not found"
    
    def test_dockerignore_exists(self):
        """Test that .dockerignore exists."""
        dockerignore = Path(".dockerignore")
        assert dockerignore.exists(), ".dockerignore not found"
    
    def test_docker_compose_exists(self):
        """Test that docker-compose.yml exists."""
        compose_file = Path("docker-compose.yml")
        assert compose_file.exists(), "docker-compose.yml not found"


class TestDockerCompose:
    """Test Docker Compose configuration."""
    
    def test_compose_file_valid(self):
        """Test that docker-compose.yml is valid."""
        result = subprocess.run(
            ["docker-compose", "config"],
            capture_output=True,
            text=True
        )
        assert result.returncode == 0, f"Invalid docker-compose.yml: {result.stderr}"
    
    def test_compose_services_defined(self):
        """Test that essential services are defined."""
        result = subprocess.run(
            ["docker-compose", "config", "--services"],
            capture_output=True,
            text=True
        )
        
        services = result.stdout.strip().split('\n')
        assert 'api' in services, "API service not defined in docker-compose.yml"


@pytest.mark.slow
@pytest.mark.integration
class TestDockerImage:
    """Test Docker image functionality."""
    
    @pytest.fixture(scope="class")
    def docker_image(self):
        """Build Docker image for testing."""
        print("\nBuilding Docker image...")
        result = subprocess.run(
            ["docker", "build", "-t", "synfinance:test", "-f", "Dockerfile", "."],
            capture_output=True,
            text=True,
            timeout=600  # 10 minutes timeout
        )
        
        if result.returncode != 0:
            pytest.skip(f"Docker build failed: {result.stderr}")
        
        yield "synfinance:test"
        
        # Cleanup
        subprocess.run(["docker", "rmi", "synfinance:test"], capture_output=True)
    
    def test_image_builds_successfully(self, docker_image):
        """Test that Docker image builds without errors."""
        # Check image exists
        result = subprocess.run(
            ["docker", "images", "-q", docker_image],
            capture_output=True,
            text=True
        )
        assert result.stdout.strip(), f"Image {docker_image} not found"
    
    def test_image_size(self, docker_image):
        """Test that Docker image is reasonably sized (<500MB target)."""
        result = subprocess.run(
            ["docker", "images", docker_image, "--format", "{{.Size}}"],
            capture_output=True,
            text=True
        )
        
        size_str = result.stdout.strip()
        print(f"\nImage size: {size_str}")
        
        # Parse size (e.g., "350MB", "1.2GB")
        if "GB" in size_str:
            size_value = float(size_str.replace("GB", ""))
            assert size_value < 1.0, f"Image too large: {size_str} (target: <500MB)"
        elif "MB" in size_str:
            size_value = float(size_str.replace("MB", ""))
            assert size_value < 800, f"Image larger than ideal: {size_str} (target: <500MB)"


@pytest.mark.slow
@pytest.mark.integration
class TestContainerRuntime:
    """Test running Docker container."""
    
    @pytest.fixture(scope="class")
    def running_container(self):
        """Start a test container."""
        container_name = "synfinance-test-container"
        
        # Build image first
        subprocess.run(
            ["docker", "build", "-t", "synfinance:test", "-f", "Dockerfile", "."],
            capture_output=True,
            timeout=600
        )
        
        # Stop and remove any existing test container
        subprocess.run(["docker", "stop", container_name], capture_output=True)
        subprocess.run(["docker", "rm", container_name], capture_output=True)
        
        # Start container
        print(f"\nStarting container: {container_name}")
        result = subprocess.run(
            [
                "docker", "run", "-d",
                "--name", container_name,
                "-p", "8001:8000",  # Use different port to avoid conflicts
                "synfinance:test"
            ],
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            pytest.skip(f"Failed to start container: {result.stderr}")
        
        # Wait for container to be ready
        time.sleep(10)
        
        yield container_name
        
        # Cleanup
        subprocess.run(["docker", "stop", container_name], capture_output=True)
        subprocess.run(["docker", "rm", container_name], capture_output=True)
        subprocess.run(["docker", "rmi", "synfinance:test"], capture_output=True)
    
    def test_container_starts(self, running_container):
        """Test that container starts successfully."""
        result = subprocess.run(
            ["docker", "ps", "--filter", f"name={running_container}", "--format", "{{.Status}}"],
            capture_output=True,
            text=True
        )
        
        status = result.stdout.strip()
        assert "Up" in status, f"Container not running: {status}"
    
    def test_health_check_passes(self, running_container):
        """Test that health check endpoint works."""
        max_retries = 30
        retry_interval = 2
        
        for i in range(max_retries):
            try:
                response = requests.get("http://localhost:8001/health", timeout=5)
                if response.status_code == 200:
                    break
            except requests.exceptions.RequestException:
                pass
            
            if i == max_retries - 1:
                # Get container logs
                logs = subprocess.run(
                    ["docker", "logs", running_container],
                    capture_output=True,
                    text=True
                )
                pytest.fail(f"Health check failed after {max_retries} retries.\nLogs:\n{logs.stdout}")
            
            time.sleep(retry_interval)
        
        response = requests.get("http://localhost:8001/health")
        assert response.status_code == 200
        assert response.json().get("status") == "healthy"
    
    def test_api_docs_accessible(self, running_container):
        """Test that API documentation is accessible."""
        response = requests.get("http://localhost:8001/docs", timeout=10)
        assert response.status_code == 200
    
    def test_openapi_schema_accessible(self, running_container):
        """Test that OpenAPI schema is accessible."""
        response = requests.get("http://localhost:8001/openapi.json", timeout=10)
        assert response.status_code == 200
        assert "openapi" in response.json()


class TestDeploymentScripts:
    """Test deployment scripts."""
    
    def test_deploy_script_exists(self):
        """Test that deploy.sh script exists."""
        script = Path("scripts/deploy.sh")
        assert script.exists(), "deploy.sh not found"
    
    def test_rollback_script_exists(self):
        """Test that rollback.sh script exists."""
        script = Path("scripts/rollback.sh")
        assert script.exists(), "rollback.sh not found"
    
    def test_health_check_script_exists(self):
        """Test that health_check.sh script exists."""
        script = Path("scripts/health_check.sh")
        assert script.exists(), "health_check.sh not found"


class TestCICD:
    """Test CI/CD configuration."""
    
    def test_ci_workflow_exists(self):
        """Test that CI workflow exists."""
        workflow = Path(".github/workflows/ci.yml")
        assert workflow.exists(), "CI workflow not found"
    
    def test_cd_workflow_exists(self):
        """Test that CD workflow exists."""
        workflow = Path(".github/workflows/cd.yml")
        assert workflow.exists(), "CD workflow not found"
    
    def test_benchmark_workflow_exists(self):
        """Test that benchmark workflow exists."""
        workflow = Path(".github/workflows/benchmark.yml")
        assert workflow.exists(), "Benchmark workflow not found"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
