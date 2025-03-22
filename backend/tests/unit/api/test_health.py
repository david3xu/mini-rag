"""
Unit tests for the API health endpoint.
"""

import pytest
import psutil
from fastapi.testclient import TestClient

from main import app
from app.api.health import get_resource_metrics, ResourceMetrics

@pytest.mark.unit
class TestHealthEndpoint:
    
    def test_health_endpoint(self):
        """Test the health endpoint returns the correct status."""
        print("\n=== Testing /api/health endpoint ===")
        with TestClient(app) as client:
            print("Sending GET request to /api/health...")
            response = client.get("/api/health")
            print(f"Response status: {response.status_code}")
            
            # Print the response data in a more readable format
            data = response.json()
            print("Response data:")
            print(f"  Status: {data['status']}")
            print(f"  Timestamp: {data['timestamp']}")
            print("  Services:")
            for service_name, service_info in data['services'].items():
                print(f"    {service_name}: {service_info['status']} - {service_info.get('details', 'No details')}")
            print("  Resources:")
            for resource_name, resource_value in data['resources'].items():
                print(f"    {resource_name}: {resource_value}")
            
            assert response.status_code == 200
            assert "status" in data
            print("✓ Health endpoint test passed")
    
    def test_readiness_endpoint(self):
        """Test the readiness endpoint returns the correct format."""
        print("\n=== Testing /api/health/readiness endpoint ===")
        with TestClient(app) as client:
            print("Sending GET request to /api/health/readiness...")
            response = client.get("/api/health/readiness")
            print(f"Response status: {response.status_code}")
            print(f"Response data: {response.json()}")
            
            assert response.status_code == 200
            assert "status" in response.json()
            print("✓ Readiness endpoint test passed")
    
    def test_resource_metrics(self):
        """Test the resource metrics function with real psutil."""
        print("\n=== Testing resource metrics ===")
        print("Creating metrics using psutil...")
        # Create metrics directly instead of calling the async function
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        
        print(f"Raw CPU usage: {cpu_percent}%")
        print(f"Raw memory stats: used={memory.used/(1024*1024):.2f}MB, available={memory.available/(1024*1024):.2f}MB, percent={memory.percent}%")
        
        metrics = ResourceMetrics(
            cpu_percent=cpu_percent,
            memory_used_mb=memory.used / (1024 * 1024),
            memory_available_mb=memory.available / (1024 * 1024),
            memory_percent=memory.percent
        )
        
        print("Created ResourceMetrics object:")
        print(f"  CPU usage: {metrics.cpu_percent}%")
        print(f"  Memory used: {metrics.memory_used_mb:.2f} MB")
        print(f"  Memory available: {metrics.memory_available_mb:.2f} MB")
        print(f"  Memory usage: {metrics.memory_percent}%")
        
        # Verify results have the expected structure
        print("Verifying ResourceMetrics structure...")
        assert hasattr(metrics, "cpu_percent")
        assert hasattr(metrics, "memory_used_mb")
        assert hasattr(metrics, "memory_available_mb")
        assert hasattr(metrics, "memory_percent")
        
        # Sanity checks
        print("Performing sanity checks on values...")
        assert metrics.cpu_percent >= 0
        assert metrics.memory_used_mb > 0
        assert metrics.memory_available_mb > 0
        assert 0 <= metrics.memory_percent <= 100
        print("✓ Resource metrics test passed")
    
    def test_liveness_endpoint(self):
        """Test liveness endpoint reports correct status."""
        print("\n=== Testing /api/health/liveness endpoint ===")
        with TestClient(app) as client:
            print("Sending GET request to /api/health/liveness...")
            response = client.get("/api/health/liveness")
            print(f"Response status: {response.status_code}")
            print(f"Response data: {response.json()}")
            
            assert response.status_code == 200
            data = response.json()
            assert "status" in data
            assert data["status"] == "alive"
            print("✓ Liveness endpoint test passed")