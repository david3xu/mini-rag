#!/usr/bin/env python
"""
Simple script to debug the health API tests.
This script demonstrates how the tests work with detailed output.
"""

import sys
import os
import psutil
from fastapi.testclient import TestClient

# Make sure we can import from the correct paths
sys.path.append('.')  

# Import the app and resources
from main import app
from app.api.health import ResourceMetrics

def test_health_endpoint():
    """Test the health endpoint with detailed output."""
    print("\n=== Testing /api/health endpoint ===")
    with TestClient(app) as client:
        print("Sending GET request to /api/health...")
        response = client.get("/api/health")
        print(f"Response status: {response.status_code}")
        
        # Print the response data in a more readable format
        data = response.json()
        print("\nResponse data:")
        print(f"  Status: {data['status']}")
        print(f"  Timestamp: {data['timestamp']}")
        print("  Services:")
        for service_name, service_info in data['services'].items():
            print(f"    {service_name}: {service_info['status']} - {service_info.get('details', 'No details')}")
        print("  Resources:")
        for resource_name, resource_value in data['resources'].items():
            print(f"    {resource_name}: {resource_value}")
        
        # Simple assertion to check we got a valid response
        if response.status_code == 200 and "status" in data:
            print("\n✓ Health endpoint test passed")
        else:
            print("\n✗ Health endpoint test failed")

def test_readiness_endpoint():
    """Test the readiness endpoint with detailed output."""
    print("\n=== Testing /api/health/readiness endpoint ===")
    with TestClient(app) as client:
        print("Sending GET request to /api/health/readiness...")
        response = client.get("/api/health/readiness")
        print(f"Response status: {response.status_code}")
        print(f"Response data: {response.json()}")
        
        # Simple assertion
        if response.status_code == 200 and "status" in response.json():
            print("\n✓ Readiness endpoint test passed")
        else:
            print("\n✗ Readiness endpoint test failed")

def test_resource_metrics():
    """Test the resource metrics with detailed output."""
    print("\n=== Testing resource metrics ===")
    print("Creating metrics using psutil...")
    # Create metrics directly
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
    
    print("\nCreated ResourceMetrics object:")
    print(f"  CPU usage: {metrics.cpu_percent}%")
    print(f"  Memory used: {metrics.memory_used_mb:.2f} MB")
    print(f"  Memory available: {metrics.memory_available_mb:.2f} MB")
    print(f"  Memory usage: {metrics.memory_percent}%")
    
    # Simple validations
    print("\nVerifying ResourceMetrics structure and values...")
    validations = []
    validations.append(hasattr(metrics, "cpu_percent"))
    validations.append(hasattr(metrics, "memory_used_mb"))
    validations.append(hasattr(metrics, "memory_available_mb"))
    validations.append(hasattr(metrics, "memory_percent"))
    validations.append(metrics.cpu_percent >= 0)
    validations.append(metrics.memory_used_mb > 0)
    validations.append(metrics.memory_available_mb > 0)
    validations.append(0 <= metrics.memory_percent <= 100)
    
    if all(validations):
        print("\n✓ Resource metrics test passed")
    else:
        print("\n✗ Resource metrics test failed")

def test_liveness_endpoint():
    """Test the liveness endpoint with detailed output."""
    print("\n=== Testing /api/health/liveness endpoint ===")
    with TestClient(app) as client:
        print("Sending GET request to /api/health/liveness...")
        response = client.get("/api/health/liveness")
        print(f"Response status: {response.status_code}")
        print(f"Response data: {response.json()}")
        
        # Simple assertion
        data = response.json()
        if response.status_code == 200 and "status" in data and data["status"] == "alive":
            print("\n✓ Liveness endpoint test passed")
        else:
            print("\n✗ Liveness endpoint test failed")

if __name__ == "__main__":
    print("=== Running Mini-RAG Health API Debug Tests ===")
    print("These tests verify the health endpoints with detailed output")
    
    # Run all tests
    test_health_endpoint()
    test_readiness_endpoint()
    test_resource_metrics()
    test_liveness_endpoint()
    
    print("\n=== All tests completed ===") 