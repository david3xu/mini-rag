"""
Unit tests for the Embedding service.
"""

import pytest
import time
import gc
import numpy as np

from app.services.embeddings import EmbeddingService, embeddings_service

@pytest.mark.unit
class TestEmbeddingService:
    
    def test_lazy_loading(self):
        """Test that the model is only loaded when accessed."""
        print("\n=== Testing lazy loading of embedding model ===")
        service = EmbeddingService()
        
        # Model should not be loaded initially
        print("Checking initial model state (should be None)...")
        assert service._model is None
        print("✓ Model is initially None as expected")
        
        # Access model property to trigger loading
        print("Accessing model property to trigger loading...")
        model = service.model
        print(f"Model loaded: {type(model).__name__}")
        
        # Model should now be loaded
        assert service._model is not None
        print("✓ Model is now loaded")
        
        # No need to force unloading as we're testing real behavior
        # Just manually clean up for the test
        print("Cleaning up model resources...")
        service._model = None
        gc.collect()
        print("✓ Resources cleaned up")
    
    def test_model_loading_timestamp(self):
        """Test that accessing the model updates the last_used_time."""
        print("\n=== Testing model loading timestamp update ===")
        service = EmbeddingService()
        
        # Record time before loading
        print("Recording time before loading model...")
        before_time = time.time()
        time.sleep(0.1)  # Small delay to ensure time difference
        
        # Access model property to trigger loading
        print("Loading model...")
        _ = service.model
        print(f"Model loaded at timestamp: {service.last_used_time}")
        
        # Verify last_used_time was updated
        print(f"Time before loading: {before_time}")
        print(f"Last used time: {service.last_used_time}")
        print(f"Time difference: {service.last_used_time - before_time} seconds")
        assert service.last_used_time > before_time
        print("✓ Last used time was updated correctly")
        
        # Clean up
        print("Cleaning up model resources...")
        service._model = None
        gc.collect()
        print("✓ Resources cleaned up")
    
    def test_generate_embeddings_batching(self):
        """Test that generate_embeddings uses batching correctly."""
        print("\n=== Testing embedding generation with batching ===")
        service = EmbeddingService()
        
        # Generate test texts
        print("Creating test documents...")
        texts = [f"Document {i}" for i in range(5)]
        print(f"Created {len(texts)} test documents")
        
        # Process with batch size 2
        print("Generating embeddings with batch size 2...")
        embeddings = service.generate_embeddings(texts, batch_size=2)
        
        # Verify results
        print(f"Generated {len(embeddings)} embeddings")
        assert len(embeddings) == 5
        
        # Each embedding should be a vector with expected dimensions
        if embeddings:
            print(f"Embedding dimensions: {len(embeddings[0])}")
            print(f"Sample embedding values (first 5): {embeddings[0][:5]}")
        
        for emb in embeddings:
            assert isinstance(emb, list)
            # Most embedding models have hundreds of dimensions
            assert len(emb) > 100
        print("✓ All embeddings have correct dimensions")
        
        # No need to artificially unload the model
        # Just manually clean up for the test
        print("Cleaning up model resources...")
        service._model = None
        gc.collect()
        print("✓ Resources cleaned up") 