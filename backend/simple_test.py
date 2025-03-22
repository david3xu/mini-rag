#!/usr/bin/env python
"""
Simple test script to verify the embedding model works.
"""

import os
import sys
import numpy as np
from sentence_transformers import SentenceTransformer

def test_embedding_model():
    """Test if the embedding model can be loaded and used."""
    print("=== Testing Embedding Model ===")
    
    # First, print current directory and available files
    current_dir = os.getcwd()
    print(f"Current working directory: {current_dir}")
    
    # Find the embedding model directory
    embedding_path = os.path.join(current_dir, "models", "embeddings", "all-MiniLM-L6-v2")
    print(f"Looking for embedding model at: {embedding_path}")
    print(f"Path exists: {os.path.exists(embedding_path)}")
    
    if os.path.exists(embedding_path):
        print("Contents of embedding model directory:")
        for item in os.listdir(embedding_path):
            if os.path.isdir(os.path.join(embedding_path, item)):
                print(f"  DIR: {item}")
            else:
                print(f"  FILE: {item}")
    
    # Try to load the model
    print("\nAttempting to load the embedding model...")
    try:
        model = SentenceTransformer(embedding_path)
        print("✓ Model loaded successfully!")
        
        # Test the model with some sample text
        print("\nGenerating embeddings for sample text...")
        samples = [
            "This is a test sentence for embeddings.",
            "Another example to verify the model works."
        ]
        
        embeddings = model.encode(samples)
        print(f"✓ Generated {len(embeddings)} embeddings!")
        print(f"  First embedding shape: {embeddings[0].shape}")
        print(f"  First few values: {embeddings[0][:5].tolist()}")
        
        # Test similarity between embeddings
        similarity = np.dot(embeddings[0], embeddings[1]) / (np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1]))
        print(f"\nSimilarity between samples: {similarity:.4f}")
        
        return True
    except Exception as e:
        print(f"✗ Error loading model: {str(e)}")
        
        # Try loading from HuggingFace hub
        print("\nAttempting to load from HuggingFace hub...")
        try:
            model = SentenceTransformer("all-MiniLM-L6-v2")
            print("✓ Model loaded from HuggingFace hub successfully!")
            return True
        except Exception as hub_error:
            print(f"✗ Error loading from hub: {str(hub_error)}")
            return False

if __name__ == "__main__":
    print("=== Simple Embedding Model Test ===")
    
    # Go to backend directory if not already there
    if not os.getcwd().endswith('backend'):
        os.chdir('backend')
        print(f"Changed directory to: {os.getcwd()}")
    
    # Run the test
    success = test_embedding_model()
    
    # Print result
    if success:
        print("\n=== Test Completed Successfully! ===")
        sys.exit(0)
    else:
        print("\n=== Test Failed! ===")
        sys.exit(1) 