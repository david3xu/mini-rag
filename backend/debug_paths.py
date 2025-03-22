#!/usr/bin/env python
"""
Debugging script to identify path issues with the embedding model.
"""

import os
import sys
from pathlib import Path

# Ensure we can import from the right place
sys.path.append('.')

from config import settings

def resolve_path(path):
    """Attempts to resolve a path in multiple ways to find the file"""
    print(f"\nAttempting to resolve path: {path}")
    
    # Original path
    print(f"Original path exists? {os.path.exists(path)}")
    
    # Absolute path
    abs_path = os.path.abspath(path)
    print(f"Absolute path: {abs_path}")
    print(f"Absolute path exists? {os.path.exists(abs_path)}")
    
    # Relative to script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    rel_to_script = os.path.join(script_dir, os.path.relpath(path) if path.startswith('./') else path)
    print(f"Relative to script: {rel_to_script}")
    print(f"Relative to script exists? {os.path.exists(rel_to_script)}")
    
    # Relative to working directory but without ./ prefix
    if path.startswith('./'):
        no_dot_prefix = path[2:]
        print(f"Without dot prefix: {no_dot_prefix}")
        print(f"Without dot prefix exists? {os.path.exists(no_dot_prefix)}")
        
        # Relative to script without dot prefix
        rel_no_dot = os.path.join(script_dir, no_dot_prefix)
        print(f"Relative to script without dot prefix: {rel_no_dot}")
        print(f"Relative to script without dot prefix exists? {os.path.exists(rel_no_dot)}")
    
    # Path components
    for i, comp in enumerate(path.split('/')):
        print(f"  Path component {i}: '{comp}'")
    
    # Find all possible paths
    if 'models' in path and 'embeddings' in path:
        # Try to find any path that contains models/embeddings
        base_component = 'models/embeddings'
        print(f"\nSearching for any path containing '{base_component}'...")
        
        for root, dirs, files in os.walk('.'):
            if 'models' in dirs:
                models_dir = os.path.join(root, 'models')
                if os.path.exists(models_dir) and 'embeddings' in os.listdir(models_dir):
                    embeddings_dir = os.path.join(models_dir, 'embeddings')
                    print(f"Found potential match: {embeddings_dir}")
                    print(f"Contents: {os.listdir(embeddings_dir)}")

def list_directory_contents(directory):
    """Lists all files and directories recursively"""
    print(f"\nListing contents of {directory}:")
    try:
        for root, dirs, files in os.walk(directory):
            level = root.replace(directory, '').count(os.sep)
            indent = ' ' * 4 * level
            print(f"{indent}{os.path.basename(root)}/")
            sub_indent = ' ' * 4 * (level + 1)
            for f in files:
                print(f"{sub_indent}{f}")
    except Exception as e:
        print(f"Error listing directory: {str(e)}")

if __name__ == "__main__":
    print("=== Path Resolution Debug Tool ===")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Script location: {os.path.dirname(os.path.abspath(__file__))}")
    print(f"Python path: {sys.path}")
    
    # Check embedding model path
    print("\n--- Embedding Model Path ---")
    print(f"Settings path: {settings.EMBEDDING_MODEL}")
    resolve_path(settings.EMBEDDING_MODEL)
    
    # List model directories 
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    if os.path.exists(model_dir):
        list_directory_contents(model_dir)
    
    # Try to find huggingface transformers cache 
    print("\n--- Huggingface Cache Location ---")
    hf_home = os.environ.get('HF_HOME')
    if hf_home:
        print(f"HF_HOME: {hf_home}")
    else:
        print("HF_HOME not set")
    
    default_cache = os.path.expanduser("~/.cache/huggingface")
    print(f"Default cache path: {default_cache}")
    print(f"Exists? {os.path.exists(default_cache)}")
    
    # Print system PATH
    print("\n--- System PATH ---")
    for i, path in enumerate(os.environ.get('PATH', '').split(os.pathsep)):
        print(f"{i}: {path}") 