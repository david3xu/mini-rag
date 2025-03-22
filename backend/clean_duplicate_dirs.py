#!/usr/bin/env python
"""
Clean up script to fix duplicate backend directories.

This script will:
1. Check if duplicate backend directories exist
2. Move any data from backend/backend/* to backend/*
3. Remove the empty backend/backend directory and its subdirectories
"""

import os
import shutil
import sys

def clean_duplicates():
    """Clean up duplicate backend directories."""
    # Get the base directory (backend)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"Base directory: {base_dir}")
    
    # Check if nested backend directory exists
    nested_backend = os.path.join(base_dir, 'backend')
    if not os.path.exists(nested_backend):
        print("No duplicate backend directory found. No action needed.")
        return True
    
    print(f"Found duplicate backend directory at: {nested_backend}")
    
    # Check what's inside the nested backend
    contents = os.listdir(nested_backend)
    print(f"Contents of duplicate directory: {contents}")
    
    # Process each item in the nested backend
    for item in contents:
        src_path = os.path.join(nested_backend, item)
        dst_path = os.path.join(base_dir, item)
        
        if os.path.exists(dst_path):
            print(f"Warning: {dst_path} already exists. Will merge contents.")
            
            if os.path.isdir(src_path):
                # If it's a directory, merge contents
                merge_directories(src_path, dst_path)
            else:
                # If it's a file, backup existing and copy new
                backup_file = f"{dst_path}.backup"
                print(f"Backing up existing file to {backup_file}")
                shutil.copy2(dst_path, backup_file)
                shutil.copy2(src_path, dst_path)
        else:
            # Simply move the item if destination doesn't exist
            print(f"Moving {src_path} to {dst_path}")
            shutil.move(src_path, dst_path)
    
    # Remove any remaining empty subdirectories
    print("\nCleaning up empty directories...")
    try:
        cleanup_empty_dirs(nested_backend)
        print("Successfully removed duplicate backend directory.")
        return True
    except Exception as e:
        print(f"Error during cleanup: {str(e)}")
        return False

def merge_directories(src_dir, dst_dir):
    """
    Merge contents from src_dir into dst_dir.
    For conflicts, keep both by renaming the src file.
    """
    print(f"Merging directory {src_dir} into {dst_dir}")
    
    # Make sure destination exists
    os.makedirs(dst_dir, exist_ok=True)
    
    # Process each item in the source directory
    for item in os.listdir(src_dir):
        src_path = os.path.join(src_dir, item)
        dst_path = os.path.join(dst_dir, item)
        
        if os.path.exists(dst_path):
            if os.path.isdir(src_path):
                # Recursively merge subdirectories
                merge_directories(src_path, dst_path)
            else:
                # For files, keep both by renaming the source
                src_backup = f"{dst_path}.from_duplicate"
                print(f"File conflict: {dst_path} already exists. Copying source to {src_backup}")
                shutil.copy2(src_path, src_backup)
        else:
            # Move item if no conflict
            print(f"Moving {src_path} to {dst_path}")
            shutil.move(src_path, dst_path)

def cleanup_empty_dirs(directory):
    """Recursively remove empty directories starting from the leaves."""
    if not os.path.isdir(directory):
        return
    
    # First clean child directories
    for item in os.listdir(directory):
        path = os.path.join(directory, item)
        if os.path.isdir(path):
            cleanup_empty_dirs(path)
    
    # Now check if this directory is empty
    if not os.listdir(directory):
        print(f"Removing empty directory: {directory}")
        os.rmdir(directory)

def force_remove_backend():
    """Force remove the backend/backend directory (use with caution)."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    nested_backend = os.path.join(base_dir, 'backend')
    
    if not os.path.exists(nested_backend):
        print("No duplicate backend directory found.")
        return
    
    print(f"Force removing directory: {nested_backend}")
    try:
        shutil.rmtree(nested_backend)
        print("Successfully removed duplicate backend directory.")
    except Exception as e:
        print(f"Error removing directory: {str(e)}")

if __name__ == "__main__":
    print("=== Cleaning up duplicate backend directories ===")
    
    # Check command line args
    force_remove = len(sys.argv) > 1 and sys.argv[1] == "--force"
    
    if force_remove:
        print("WARNING: Using force removal mode!")
        force_remove_backend()
    else:
        # Run the cleanup
        success = clean_duplicates()
        
        # Print result
        if success:
            print("\n=== Cleanup completed successfully! ===")
        else:
            print("\n=== Cleanup failed! You can try again with --force to completely remove the duplicate directory. ===")
            sys.exit(1)
            
    print("\nRemember to run tests to verify everything works correctly after cleanup.")
    sys.exit(0) 