#!/usr/bin/env python
"""
Script that runs both our search tests sequentially and logs results to files.
"""
import os
import sys
import time
import subprocess
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_test(test_script, output_file):
    """Run a test script and save output to a file.
    
    Args:
        test_script: Path to the test script
        output_file: File to save output to
        
    Returns:
        True if script completed, False otherwise
    """
    logger.info(f"Running test: {test_script}")
    logger.info(f"Saving output to: {output_file}")
    
    try:
        # Run the test script with a timeout of 30 seconds
        with open(output_file, 'w') as f:
            process = subprocess.Popen(
                [sys.executable, test_script],
                stdout=f,
                stderr=subprocess.STDOUT,
                text=True
            )
            
            # Wait for process to complete with a 30 second timeout
            try:
                process.wait(timeout=30)
                logger.info(f"Test {test_script} completed with return code {process.returncode}")
                return True
            except subprocess.TimeoutExpired:
                logger.error(f"Test {test_script} timed out after 30 seconds")
                process.kill()
                f.write("\nPROCESS TIMED OUT AFTER 30 SECONDS\n")
                return False
            
    except Exception as e:
        logger.error(f"Error running test {test_script}: {str(e)}")
        return False

def main():
    """Run all search tests."""
    logger.info("Starting search tests")
    
    # Run simple_direct_search.py
    direct_search_output = "direct_search_results.txt"
    direct_search_success = run_test("simple_direct_search.py", direct_search_output)
    
    # Run search_diagnostic.py
    diagnostic_output = "search_diagnostic_results.txt"
    diagnostic_success = run_test("search_diagnostic.py", diagnostic_output)
    
    # Print summary
    logger.info("\nTest Results Summary:")
    logger.info(f"Direct Search Test: {'COMPLETED' if direct_search_success else 'TIMED OUT'}")
    logger.info(f"Search Diagnostic Test: {'COMPLETED' if diagnostic_success else 'TIMED OUT'}")
    logger.info(f"\nSee {direct_search_output} and {diagnostic_output} for detailed output")

if __name__ == "__main__":
    main() 