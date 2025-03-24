#!/bin/bash
# Script to install optimized llama-cpp-python with BLAS acceleration

echo "Installing optimized llama-cpp-python with BLAS support"
echo "====================================================="

# Check if pip is available
if ! command -v pip &> /dev/null; then
    echo "Error: pip is not installed or not in PATH"
    exit 1
fi

# Check for OpenBLAS development package
echo "Checking for OpenBLAS development package..."
if ! dpkg -l | grep -q libopenblas-dev; then
    echo "Installing OpenBLAS development package..."
    apt-get update && apt-get install -y libopenblas-dev
fi

# Check for Python development package
if ! dpkg -l | grep -q python3-dev; then
    echo "Installing Python development package..."
    apt-get install -y python3-dev
fi

# Check for build tools
if ! command -v cmake &> /dev/null; then
    echo "Installing build tools (cmake)..."
    apt-get install -y cmake build-essential
fi

echo "Uninstalling current llama-cpp-python installation..."
pip uninstall -y llama-cpp-python

echo "Installing optimized llama-cpp-python with BLAS support..."
CMAKE_ARGS="-DLLAMA_BLAS=ON -DLLAMA_BLAS_VENDOR=OpenBLAS -DLLAMA_AVX2=ON -DLLAMA_F16C=ON -DLLAMA_FMA=ON" pip install llama-cpp-python

# Check installation success
if [ $? -eq 0 ]; then
    echo "====================================================="
    echo "Successfully installed optimized llama-cpp-python!"
    echo "Performance should be significantly improved."
    echo 
    echo "Additional optimizations enabled:"
    echo "- BLAS matrix operations (OpenBLAS)"
    echo "- AVX2 instruction set"
    echo "- F16C instruction set"
    echo "- FMA instruction set"
    echo "====================================================="
else
    echo "====================================================="
    echo "Error: Failed to install optimized llama-cpp-python."
    echo "Please check the error messages above."
    echo "====================================================="
    exit 1
fi

# Optional: Add more specific CPU optimizations
if grep -q avx512 /proc/cpuinfo; then
    echo "Your CPU supports AVX-512 instructions!"
    echo "For even better performance, you can reinstall with:"
    echo 'CMAKE_ARGS="-DLLAMA_BLAS=ON -DLLAMA_BLAS_VENDOR=OpenBLAS -DLLAMA_AVX512=ON" pip install llama-cpp-python'
fi

exit 0 