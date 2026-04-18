#!/bin/bash
set -e

echo "=================================="
echo " TensorScript 7-Phase Compilation"
echo "=================================="

# Check and install requirements
echo "\n[System] Checking dependencies..."
python3 -m pip install -r requirements.txt > /dev/null 2>&1 || echo "Warning: Could not install pip dependencies, ensure torch is installed."

# Run the 7-phase Compiler
echo "\n[System] Running Compiler..."
python3 compiler.py example.ts generated_model.py

# Run the generated module
echo "\n=================================="
echo " Executing Generated PyTorch Code"
echo "=================================="
python3 generated_model.py
