#!/bin/bash
# Run all tests and comparisons for SDCN-Spatial

# Set up environment
echo "Setting up environment..."
# Uncomment the following lines if you need to set up a virtual environment
# python -m venv venv
# source venv/bin/activate
# pip install -r requirements.txt

# Run tests
echo "Running tests for SDCN_Spatial model..."
python test_sdcn_spatial.py

# Run comparison on a small dataset (reut)
echo "Running comparison on reut dataset..."
python compare_sdcn_models.py --name reut --k 3 --n_clusters 4 --n_z 10 --lr 1e-4 --epochs 50

# Optional: Run comparison on other datasets
# Uncomment the following lines to run on other datasets
# echo "Running comparison on usps dataset..."
# python compare_sdcn_models.py --name usps --k 3 --n_clusters 10 --n_z 10 --lr 1e-3 --epochs 50
# 
# echo "Running comparison on hhar dataset..."
# python compare_sdcn_models.py --name hhar --k 5 --n_clusters 6 --n_z 10 --lr 1e-3 --epochs 50

echo "All tests and comparisons completed!"
echo "Check the results in the CSV files and PNG images."