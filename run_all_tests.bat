@echo off
REM Run all tests and comparisons for SDCN-Spatial

REM Set up environment
echo Setting up environment...
REM Uncomment the following lines if you need to set up a virtual environment
REM python -m venv venv
REM call venv\Scripts\activate.bat
REM pip install -r requirements.txt

REM Run tests
echo Running tests for SDCN_Spatial model...
python test_sdcn_spatial.py

REM Run comparison on a small dataset (reut)
echo Running comparison on reut dataset...
python compare_sdcn_models.py --name reut --k 3 --n_clusters 4 --n_z 10 --lr 1e-4 --epochs 50

REM Optional: Run comparison on other datasets
REM Uncomment the following lines to run on other datasets
REM echo Running comparison on usps dataset...
REM python compare_sdcn_models.py --name usps --k 3 --n_clusters 10 --n_z 10 --lr 1e-3 --epochs 50
REM 
REM echo Running comparison on hhar dataset...
REM python compare_sdcn_models.py --name hhar --k 5 --n_clusters 6 --n_z 10 --lr 1e-3 --epochs 50

echo All tests and comparisons completed!
echo Check the results in the CSV files and PNG images.
pause