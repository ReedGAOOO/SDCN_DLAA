# Experiments

Runnable scripts that call the main pipeline (`sdcn_dlaa_NEW.py`) on preprocessed datasets.

## Common entrypoints

- `experiments/test_sdcn_dlaa_NEW_sparse_KNN.py`: run on KNN-sparsified graphs
- `experiments/test_sdcn_dlaa_NEW_sparse_threshold.py`: run on threshold-sparsified graphs
- `experiments/test_sdcn_dlaa_NEW_sparse.py`: generic sparse run

## Sweeps

- `experiments/sweeps/run_batch_test.py`: quick batch runs over `--heads`
- `experiments/sweeps/run_multiple_heads.py`: simple multi-head launcher
