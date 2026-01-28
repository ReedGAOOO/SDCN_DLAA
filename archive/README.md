# Archive

This folder contains **archived experimental variants** (AMP / hetero / hidden-size sweeps) that are not used by the default pipeline.

## Layout

- `archive/models/`: archived model/training implementations
- `archive/experiments/`: archived runnable scripts that depend on `archive/models/`

## Note

The current default pipeline uses:
- `sdcn_dlaa_NEW.py` (main training entry)
- `DLAA_NEW.py` (SpatialConv + variants)
- runnable entry scripts under `experiments/`
