# Feed Concentration Control

Model process plant feed concentration blending tank operations to minimize variations in plant feed stream due to unpredictable disturbances in the concentration of delivered quantities.

## Installation

```bash
pip install -e .[dev]
```

## Description

This package provides tools for modeling and controlling feed concentration in process plant blending tanks, helping to minimize variations in the plant feed stream caused by unpredictable disturbances.

## Contents

### Primary Notebooks

- `buffer-tanks.ipynb`
	- Main notebook for feed concentration smoothing with buffer-tank dynamics.
	- Includes examples using transfer-function analysis and dynamic simulation.
- `blending.ipynb`
	- Main notebook for multi-stream blending and ratio-control scenarios.
	- Covers baseline blending behavior and controlled blending with a buffer tank.

### Utility Modules Used by These Notebooks

- `bounded_random_walk.py`
	- Generates bounded random-walk disturbance sequences used as realistic feed disturbances.
- `src/feed_conc_ctrl/plot_utils.py`
	- Provides standardized time-series plotting utilities used throughout the notebooks.
- `src/feed_conc_ctrl/models.py`
	- Defines process models used in simulations (for example, mixer and tank models).
- `python_pid` (from `python-pid` dependency)
	- Provides the PID controller used in ratio-trim control examples.

### Other Notebooks (Work In Progress)

The remaining notebooks include exploratory and in-progress work on 2-tank and 4-tank optimal blending/control studies. They use CasADi-based modeling and simulation tools (including related dynamic model and optimization workflows) and are not yet fully consolidated or finalized.

## License

MIT License - see LICENSE file for details.
