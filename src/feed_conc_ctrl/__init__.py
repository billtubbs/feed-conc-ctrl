"""
Feed Concentration Control Package

Model process plant feed concentration blending tank operations to minimize
variations in plant feed stream due to unpredictable disturbances.
"""

from feed_conc_ctrl.models import MixingTankModelCT, MixingTankModelDT

__version__ = "0.1.0"

__all__ = ["MixingTankModelCT", "MixingTankModelDT"]
