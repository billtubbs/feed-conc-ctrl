"""Anti-windup logic for the PID controller integral term.

This module provides functionality to prevent integrator windup when the
control signal is saturated.
"""

from enum import Enum


class WindupMode(Enum):
    """Windup mode enumeration."""

    NONE = "none"
    UPPER = "upper"
    LOWER = "lower"
    BOTH = "both"


def anti_windup(Dui, windup):
    """Apply anti-windup logic to the integral increment.

    Prevents the integrator from accumulating error when the control
    signal is saturated by restricting integration in the direction
    of saturation.

    Args:
        Dui: Integral control signal increment
        windup: Windup status indicating which limit(s) are active.
                Can be WindupMode enum, string ("upper", "lower",
                "both", "none"), or None/False for no windup.

    Returns:
        float: Modified integral increment that respects saturation
               constraints
    """
    # Handle different input types
    if windup is None or windup is False:
        return Dui

    if isinstance(windup, str):
        windup = windup.lower()
    elif isinstance(windup, WindupMode):
        windup = windup.value

    # Prevent increase, decrease, or both
    if windup in ("both", "lower"):
        Dui = max(Dui, 0)

    if windup in ("both", "upper"):
        Dui = min(Dui, 0)

    return Dui
