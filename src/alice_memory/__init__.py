"""Independent parametric memory prototype; Stage 1 stays separately reviewable."""

from .bridge import ParametricBridge
from .matrix import AssociativeMemory

__all__ = ["AssociativeMemory", "ParametricBridge"]
