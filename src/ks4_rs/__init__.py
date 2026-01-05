"""
KS4 Red-Sequence Cluster Detection Pipeline
==========================================

This package provides a modular implementation of the red-sequence-based
galaxy cluster detection pipeline developed for the KS4 survey.

Main Components
---------------
- pipeline      : High-level orchestration of the 7-step RS pipeline
- runners       : Execution strategies (field-wise / redshift-wise)
- steps         : Core algorithmic steps (RS selection → cluster detection)
- classification: Star–galaxy separation utilities
- consolidation : Post-processing and candidate merging
- config        : Configuration loader
- utils         : Shared helper functions

This package is designed to expose *methods*, not survey-specific execution
environments (e.g., BC03 simulations, validation scripts).
"""

__author__ = "Bomi Park"
__email__ = "sadalsuud14@gmail.com"
__version__ = "0.1.0"

# ---- Public API -------------------------------------------------------------

from ks4_rs.pipeline import run_pipeline
from ks4_rs.runners import (
    run_single_field_all_z,
    run_single_z_all_fields,
)

from ks4_rs.config import load_config

from ks4_rs.consolidation import consolidate_clusters

# Optional convenience imports (safe, non-heavy)
from ks4_rs.utils import get_optimal_color

__all__ = [
    "run_pipeline",
    "run_single_field_all_z",
    "run_single_z_all_fields",
    "load_config",
    "consolidate_clusters",
    "get_optimal_color",
]
