"""ANN-UTADIS building blocks.

Monotonic, interpretable PyTorch layers used to implement the additive
utility function U(a) = Σ_j w_j u_j(g_j(a)) and the ordinal threshold
mechanism that assigns alternatives to preference-ordered classes.
"""

from .criterion_layer_combine import CriterionLayerCombine
from .criterion_layer_spread import CriterionLayerSpread
from .leaky_hard_sigmoid import LeakyHardSigmoid
from .monotonic_layer import MonotonicLayer
from .norm_layer import NormLayer
from .threshold_layer import OrdinalThresholdLayer, ThresholdLayer
from .uta import Uta

__all__ = [
    "CriterionLayerCombine",
    "CriterionLayerSpread",
    "LeakyHardSigmoid",
    "MonotonicLayer",
    "NormLayer",
    "OrdinalThresholdLayer",
    "ThresholdLayer",
    "Uta",
]
