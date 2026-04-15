import torch
import torch.nn as nn

from .monotonic_layer import MonotonicLayer


class Uta(nn.Module):
    """
    ANN-UTADIS value function U(a) = Σ_j u_j(g_j(a)).

    The network is a single monotonic block followed by a sum over criteria.
    The resulting scalar is the comprehensive utility of the alternative —
    the output of this module is *unnormalized* and *unthresholded*; wrap it
    in ``NormLayer`` and an ``OrdinalThresholdLayer`` for full end-to-end
    sorting.

    Args:
        num_criteria: Number of criteria.
        num_hidden_components: Number L of components per criterion.
        slope: Initial slope of the LeakyHardSigmoid.
    """

    def __init__(
        self,
        num_criteria: int,
        num_hidden_components: int,
        slope: float = 0.01,
        **kwargs,
    ):
        super().__init__()
        self.monotonic_layer = MonotonicLayer(
            num_criteria, num_hidden_components, slope, **kwargs
        )

    def set_slope(self, val: float) -> None:
        self.monotonic_layer.set_slope(val)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = self.monotonic_layer(input)
        return x.sum(1)
