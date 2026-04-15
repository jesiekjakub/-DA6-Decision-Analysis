import torch.nn as nn

from .criterion_layer_combine import CriterionLayerCombine
from .criterion_layer_spread import CriterionLayerSpread
from .leaky_hard_sigmoid import LeakyHardSigmoid


class MonotonicLayer(nn.Sequential):
    """
    Monotonic block producing one marginal utility value per criterion.

    The block stacks three stages:
    spread → LeakyHardSigmoid activation → combine.

    Because the LeakyHardSigmoid is monotonic and the combine weights are
    non-negative, the whole block is guaranteed monotonic with respect to
    every criterion.

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
        self.criterion_layer_spread = CriterionLayerSpread(
            num_criteria, num_hidden_components, **kwargs
        )
        self.activation_function = LeakyHardSigmoid(slope=slope, **kwargs)
        self.criterion_layer_combine = CriterionLayerCombine(
            num_criteria, num_hidden_components, **kwargs
        )

    def set_slope(self, val: float) -> None:
        self.activation_function.set_slope(val)

    def forward(self, x):
        for module in self._modules.values():
            x = module(x)
        return x
