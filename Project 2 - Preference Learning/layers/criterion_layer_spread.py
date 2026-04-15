from typing import Tuple

import torch
import torch.nn as nn


class CriterionLayerSpread(nn.Module):
    """
    Spread each criterion value into ``num_hidden_components`` trainable
    affine projections, one per hidden component.

    This is the first stage of the monotonic block: for each criterion
    g_j(a) the layer outputs L values of the form w_{l,j} * (g_j(a) + b_{l,j})
    that are later passed through the LeakyHardSigmoid activation to form
    the piecewise-linear marginal utility function.

    Args:
        num_criteria: Number of criteria.
        num_hidden_components: Number L of components per criterion.
        input_range: Range of input values (used for bias initialization).
        normalize_bias: Whether to clamp the bias into ``input_range``.
    """

    def __init__(
        self,
        num_criteria: int,
        num_hidden_components: int,
        input_range: Tuple[float, float] = (0, 1),
        normalize_bias: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        self.num_criteria = num_criteria
        input_range = (-input_range[0], -input_range[1])
        self.max_bias = max(input_range)
        self.min_bias = min(input_range)
        self.normalize_bias = normalize_bias
        self.bias = nn.Parameter(torch.FloatTensor(num_hidden_components, num_criteria))
        self.weight = nn.Parameter(
            torch.FloatTensor(num_hidden_components, num_criteria)
        )
        self.reset_parameters()
        self.min_w = 0

    def reset_parameters(self) -> None:
        nn.init.uniform_(self.weight, 1, 10.0)
        nn.init.uniform_(self.bias, self.min_bias, self.max_bias)

    def compute_bias(self) -> torch.Tensor:
        if self.normalize_bias:
            return torch.clamp(self.bias, self.min_bias, self.max_bias)
        return self.bias

    def compute_weight(self) -> torch.Tensor:
        with torch.no_grad():
            self.weight.data[self.weight.data < 0] = self.min_w
        return self.weight

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input.view(-1, 1, self.num_criteria)
        return (x + self.compute_bias()) * self.compute_weight()
