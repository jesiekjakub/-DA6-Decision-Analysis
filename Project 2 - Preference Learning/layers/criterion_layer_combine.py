import torch
import torch.nn as nn


class CriterionLayerCombine(nn.Module):
    """
    Aggregate the ``num_hidden_components`` outputs of the activation stage
    back into a single value per criterion, using non-negative weights.

    This is the third stage of the monotonic block and produces the marginal
    utility u_j(g_j(a)) for each criterion j. Weights are kept non-negative
    to preserve monotonicity; they are initialized so they sum to one per
    criterion and clamped back to ``min_weight`` if they drift negative.

    Args:
        num_criteria: Number of criteria.
        num_hidden_components: Number L of components per criterion.
        min_weight: Lower bound clamp applied to negative weights.
    """

    def __init__(
        self,
        num_criteria: int,
        num_hidden_components: int,
        min_weight: float = 0.001,
        **kwargs,
    ):
        super().__init__()
        self.min_weight = min_weight
        self.weight = nn.Parameter(
            torch.FloatTensor(num_hidden_components, num_criteria)
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.uniform_(self.weight, 0.2, 1.0)
        self.weight.data = self.weight.data / torch.sum(self.weight.data)

    def compute_weight(self) -> torch.Tensor:
        with torch.no_grad():
            self.weight.data[self.weight.data < 0] = self.min_weight
        return self.weight

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return (input * self.compute_weight()).sum(1)
