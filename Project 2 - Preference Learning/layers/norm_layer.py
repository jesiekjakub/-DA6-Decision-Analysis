import torch
import torch.nn as nn


class NormLayer(nn.Module):
    """Min-max normalizer for the ANN-UTADIS comprehensive utility.

    Rescales the inner method's output so that an all-zeros input maps to 0
    and an all-ones input maps to 1. This enforces U(0) = 0 and U(1) = 1,
    which is the standard UTADIS anchoring convention and makes the learned
    thresholds directly interpretable on the [0, 1] scale.

    Args:
        method_instance: The inner module whose output should be normalized.
        num_criteria: Number of criteria (dimension of the input).
    """

    def __init__(self, method_instance: torch.nn.Module, num_criteria: int):
        super().__init__()
        self.method_instance = method_instance
        self.num_criteria = num_criteria

    def set_slope(self, slope: float):
        self.method_instance.set_slope(slope)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        out = self.method_instance(input)

        zero_input = torch.zeros(self.num_criteria).view(1, self.num_criteria).to(out.device)
        one_input = torch.ones(self.num_criteria).view(1, self.num_criteria).to(out.device)
        zero = self.method_instance(zero_input)
        one = self.method_instance(one_input)

        return (out - zero) / (one - zero + 1e-12)
