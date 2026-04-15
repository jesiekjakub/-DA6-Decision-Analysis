import torch.nn as nn
import torch.nn.functional as F


class LeakyHardSigmoid(nn.Module):
    """
    LeakyHardSigmoid activation function.

    A monotonic approximation of the sigmoid built from two leaky-ReLU
    operations. It is used inside the monotonic block of ANN-UTADIS to
    produce piecewise-linear, saturating marginal utility components while
    avoiding the vanishing-gradient problem of the standard hard sigmoid.

    Args:
        slope (float, optional): Slope of the leaky parts. Defaults to 0.01.
    """

    def __init__(self, slope: float = 0.01, **kwargs):
        super().__init__()
        self.slope = slope

    def set_slope(self, val: float) -> None:
        self.slope = val

    def forward(self, input):
        return F.leaky_relu(1.0 - F.leaky_relu(1 - input, self.slope), self.slope)
