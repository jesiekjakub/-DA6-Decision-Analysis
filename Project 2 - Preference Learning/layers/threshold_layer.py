import torch
import torch.nn as nn


class ThresholdLayer(nn.Module):
    """Single learnable threshold used in the binary ANN-UTADIS variant.

    For multiclass sorting use ``OrdinalThresholdLayer`` instead.

    Args:
        threshold: Fixed initial value. If ``None``, initialized uniformly
            in (0.1, 0.9).
        requires_grad: Whether the threshold is trainable.
    """

    def __init__(self, threshold: float = None, requires_grad: bool = True):
        super().__init__()
        if threshold is None:
            self.threshold = nn.Parameter(
                torch.FloatTensor(1).uniform_(0.1, 0.9), requires_grad=requires_grad
            )
        else:
            self.threshold = nn.Parameter(
                torch.FloatTensor([threshold]), requires_grad=requires_grad
            )

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        return x - self.threshold


class OrdinalThresholdLayer(nn.Module):
    """Multiclass ordinal thresholds for K-class sorting.

    Holds K-1 thresholds t_1 < t_2 < ... < t_{K-1} and assigns an
    alternative with utility U(a) to class k iff t_{k-1} <= U(a) < t_k
    (with t_0 = -inf, t_K = +inf).

    Monotonic ordering is enforced via cumulative softplus parameterization,
    so the thresholds stay strictly increasing during training.

    The forward pass returns a proper probability distribution derived from
    the cumulative-sigmoid construction (CORAL-style ordinal regression):

        P(y >= k | U) = sigmoid((U - t_{k-1}) / temperature)
        P(y == k | U) = P(y >= k) - P(y >= k+1)

    At low temperature the distribution concentrates on the bucket selected
    by ``predict``, so argmax of the probabilities matches the bucketize
    decision. This makes the probabilities usable for AUC computation and
    for SHAP attributions while keeping the class prediction exactly the
    UTADIS one.

    Args:
        num_classes: Number of classes K. Creates K-1 thresholds.
        temperature: Sharpness of the cumulative sigmoids used for the
            probability output. Lower = sharper.
    """

    def __init__(
        self,
        num_classes: int,
        temperature: float = 0.05,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_thresholds = num_classes - 1
        self.temperature = temperature
        # initialize so thresholds start roughly at 1/K, 2/K, ..., (K-1)/K
        step = 1.0 / num_classes
        raw_init = torch.empty(self.num_thresholds)
        raw_init[0] = step
        if self.num_thresholds > 1:
            # softplus^{-1}(step) so softplus(raw[1..]) == step
            inv_softplus = float(torch.log(torch.expm1(torch.tensor(step))))
            raw_init[1:] = inv_softplus
        self.raw = nn.Parameter(raw_init)

    def thresholds(self) -> torch.Tensor:
        """Return the K-1 ordered thresholds."""
        if self.num_thresholds == 1:
            return self.raw
        gaps = torch.nn.functional.softplus(self.raw[1:])
        return torch.cat([self.raw[:1], self.raw[:1] + torch.cumsum(gaps, dim=0)])

    def forward(self, utility: torch.Tensor) -> torch.Tensor:
        """Return class probabilities for each alternative.

        Args:
            utility: Shape (batch,) — the comprehensive utility U(a).

        Returns:
            Tensor of shape (batch, num_classes) with class probabilities.
        """
        t = self.thresholds()
        u = utility.unsqueeze(1)  # (batch, 1)
        tau = self.temperature
        cumulative = torch.sigmoid((u - t.unsqueeze(0)) / tau)  # P(y > k) for k = 0..K-2
        one = torch.ones_like(cumulative[:, :1])
        zero = torch.zeros_like(cumulative[:, :1])
        # P(y >= k) for k = 0..K-1
        p_ge = torch.cat([one, cumulative], dim=1)
        # P(y > k) for k = 0..K-1 (== P(y >= k+1))
        p_gt = torch.cat([cumulative, zero], dim=1)
        return p_ge - p_gt
