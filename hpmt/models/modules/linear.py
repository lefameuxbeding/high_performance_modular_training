import torch
import torch.nn as nn


class Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        k = in_features
        bound = 1 / (k**0.5)
        self.weight = nn.Parameter(
            torch.empty(
                (in_features, out_features), device=device, dtype=dtype
            ).uniform_(-bound, bound)
        )

        if bias:
            self.bias = nn.Parameter(
                torch.empty(out_features, device=device, dtype=dtype).uniform_(
                    -bound, bound
                )
            )
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor):
        output = x @ self.weight

        if self.bias is not None:
            output = output + self.bias

        return output
