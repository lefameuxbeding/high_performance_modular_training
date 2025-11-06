import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    def __init__(
        self,
        normalized_shape: int | list[int] | torch.Size,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        bias: bool = True,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        tuple_normalized_shape: tuple[int, ...] = (
            (normalized_shape,)
            if isinstance(normalized_shape, int)
            else tuple(normalized_shape)
        )
        self.dim = list(range(-len(tuple_normalized_shape), 0))
        self.eps = eps

        if elementwise_affine:
            self.weight = nn.Parameter(
                torch.ones(tuple_normalized_shape, device=device, dtype=dtype)
            )
            if bias:
                self.bias = nn.Parameter(
                    torch.zeros(tuple_normalized_shape, device=device, dtype=dtype)
                )
            else:
                self.register_parameter("bias", None)
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=self.dim, keepdim=True)
        var = x.var(dim=self.dim, unbiased=False, keepdim=True)

        x_norm = (x - mean) / torch.sqrt(var + self.eps)

        if self.weight is not None:
            x_norm = x_norm * self.weight
        if self.bias is not None:
            x_norm = x_norm + self.bias

        return x_norm
