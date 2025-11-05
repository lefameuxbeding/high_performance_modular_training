import torch
import torch.nn as nn

from hpmt.models.activations import get_activation


class FFN(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        activation: str,
        bias: bool = True,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        activation_class = get_activation(activation)

        self.lin1 = nn.Linear(hidden_size, intermediate_size, bias, device, dtype)
        self.activation = activation_class()
        self.lin2 = nn.Linear(intermediate_size, hidden_size, bias, device, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.lin1(x)
        x = self.activation(x)
        x = self.lin2(x)

        return x
