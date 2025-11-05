import torch.nn as nn

from hpmt.models.activations.gelu import GELU

_ACTIVATION_REGISTRY: dict[str, type[nn.Module]] = {
    "gelu": GELU,
}


def get_activation(name: str) -> type[nn.Module]:
    if name not in _ACTIVATION_REGISTRY:
        available = ", ".join(sorted(_ACTIVATION_REGISTRY.keys()))
        raise ValueError(
            f"Unknown activation: '{name}'. Available activations: {available}"
        )
    return _ACTIVATION_REGISTRY[name]


__all__ = ["get_activation"]
