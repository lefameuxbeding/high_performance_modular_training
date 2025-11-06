import importlib
from collections.abc import Callable
from pathlib import Path

import torch.nn as nn

_ACTIVATION_REGISTRY: dict[str, type[nn.Module]] = {}


def register_activation(name: str) -> Callable[[type[nn.Module]], type[nn.Module]]:
    def decorator(cls: type[nn.Module]) -> type[nn.Module]:
        if name in _ACTIVATION_REGISTRY:
            raise ValueError(
                f"Activation '{name}' is already registered by "
                f"{_ACTIVATION_REGISTRY[name].__name__}"
            )
        _ACTIVATION_REGISTRY[name] = cls

        return cls

    return decorator


def get_activation(name: str) -> type[nn.Module]:
    if name not in _ACTIVATION_REGISTRY:
        available = ", ".join(sorted(_ACTIVATION_REGISTRY.keys()))
        raise ValueError(
            f"Unknown activation: '{name}'. Available activations: {available}"
        )

    return _ACTIVATION_REGISTRY[name]


def _discover_activations() -> None:
    activations_dir: Path = Path(__file__).parent
    for module_path in activations_dir.glob("*.py"):
        module_name = module_path.stem
        if module_name != "__init__" and not module_name.startswith("_"):
            importlib.import_module(f"hpmt.models.activations.{module_name}")


_discover_activations()

__all__ = ["register_activation", "get_activation"]
