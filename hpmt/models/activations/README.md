# Activation Functions

This module provides a registry-based system for activation functions with automatic discovery.

## Implementing a New Activation Function

To add a new activation function to the registry:

1. Create a new Python file in this directory (e.g., `gelu.py`)
2. Import the `register_activation` decorator and `nn.Module`
3. Create a class that inherits from `nn.Module`
4. Decorate your class with `@register_activation("name")`
5. Implement the `forward()` method

### Example

See the current GELU implementation as reference:

```python
# gelu.py
import torch
import torch.nn as nn

from hpmt.models.activations import register_activation


@register_activation("gelu")
class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ...
```

The activation will be automatically discovered and registered when the module is imported.

## Using Activation Functions

The preferred way to instantiate activation functions is through the `get_activation()` function:

```python
from hpmt.models.activations import get_activation

activation_cls = get_activation("gelu")
activation = activation_cls()
output = activation(input_tensor)
```
