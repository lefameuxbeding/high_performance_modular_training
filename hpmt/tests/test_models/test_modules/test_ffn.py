import pytest
import torch

from hpmt.models.modules.ffn import FFN

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def standard_dims():
    """Standard dimensions used in most tests."""
    return {"hidden_size": 512, "intermediate_size": 2048}


@pytest.fixture
def small_dims():
    """Small dimensions for quick tests (device/dtype tests)."""
    return {"hidden_size": 128, "intermediate_size": 512}


# ============================================================================
# Shape Propagation Tests
# ============================================================================


def test_output_shape_2d_basic(standard_dims):
    """Test output shape with 2D input (batch, hidden_size)."""
    ffn = FFN(**standard_dims, activation="gelu")
    x = torch.randn(8, standard_dims["hidden_size"])
    output = ffn(x)
    assert output.shape == (8, standard_dims["hidden_size"])


def test_output_shape_3d_transformer(standard_dims):
    """Test output shape with 3D input (batch, seq_len, hidden_size)."""
    ffn = FFN(**standard_dims, activation="gelu")
    x = torch.randn(4, 32, standard_dims["hidden_size"])
    output = ffn(x)
    assert output.shape == (4, 32, standard_dims["hidden_size"])


def test_output_shape_4d(standard_dims):
    """Test output shape with 4D input (batch, channels, height, hidden_size)."""
    ffn = FFN(**standard_dims, activation="gelu")
    x = torch.randn(2, 3, 16, standard_dims["hidden_size"])
    output = ffn(x)
    assert output.shape == (2, 3, 16, standard_dims["hidden_size"])


def test_output_shape_different_batch_sizes(standard_dims):
    """Test output shape with different batch sizes."""
    ffn = FFN(**standard_dims, activation="gelu")

    for batch_size in [1, 4, 16, 64]:
        x = torch.randn(batch_size, standard_dims["hidden_size"])
        output = ffn(x)
        assert output.shape == (batch_size, standard_dims["hidden_size"])


# ============================================================================
# Gradient Flow Tests
# ============================================================================


def test_gradients_flow_through_pipeline(standard_dims):
    """Test that gradients flow through lin1 -> activation -> lin2."""
    ffn = FFN(**standard_dims, activation="gelu", bias=True)
    x = torch.randn(8, standard_dims["hidden_size"], requires_grad=True)

    output = ffn(x)
    loss = output.sum()
    loss.backward()

    # Check input gradient exists
    assert x.grad is not None
    assert x.grad.shape == x.shape

    # Check all parameter gradients exist
    assert ffn.lin1.weight.grad is not None
    assert ffn.lin1.bias.grad is not None
    assert ffn.lin2.weight.grad is not None
    assert ffn.lin2.bias.grad is not None


def test_gradients_correct_shape(standard_dims):
    """Test that all parameter gradients have correct shapes."""
    hidden_size = standard_dims["hidden_size"]
    intermediate_size = standard_dims["intermediate_size"]
    ffn = FFN(**standard_dims, activation="gelu", bias=True)
    x = torch.randn(8, hidden_size, requires_grad=True)

    output = ffn(x)
    loss = output.sum()
    loss.backward()

    # Check gradient shapes match parameter shapes
    assert ffn.lin1.weight.grad is not None
    assert ffn.lin1.weight.grad.shape == (intermediate_size, hidden_size)
    assert ffn.lin1.bias.grad is not None
    assert ffn.lin1.bias.grad.shape == (intermediate_size,)
    assert ffn.lin2.weight.grad is not None
    assert ffn.lin2.weight.grad.shape == (hidden_size, intermediate_size)
    assert ffn.lin2.bias.grad is not None
    assert ffn.lin2.bias.grad.shape == (hidden_size,)


def test_gradients_without_bias(standard_dims):
    """Test gradient flow when bias=False."""
    ffn = FFN(**standard_dims, activation="gelu", bias=False)
    x = torch.randn(8, standard_dims["hidden_size"], requires_grad=True)

    output = ffn(x)
    loss = output.sum()
    loss.backward()

    # Check input gradient exists
    assert x.grad is not None

    # Check weight gradients exist
    assert ffn.lin1.weight.grad is not None
    assert ffn.lin2.weight.grad is not None

    # Check bias gradients don't exist (bias=False)
    assert ffn.lin1.bias is None
    assert ffn.lin2.bias is None


# ============================================================================
# Activation Application Tests
# ============================================================================


def test_activation_is_applied(small_dims):
    """Verify activation function is actually applied in forward pass."""
    ffn = FFN(**small_dims, activation="gelu", bias=False)
    x = torch.randn(8, small_dims["hidden_size"])

    # Capture intermediate values
    with torch.no_grad():
        intermediate = ffn.lin1(x)
        activated = ffn.activation(intermediate)

    # They should be different (activation was applied)
    assert not torch.allclose(intermediate, activated)

    # Full forward should match manual computation
    with torch.no_grad():
        output_manual = ffn.lin2(activated)
        output_ffn = ffn(x)

    assert torch.allclose(output_manual, output_ffn, rtol=1e-5, atol=1e-7)


# ============================================================================
# Parameter Count Tests
# ============================================================================


def test_parameter_count_with_bias(standard_dims):
    """Test total parameter count with bias=True."""
    hidden_size = standard_dims["hidden_size"]
    intermediate_size = standard_dims["intermediate_size"]
    ffn = FFN(**standard_dims, activation="gelu", bias=True)

    # Expected: lin1 (weight + bias) + lin2 (weight + bias)
    expected = (hidden_size * intermediate_size + intermediate_size) + (
        intermediate_size * hidden_size + hidden_size
    )

    actual = sum(p.numel() for p in ffn.parameters())

    assert actual == expected


def test_parameter_count_without_bias(standard_dims):
    """Test total parameter count with bias=False."""
    hidden_size = standard_dims["hidden_size"]
    intermediate_size = standard_dims["intermediate_size"]
    ffn = FFN(**standard_dims, activation="gelu", bias=False)

    # Expected: lin1 (weight only) + lin2 (weight only)
    expected = (hidden_size * intermediate_size) + (intermediate_size * hidden_size)

    actual = sum(p.numel() for p in ffn.parameters())

    assert actual == expected


def test_parameter_shapes_with_bias(standard_dims):
    """Test individual parameter shapes with bias=True."""
    hidden_size = standard_dims["hidden_size"]
    intermediate_size = standard_dims["intermediate_size"]
    ffn = FFN(**standard_dims, activation="gelu", bias=True)

    # Check lin1 parameters
    assert ffn.lin1.weight.shape == (intermediate_size, hidden_size)
    assert ffn.lin1.bias.shape == (intermediate_size,)

    # Check lin2 parameters
    assert ffn.lin2.weight.shape == (hidden_size, intermediate_size)
    assert ffn.lin2.bias.shape == (hidden_size,)


def test_parameter_shapes_without_bias(standard_dims):
    """Test individual parameter shapes with bias=False."""
    hidden_size = standard_dims["hidden_size"]
    intermediate_size = standard_dims["intermediate_size"]
    ffn = FFN(**standard_dims, activation="gelu", bias=False)

    # Check lin1 parameters
    assert ffn.lin1.weight.shape == (intermediate_size, hidden_size)
    assert ffn.lin1.bias is None

    # Check lin2 parameters
    assert ffn.lin2.weight.shape == (hidden_size, intermediate_size)
    assert ffn.lin2.bias is None
