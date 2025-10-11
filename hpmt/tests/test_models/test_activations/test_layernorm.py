import pytest
import torch

from hpmt.models.activations.layernorm import LayerNorm

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def simple_input_2d():
    """2D input tensor (batch, features)."""
    return torch.randn(8, 128)


@pytest.fixture
def simple_input_3d():
    """3D input tensor (batch, sequence, features) - common in transformers."""
    return torch.randn(4, 32, 512)


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    """Parametrized fixture for CPU and CUDA devices."""
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device(request.param)


# ============================================================================
# Shape Tests
# ============================================================================


def test_output_shape_1d_normalized():
    """Test output shape with 1D normalized shape."""
    layer = LayerNorm(128)
    x = torch.randn(8, 128)
    output = layer(x)
    assert output.shape == x.shape == (8, 128)


def test_output_shape_2d_normalized():
    """Test output shape with 2D normalized shape."""
    layer = LayerNorm([64, 128])
    x = torch.randn(4, 64, 128)
    output = layer(x)
    assert output.shape == x.shape == (4, 64, 128)


def test_output_shape_3d_input():
    """Test output shape with 3D input (transformer case)."""
    layer = LayerNorm(512)
    x = torch.randn(4, 32, 512)
    output = layer(x)
    assert output.shape == x.shape == (4, 32, 512)


def test_output_shape_cpu():
    """Test output shape on CPU device."""
    layer = LayerNorm(128, device=torch.device("cpu"))
    x = torch.randn(8, 128, device=torch.device("cpu"))
    output = layer(x)
    assert output.shape == x.shape
    assert output.device.type == "cpu"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_output_shape_cuda():
    """Test output shape on CUDA device."""
    layer = LayerNorm(128, device=torch.device("cuda"))
    x = torch.randn(8, 128, device=torch.device("cuda"))
    output = layer(x)
    assert output.shape == x.shape
    assert output.device.type == "cuda"


# ============================================================================
# Gradient Tests
# ============================================================================


def test_gradients_flow_through_layer():
    """Test that gradients flow through the layer."""
    layer = LayerNorm(128)
    x = torch.randn(8, 128, requires_grad=True)
    output = layer(x)
    loss = output.sum()
    loss.backward()
    assert x.grad is not None
    assert x.grad.shape == x.shape


def test_gradients_with_affine_and_bias():
    """Test gradients exist for weight and bias when elementwise_affine=True, bias=True."""
    layer = LayerNorm(128, elementwise_affine=True, bias=True)
    x = torch.randn(8, 128, requires_grad=True)
    output = layer(x)
    loss = output.sum()
    loss.backward()

    assert layer.weight is not None
    assert layer.weight.grad is not None
    assert layer.weight.grad.shape == (128,)

    assert layer.bias is not None
    assert layer.bias.grad is not None
    assert layer.bias.grad.shape == (128,)


def test_gradients_with_affine_no_bias():
    """Test gradients exist for weight only when elementwise_affine=True, bias=False."""
    layer = LayerNorm(128, elementwise_affine=True, bias=False)
    x = torch.randn(8, 128, requires_grad=True)
    output = layer(x)
    loss = output.sum()
    loss.backward()

    assert layer.weight is not None
    assert layer.weight.grad is not None
    assert layer.weight.grad.shape == (128,)

    assert layer.bias is None


def test_gradients_without_affine():
    """Test no parameter gradients when elementwise_affine=False."""
    layer = LayerNorm(128, elementwise_affine=False)
    x = torch.randn(8, 128, requires_grad=True)
    output = layer(x)
    loss = output.sum()
    loss.backward()

    assert layer.weight is None
    assert layer.bias is None


def test_gradients_correct_shape():
    """Test gradient shapes match parameter shapes."""
    layer = LayerNorm(256, elementwise_affine=True, bias=True)
    x = torch.randn(4, 256, requires_grad=True)
    output = layer(x)
    loss = output.sum()
    loss.backward()

    assert layer.weight is not None
    assert layer.bias is not None
    assert layer.weight.grad is not None
    assert layer.bias.grad is not None
    assert layer.weight.grad.shape == layer.weight.shape == (256,)
    assert layer.bias.grad.shape == layer.bias.shape == (256,)


# ============================================================================
# Numerical Correctness Tests
# ============================================================================


def test_numerical_correctness_default():
    """Test numerical correctness against torch.nn.LayerNorm with default settings."""
    normalized_shape = 128
    x = torch.randn(8, 128)

    # Our implementation
    layer_custom = LayerNorm(normalized_shape)
    # PyTorch implementation
    layer_torch = torch.nn.LayerNorm(normalized_shape)

    # Copy parameters to ensure same initialization
    with torch.no_grad():
        layer_torch.weight.copy_(layer_custom.weight)
        layer_torch.bias.copy_(layer_custom.bias)

    output_custom = layer_custom(x)
    output_torch = layer_torch(x)

    assert torch.allclose(output_custom, output_torch)


def test_numerical_correctness_no_affine():
    """Test numerical correctness with elementwise_affine=False."""
    normalized_shape = 128
    x = torch.randn(8, 128)

    layer_custom = LayerNorm(normalized_shape, elementwise_affine=False)
    layer_torch = torch.nn.LayerNorm(normalized_shape, elementwise_affine=False)

    output_custom = layer_custom(x)
    output_torch = layer_torch(x)

    assert torch.allclose(output_custom, output_torch)


def test_numerical_correctness_no_bias():
    """Test numerical correctness with bias=False."""
    normalized_shape = 128
    x = torch.randn(8, 128)

    layer_custom = LayerNorm(normalized_shape, bias=False)
    layer_torch = torch.nn.LayerNorm(normalized_shape, bias=False)

    # Copy weight parameter
    with torch.no_grad():
        layer_torch.weight.copy_(layer_custom.weight)

    output_custom = layer_custom(x)
    output_torch = layer_torch(x)

    assert torch.allclose(output_custom, output_torch)


@pytest.mark.parametrize("eps", [1e-3, 1e-5, 1e-6])
def test_numerical_correctness_different_eps(eps):
    """Test numerical correctness with different epsilon values."""
    normalized_shape = 128
    x = torch.randn(8, 128)

    layer_custom = LayerNorm(normalized_shape, eps=eps)
    layer_torch = torch.nn.LayerNorm(normalized_shape, eps=eps)

    # Copy parameters
    with torch.no_grad():
        layer_torch.weight.copy_(layer_custom.weight)
        layer_torch.bias.copy_(layer_custom.bias)

    output_custom = layer_custom(x)
    output_torch = layer_torch(x)

    assert torch.allclose(output_custom, output_torch)


def test_numerical_edge_case_small_values():
    """Test with very small input values near zero."""
    normalized_shape = 128
    x = torch.randn(8, 128) * 1e-7

    layer_custom = LayerNorm(normalized_shape)
    layer_torch = torch.nn.LayerNorm(normalized_shape)

    # Copy parameters
    with torch.no_grad():
        layer_torch.weight.copy_(layer_custom.weight)
        layer_torch.bias.copy_(layer_custom.bias)

    output_custom = layer_custom(x)
    output_torch = layer_torch(x)

    assert torch.allclose(output_custom, output_torch)


def test_numerical_edge_case_large_values():
    """Test with large input values."""
    normalized_shape = 128
    x = torch.randn(8, 128) * 1e3

    layer_custom = LayerNorm(normalized_shape)
    layer_torch = torch.nn.LayerNorm(normalized_shape)

    # Copy parameters
    with torch.no_grad():
        layer_torch.weight.copy_(layer_custom.weight)
        layer_torch.bias.copy_(layer_custom.bias)

    output_custom = layer_custom(x)
    output_torch = layer_torch(x)

    assert torch.allclose(output_custom, output_torch)


def test_numerical_edge_case_mixed_signs():
    """Test with mixed positive and negative values."""
    normalized_shape = 128
    x = torch.randn(8, 128)  # Already mixed signs

    layer_custom = LayerNorm(normalized_shape)
    layer_torch = torch.nn.LayerNorm(normalized_shape)

    # Copy parameters
    with torch.no_grad():
        layer_torch.weight.copy_(layer_custom.weight)
        layer_torch.bias.copy_(layer_custom.bias)

    output_custom = layer_custom(x)
    output_torch = layer_torch(x)

    assert torch.allclose(output_custom, output_torch)


def test_numerical_edge_case_near_zero_variance():
    """Test with input where all values are very close (low variance)."""
    normalized_shape = 128
    # Create input with very low variance
    x = torch.ones(8, 128) + torch.randn(8, 128) * 1e-6

    layer_custom = LayerNorm(normalized_shape)
    layer_torch = torch.nn.LayerNorm(normalized_shape)

    # Copy parameters
    with torch.no_grad():
        layer_torch.weight.copy_(layer_custom.weight)
        layer_torch.bias.copy_(layer_custom.bias)

    output_custom = layer_custom(x)
    output_torch = layer_torch(x)

    assert torch.allclose(output_custom, output_torch)


# ============================================================================
# Additional Tests
# ============================================================================


def test_parameter_initialization_with_affine():
    """Test that weight parameter is initialized when elementwise_affine=True."""
    layer = LayerNorm(128, elementwise_affine=True)
    assert layer.weight is not None
    assert layer.weight.shape == (128,)
    assert isinstance(layer.weight, torch.nn.Parameter)


def test_parameter_initialization_with_bias():
    """Test that bias parameter is initialized when bias=True."""
    layer = LayerNorm(128, elementwise_affine=True, bias=True)
    assert layer.bias is not None
    assert layer.bias.shape == (128,)
    assert isinstance(layer.bias, torch.nn.Parameter)


def test_parameter_initialization_no_affine():
    """Test that no parameters are created when elementwise_affine=False."""
    layer = LayerNorm(128, elementwise_affine=False)
    assert layer.weight is None
    assert layer.bias is None


def test_device_cpu():
    """Test layer creation on CPU device."""
    device = torch.device("cpu")
    layer = LayerNorm(128, device=device)

    if layer.weight is not None:
        assert layer.weight.device == device
    if layer.bias is not None:
        assert layer.bias.device == device


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_device_cuda():
    """Test layer creation on CUDA device."""
    device = torch.device("cuda")
    layer = LayerNorm(128, device=device)

    if layer.weight is not None:
        assert layer.weight.device.type == "cuda"
    if layer.bias is not None:
        assert layer.bias.device.type == "cuda"


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_dtype_support(dtype):
    """Test layer creation with different dtypes."""
    layer = LayerNorm(128, dtype=dtype)

    if layer.weight is not None:
        assert layer.weight.dtype == dtype
    if layer.bias is not None:
        assert layer.bias.dtype == dtype


# ============================================================================
# Benchmark Tests
# ============================================================================


@pytest.mark.benchmark(group="forward_small")
def test_benchmark_forward_small_custom(benchmark):
    """Benchmark custom LayerNorm forward pass with small hidden size (128)."""
    hidden_size = 128
    batch_size = 8

    layer = LayerNorm(hidden_size)
    x = torch.randn(batch_size, hidden_size)

    benchmark(layer, x)


@pytest.mark.benchmark(group="forward_small")
def test_benchmark_forward_small_pytorch(benchmark):
    """Benchmark PyTorch LayerNorm forward pass with small hidden size (128)."""
    hidden_size = 128
    batch_size = 8

    layer = torch.nn.LayerNorm(hidden_size)
    x = torch.randn(batch_size, hidden_size)

    benchmark(layer, x)


@pytest.mark.benchmark(group="forward_medium")
def test_benchmark_forward_medium_custom(benchmark):
    """Benchmark custom LayerNorm forward pass with medium hidden size (512)."""
    hidden_size = 512
    batch_size = 8

    layer = LayerNorm(hidden_size)
    x = torch.randn(batch_size, hidden_size)

    benchmark(layer, x)


@pytest.mark.benchmark(group="forward_medium")
def test_benchmark_forward_medium_pytorch(benchmark):
    """Benchmark PyTorch LayerNorm forward pass with medium hidden size (512)."""
    hidden_size = 512
    batch_size = 8

    layer = torch.nn.LayerNorm(hidden_size)
    x = torch.randn(batch_size, hidden_size)

    benchmark(layer, x)


@pytest.mark.benchmark(group="forward_large")
def test_benchmark_forward_large_custom(benchmark):
    """Benchmark custom LayerNorm forward pass with large hidden size (4096)."""
    hidden_size = 4096
    batch_size = 8

    layer = LayerNorm(hidden_size)
    x = torch.randn(batch_size, hidden_size)

    benchmark(layer, x)


@pytest.mark.benchmark(group="forward_large")
def test_benchmark_forward_large_pytorch(benchmark):
    """Benchmark PyTorch LayerNorm forward pass with large hidden size (4096)."""
    hidden_size = 4096
    batch_size = 8

    layer = torch.nn.LayerNorm(hidden_size)
    x = torch.randn(batch_size, hidden_size)

    benchmark(layer, x)


@pytest.mark.benchmark(group="forward_backward")
def test_benchmark_forward_backward_custom(benchmark):
    """Benchmark custom LayerNorm full training step (forward + backward)."""
    hidden_size = 512
    batch_size = 8

    def training_step():
        layer = LayerNorm(hidden_size)
        x = torch.randn(batch_size, hidden_size, requires_grad=True)
        output = layer(x)
        loss = output.sum()
        loss.backward()
        return loss

    benchmark(training_step)


@pytest.mark.benchmark(group="forward_backward")
def test_benchmark_forward_backward_pytorch(benchmark):
    """Benchmark PyTorch LayerNorm full training step (forward + backward)."""
    hidden_size = 512
    batch_size = 8

    def training_step():
        layer = torch.nn.LayerNorm(hidden_size)
        x = torch.randn(batch_size, hidden_size, requires_grad=True)
        output = layer(x)
        loss = output.sum()
        loss.backward()
        return loss

    benchmark(training_step)
