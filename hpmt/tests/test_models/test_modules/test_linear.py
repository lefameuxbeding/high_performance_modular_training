import pytest
import torch

from hpmt.models.modules.linear import Linear

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def simple_input_2d():
    """2D input tensor (batch, in_features)."""
    return torch.randn(8, 128)


@pytest.fixture
def simple_input_3d():
    """3D input tensor (batch, sequence, in_features) - common in transformers."""
    return torch.randn(4, 32, 512)


@pytest.fixture
def simple_input_4d():
    """4D input tensor (batch, channels, height, in_features)."""
    return torch.randn(2, 3, 16, 64)


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    """Parametrized fixture for CPU and CUDA devices."""
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device(request.param)


# ============================================================================
# Shape Tests
# ============================================================================


def test_output_shape_2d_basic():
    """Test output shape with 2D input (batch, in_features)."""
    layer = Linear(128, 256)
    x = torch.randn(8, 128)
    output = layer(x)
    assert output.shape == (8, 256)


def test_output_shape_3d_transformer():
    """Test output shape with 3D input (batch, seq_len, in_features) - transformer use case."""
    layer = Linear(512, 1024)
    x = torch.randn(4, 32, 512)
    output = layer(x)
    assert output.shape == (4, 32, 1024)


def test_output_shape_4d():
    """Test output shape with 4D input (batch, channels, height, in_features)."""
    layer = Linear(64, 128)
    x = torch.randn(2, 3, 16, 64)
    output = layer(x)
    assert output.shape == (2, 3, 16, 128)


def test_output_shape_small_features():
    """Test output shape with small feature sizes."""
    layer = Linear(16, 32)
    x = torch.randn(8, 16)
    output = layer(x)
    assert output.shape == (8, 32)


def test_output_shape_large_features():
    """Test output shape with large feature sizes."""
    layer = Linear(4096, 8192)
    x = torch.randn(2, 4096)
    output = layer(x)
    assert output.shape == (2, 8192)


def test_output_shape_cpu():
    """Test output shape on CPU device."""
    layer = Linear(128, 256, device=torch.device("cpu"))
    x = torch.randn(8, 128, device=torch.device("cpu"))
    output = layer(x)
    assert output.shape == (8, 256)
    assert output.device.type == "cpu"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_output_shape_cuda():
    """Test output shape on CUDA device."""
    layer = Linear(128, 256, device=torch.device("cuda"))
    x = torch.randn(8, 128, device=torch.device("cuda"))
    output = layer(x)
    assert output.shape == (8, 256)
    assert output.device.type == "cuda"


# ============================================================================
# Gradient Tests
# ============================================================================


def test_gradients_flow_through_layer():
    """Test that gradients flow through the layer."""
    layer = Linear(128, 256)
    x = torch.randn(8, 128, requires_grad=True)
    output = layer(x)
    loss = output.sum()
    loss.backward()
    assert x.grad is not None
    assert x.grad.shape == x.shape


def test_gradients_with_bias():
    """Test gradients exist for weight and bias when bias=True."""
    layer = Linear(128, 256, bias=True)
    x = torch.randn(8, 128, requires_grad=True)
    output = layer(x)
    loss = output.sum()
    loss.backward()

    assert layer.weight is not None
    assert layer.weight.grad is not None
    assert layer.weight.grad.shape == (128, 256)

    assert layer.bias is not None
    assert layer.bias.grad is not None
    assert layer.bias.grad.shape == (256,)


def test_gradients_without_bias():
    """Test gradients exist for weight only when bias=False."""
    layer = Linear(128, 256, bias=False)
    x = torch.randn(8, 128, requires_grad=True)
    output = layer(x)
    loss = output.sum()
    loss.backward()

    assert layer.weight is not None
    assert layer.weight.grad is not None
    assert layer.weight.grad.shape == (128, 256)

    assert layer.bias is None


def test_gradients_correct_shape():
    """Test gradient shapes match parameter shapes."""
    layer = Linear(256, 512, bias=True)
    x = torch.randn(4, 256, requires_grad=True)
    output = layer(x)
    loss = output.sum()
    loss.backward()

    assert layer.weight is not None
    assert layer.bias is not None
    assert layer.weight.grad is not None
    assert layer.bias.grad is not None
    assert layer.weight.grad.shape == layer.weight.shape == (256, 512)
    assert layer.bias.grad.shape == layer.bias.shape == (512,)


def test_gradients_accumulate():
    """Test that gradients accumulate correctly over multiple backward passes."""
    layer = Linear(64, 128, bias=True)
    x1 = torch.randn(4, 64, requires_grad=True)
    x2 = torch.randn(4, 64, requires_grad=True)

    # First backward pass
    output1 = layer(x1)
    loss1 = output1.sum()
    loss1.backward()

    # Store first gradients
    assert layer.weight.grad is not None
    assert layer.bias is not None
    assert layer.bias.grad is not None
    first_weight_grad = layer.weight.grad.clone()
    first_bias_grad = layer.bias.grad.clone()

    # Second backward pass (accumulates)
    output2 = layer(x2)
    loss2 = output2.sum()
    loss2.backward()

    # Gradients should have accumulated
    assert layer.weight.grad is not None
    assert layer.bias.grad is not None
    assert not torch.allclose(layer.weight.grad, first_weight_grad)
    assert not torch.allclose(layer.bias.grad, first_bias_grad)


def test_gradients_3d_input():
    """Test gradients flow correctly with 3D input (batch, seq_len, in_features)."""
    layer = Linear(512, 256, bias=True)
    x = torch.randn(4, 32, 512, requires_grad=True)
    output = layer(x)
    loss = output.sum()
    loss.backward()

    assert x.grad is not None
    assert x.grad.shape == x.shape
    assert layer.weight.grad is not None
    assert layer.weight.grad.shape == (512, 256)
    assert layer.bias is not None
    assert layer.bias.grad is not None
    assert layer.bias.grad.shape == (256,)


# ============================================================================
# Numerical Correctness Tests
# ============================================================================


@pytest.mark.parametrize(
    "in_features,out_features,batch_size",
    [
        (128, 256, 8),  # Small
        (512, 1024, 4),  # Medium
        (2048, 4096, 2),  # Large
    ],
)
def test_numerical_correctness_with_bias(in_features, out_features, batch_size):
    """Test numerical correctness against torch.nn.Linear with bias=True."""
    x = torch.randn(batch_size, in_features)

    # Our implementation
    layer_custom = Linear(in_features, out_features, bias=True)
    # PyTorch implementation
    layer_torch = torch.nn.Linear(in_features, out_features, bias=True)

    # Copy parameters to ensure same initialization
    with torch.no_grad():
        layer_torch.weight.copy_(
            layer_custom.weight.T
        )  # PyTorch uses transposed weight
        layer_torch.bias.copy_(layer_custom.bias)

    output_custom = layer_custom(x)
    output_torch = layer_torch(x)

    assert torch.allclose(output_custom, output_torch, rtol=1e-2, atol=1e-4)


@pytest.mark.parametrize(
    "in_features,out_features,batch_size",
    [
        (128, 256, 8),  # Small
        (512, 1024, 4),  # Medium
        (2048, 4096, 2),  # Large
    ],
)
def test_numerical_correctness_without_bias(in_features, out_features, batch_size):
    """Test numerical correctness against torch.nn.Linear with bias=False."""
    x = torch.randn(batch_size, in_features)

    layer_custom = Linear(in_features, out_features, bias=False)
    layer_torch = torch.nn.Linear(in_features, out_features, bias=False)

    # Copy parameters
    with torch.no_grad():
        layer_torch.weight.copy_(
            layer_custom.weight.T
        )  # PyTorch uses transposed weight

    output_custom = layer_custom(x)
    output_torch = layer_torch(x)

    assert torch.allclose(output_custom, output_torch, rtol=1e-2, atol=1e-4)


def test_numerical_correctness_3d_input():
    """Test numerical correctness with 3D input (batch, seq_len, in_features)."""
    in_features, out_features = 512, 256
    x = torch.randn(4, 32, in_features)

    layer_custom = Linear(in_features, out_features, bias=True)
    layer_torch = torch.nn.Linear(in_features, out_features, bias=True)

    # Copy parameters
    with torch.no_grad():
        layer_torch.weight.copy_(layer_custom.weight.T)
        layer_torch.bias.copy_(layer_custom.bias)

    output_custom = layer_custom(x)
    output_torch = layer_torch(x)

    assert torch.allclose(output_custom, output_torch, rtol=1e-2, atol=1e-4)


def test_numerical_correctness_4d_input():
    """Test numerical correctness with 4D input (batch, channels, height, in_features)."""
    in_features, out_features = 64, 128
    x = torch.randn(2, 3, 16, in_features)

    layer_custom = Linear(in_features, out_features, bias=True)
    layer_torch = torch.nn.Linear(in_features, out_features, bias=True)

    # Copy parameters
    with torch.no_grad():
        layer_torch.weight.copy_(layer_custom.weight.T)
        layer_torch.bias.copy_(layer_custom.bias)

    output_custom = layer_custom(x)
    output_torch = layer_torch(x)

    assert torch.allclose(output_custom, output_torch, rtol=1e-2, atol=1e-4)


def test_numerical_edge_case_small_values():
    """Test with very small input values near zero."""
    in_features, out_features = 128, 256
    x = torch.randn(8, in_features) * 1e-7

    layer_custom = Linear(in_features, out_features)
    layer_torch = torch.nn.Linear(in_features, out_features)

    # Copy parameters
    with torch.no_grad():
        layer_torch.weight.copy_(layer_custom.weight.T)
        layer_torch.bias.copy_(layer_custom.bias)

    output_custom = layer_custom(x)
    output_torch = layer_torch(x)

    assert torch.allclose(output_custom, output_torch, rtol=1e-2, atol=1e-4)


def test_numerical_edge_case_large_values():
    """Test with large input values."""
    in_features, out_features = 128, 256
    x = torch.randn(8, in_features) * 1e3

    layer_custom = Linear(in_features, out_features)
    layer_torch = torch.nn.Linear(in_features, out_features)

    # Copy parameters
    with torch.no_grad():
        layer_torch.weight.copy_(layer_custom.weight.T)
        layer_torch.bias.copy_(layer_custom.bias)

    output_custom = layer_custom(x)
    output_torch = layer_torch(x)

    assert torch.allclose(output_custom, output_torch, rtol=1e-2, atol=1e-4)


def test_numerical_edge_case_zero_input():
    """Test with zero input."""
    in_features, out_features = 128, 256
    x = torch.zeros(8, in_features)

    layer_custom = Linear(in_features, out_features)
    layer_torch = torch.nn.Linear(in_features, out_features)

    # Copy parameters
    with torch.no_grad():
        layer_torch.weight.copy_(layer_custom.weight.T)
        layer_torch.bias.copy_(layer_custom.bias)

    output_custom = layer_custom(x)
    output_torch = layer_torch(x)

    assert torch.allclose(output_custom, output_torch, rtol=1e-2, atol=1e-4)


def test_numerical_edge_case_negative_input():
    """Test with all negative input values."""
    in_features, out_features = 128, 256
    x = -torch.abs(torch.randn(8, in_features))

    layer_custom = Linear(in_features, out_features)
    layer_torch = torch.nn.Linear(in_features, out_features)

    # Copy parameters
    with torch.no_grad():
        layer_torch.weight.copy_(layer_custom.weight.T)
        layer_torch.bias.copy_(layer_custom.bias)

    output_custom = layer_custom(x)
    output_torch = layer_torch(x)

    assert torch.allclose(output_custom, output_torch, rtol=1e-2, atol=1e-4)


def test_numerical_identity_mapping():
    """Test with identity mapping (in_features == out_features)."""
    features = 256
    x = torch.randn(8, features)

    layer_custom = Linear(features, features)
    layer_torch = torch.nn.Linear(features, features)

    # Copy parameters
    with torch.no_grad():
        layer_torch.weight.copy_(layer_custom.weight.T)
        layer_torch.bias.copy_(layer_custom.bias)

    output_custom = layer_custom(x)
    output_torch = layer_torch(x)

    assert torch.allclose(output_custom, output_torch, rtol=1e-2, atol=1e-4)


def test_numerical_dimension_expansion():
    """Test with dimension expansion (16 -> 512)."""
    in_features, out_features = 16, 512
    x = torch.randn(8, in_features)

    layer_custom = Linear(in_features, out_features)
    layer_torch = torch.nn.Linear(in_features, out_features)

    # Copy parameters
    with torch.no_grad():
        layer_torch.weight.copy_(layer_custom.weight.T)
        layer_torch.bias.copy_(layer_custom.bias)

    output_custom = layer_custom(x)
    output_torch = layer_torch(x)

    assert torch.allclose(output_custom, output_torch, rtol=1e-2, atol=1e-4)


def test_numerical_dimension_reduction():
    """Test with dimension reduction (512 -> 16)."""
    in_features, out_features = 512, 16
    x = torch.randn(8, in_features)

    layer_custom = Linear(in_features, out_features)
    layer_torch = torch.nn.Linear(in_features, out_features)

    # Copy parameters
    with torch.no_grad():
        layer_torch.weight.copy_(layer_custom.weight.T)
        layer_torch.bias.copy_(layer_custom.bias)

    output_custom = layer_custom(x)
    output_torch = layer_torch(x)

    assert torch.allclose(output_custom, output_torch, rtol=1e-2, atol=1e-4)


# ============================================================================
# Parameter Initialization Tests
# ============================================================================


def test_parameter_initialization_weight_shape():
    """Test that weight parameter has correct shape (in_features, out_features)."""
    layer = Linear(128, 256)
    assert layer.weight is not None
    assert layer.weight.shape == (128, 256)
    assert isinstance(layer.weight, torch.nn.Parameter)


def test_parameter_initialization_bias_shape():
    """Test that bias parameter has correct shape (out_features,)."""
    layer = Linear(128, 256, bias=True)
    assert layer.bias is not None
    assert layer.bias.shape == (256,)
    assert isinstance(layer.bias, torch.nn.Parameter)


def test_parameter_initialization_no_bias():
    """Test that no bias parameter is created when bias=False."""
    layer = Linear(128, 256, bias=False)
    assert layer.bias is None


def test_parameter_count():
    """Test total parameter count."""
    in_features, out_features = 128, 256

    # With bias
    layer_with_bias = Linear(in_features, out_features, bias=True)
    expected_params_with_bias = (in_features * out_features) + out_features
    actual_params_with_bias = sum(p.numel() for p in layer_with_bias.parameters())
    assert actual_params_with_bias == expected_params_with_bias

    # Without bias
    layer_without_bias = Linear(in_features, out_features, bias=False)
    expected_params_without_bias = in_features * out_features
    actual_params_without_bias = sum(p.numel() for p in layer_without_bias.parameters())
    assert actual_params_without_bias == expected_params_without_bias


# ============================================================================
# Device and Dtype Tests
# ============================================================================


def test_device_cpu():
    """Test layer creation on CPU device."""
    device = torch.device("cpu")
    layer = Linear(128, 256, device=device)

    assert layer.weight.device == device
    if layer.bias is not None:
        assert layer.bias.device == device


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_device_cuda():
    """Test layer creation on CUDA device."""
    device = torch.device("cuda")
    layer = Linear(128, 256, device=device)

    assert layer.weight.device.type == "cuda"
    if layer.bias is not None:
        assert layer.bias.device.type == "cuda"


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_dtype_support(dtype):
    """Test layer creation with different dtypes."""
    layer = Linear(128, 256, dtype=dtype)

    assert layer.weight.dtype == dtype
    if layer.bias is not None:
        assert layer.bias.dtype == dtype


# ============================================================================
# Benchmark Tests
# ============================================================================


@pytest.mark.benchmark(group="forward_small")
def test_benchmark_forward_small_custom(benchmark):
    """Benchmark custom Linear forward pass with small features (128->128)."""
    in_features, out_features = 128, 128
    batch_size = 8

    layer = Linear(in_features, out_features)
    x = torch.randn(batch_size, in_features)

    benchmark(layer, x)


@pytest.mark.benchmark(group="forward_small")
def test_benchmark_forward_small_pytorch(benchmark):
    """Benchmark PyTorch Linear forward pass with small features (128->128)."""
    in_features, out_features = 128, 128
    batch_size = 8

    layer = torch.nn.Linear(in_features, out_features)
    x = torch.randn(batch_size, in_features)

    benchmark(layer, x)


@pytest.mark.benchmark(group="forward_medium")
def test_benchmark_forward_medium_custom(benchmark):
    """Benchmark custom Linear forward pass with medium features (512->512)."""
    in_features, out_features = 512, 512
    batch_size = 8

    layer = Linear(in_features, out_features)
    x = torch.randn(batch_size, in_features)

    benchmark(layer, x)


@pytest.mark.benchmark(group="forward_medium")
def test_benchmark_forward_medium_pytorch(benchmark):
    """Benchmark PyTorch Linear forward pass with medium features (512->512)."""
    in_features, out_features = 512, 512
    batch_size = 8

    layer = torch.nn.Linear(in_features, out_features)
    x = torch.randn(batch_size, in_features)

    benchmark(layer, x)


@pytest.mark.benchmark(group="forward_large")
def test_benchmark_forward_large_custom(benchmark):
    """Benchmark custom Linear forward pass with large features (4096->4096)."""
    in_features, out_features = 4096, 4096
    batch_size = 8

    layer = Linear(in_features, out_features)
    x = torch.randn(batch_size, in_features)

    benchmark(layer, x)


@pytest.mark.benchmark(group="forward_large")
def test_benchmark_forward_large_pytorch(benchmark):
    """Benchmark PyTorch Linear forward pass with large features (4096->4096)."""
    in_features, out_features = 4096, 4096
    batch_size = 8

    layer = torch.nn.Linear(in_features, out_features)
    x = torch.randn(batch_size, in_features)

    benchmark(layer, x)


@pytest.mark.benchmark(group="forward_backward")
def test_benchmark_forward_backward_custom(benchmark):
    """Benchmark custom Linear full training step (forward + backward)."""
    in_features, out_features = 512, 512
    batch_size = 8

    def training_step():
        layer = Linear(in_features, out_features)
        x = torch.randn(batch_size, in_features, requires_grad=True)
        output = layer(x)
        loss = output.sum()
        loss.backward()
        return loss

    benchmark(training_step)


@pytest.mark.benchmark(group="forward_backward")
def test_benchmark_forward_backward_pytorch(benchmark):
    """Benchmark PyTorch Linear full training step (forward + backward)."""
    in_features, out_features = 512, 512
    batch_size = 8

    def training_step():
        layer = torch.nn.Linear(in_features, out_features)
        x = torch.randn(batch_size, in_features, requires_grad=True)
        output = layer(x)
        loss = output.sum()
        loss.backward()
        return loss

    benchmark(training_step)
