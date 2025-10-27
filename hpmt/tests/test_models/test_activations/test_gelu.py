import pytest
import torch

from hpmt.models.activations.gelu import GELU

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def simple_input_1d():
    """1D input tensor (features only)."""
    return torch.randn(128)


@pytest.fixture
def simple_input_2d():
    """2D input tensor (batch, features)."""
    return torch.randn(8, 128)


@pytest.fixture
def simple_input_3d():
    """3D input tensor (batch, sequence, features) - common in transformers."""
    return torch.randn(4, 32, 512)


@pytest.fixture
def simple_input_4d():
    """4D input tensor (batch, channels, height, width)."""
    return torch.randn(2, 3, 16, 16)


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    """Parametrized fixture for CPU and CUDA devices."""
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device(request.param)


# ============================================================================
# Shape Tests
# ============================================================================


def test_output_shape_1d():
    """Test output shape with 1D input."""
    layer = GELU()
    x = torch.randn(128)
    output = layer(x)
    assert output.shape == x.shape == (128,)


def test_output_shape_2d():
    """Test output shape with 2D input (batch, features)."""
    layer = GELU()
    x = torch.randn(8, 128)
    output = layer(x)
    assert output.shape == x.shape == (8, 128)


def test_output_shape_3d():
    """Test output shape with 3D input (batch, seq_len, features)."""
    layer = GELU()
    x = torch.randn(4, 32, 512)
    output = layer(x)
    assert output.shape == x.shape == (4, 32, 512)


def test_output_shape_4d():
    """Test output shape with 4D input (batch, channels, height, width)."""
    layer = GELU()
    x = torch.randn(2, 3, 16, 16)
    output = layer(x)
    assert output.shape == x.shape == (2, 3, 16, 16)


def test_output_shape_cpu():
    """Test output shape and device on CPU."""
    layer = GELU()
    x = torch.randn(8, 128, device=torch.device("cpu"))
    output = layer(x)
    assert output.shape == x.shape
    assert output.device.type == "cpu"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_output_shape_cuda():
    """Test output shape and device on CUDA."""
    layer = GELU()
    x = torch.randn(8, 128, device=torch.device("cuda"))
    output = layer(x)
    assert output.shape == x.shape
    assert output.device.type == "cuda"


# ============================================================================
# Gradient Tests
# ============================================================================


def test_gradients_flow_through_layer():
    """Test that gradients flow through the layer."""
    layer = GELU()
    x = torch.randn(8, 128, requires_grad=True)
    output = layer(x)
    loss = output.sum()
    loss.backward()
    assert x.grad is not None
    assert x.grad.shape == x.shape


def test_gradients_correct_shape_1d():
    """Test gradient shapes with 1D input."""
    layer = GELU()
    x = torch.randn(128, requires_grad=True)
    output = layer(x)
    loss = output.sum()
    loss.backward()

    assert x.grad is not None
    assert x.grad.shape == x.shape == (128,)


def test_gradients_correct_shape_3d():
    """Test gradient shapes with 3D input (batch, seq_len, features)."""
    layer = GELU()
    x = torch.randn(4, 32, 512, requires_grad=True)
    output = layer(x)
    loss = output.sum()
    loss.backward()

    assert x.grad is not None
    assert x.grad.shape == x.shape == (4, 32, 512)


def test_gradients_no_explosion_or_vanishing():
    """Test that gradients are neither exploding nor vanishing."""
    layer = GELU()
    x = torch.randn(8, 128, requires_grad=True)
    output = layer(x)
    loss = output.sum()
    loss.backward()

    assert x.grad is not None
    assert not torch.isnan(x.grad).any()
    assert not torch.isinf(x.grad).any()
    assert (x.grad.abs() > 1e-10).any()


# ============================================================================
# Numerical Correctness Tests
# ============================================================================


@pytest.mark.parametrize(
    "batch_size,features",
    [
        (8, 128),  # Small
        (8, 512),  # Medium
        (8, 4096),  # Large
    ],
)
def test_numerical_correctness_2d(batch_size, features):
    """Test numerical correctness against torch.nn.GELU with 2D input."""
    x = torch.randn(batch_size, features)

    layer_custom = GELU()
    layer_torch = torch.nn.GELU(approximate="none")

    output_custom = layer_custom(x)
    output_torch = layer_torch(x)

    assert torch.allclose(output_custom, output_torch, rtol=1e-2, atol=1e-4)


def test_numerical_correctness_1d():
    """Test numerical correctness with 1D input."""
    x = torch.randn(128)

    layer_custom = GELU()
    layer_torch = torch.nn.GELU(approximate="none")

    output_custom = layer_custom(x)
    output_torch = layer_torch(x)

    assert torch.allclose(output_custom, output_torch, rtol=1e-2, atol=1e-4)


def test_numerical_correctness_3d():
    """Test numerical correctness with 3D input (batch, seq_len, features)."""
    x = torch.randn(4, 32, 512)

    layer_custom = GELU()
    layer_torch = torch.nn.GELU(approximate="none")

    output_custom = layer_custom(x)
    output_torch = layer_torch(x)

    assert torch.allclose(output_custom, output_torch, rtol=1e-2, atol=1e-4)


def test_numerical_correctness_4d():
    """Test numerical correctness with 4D input."""
    x = torch.randn(2, 3, 16, 16)

    layer_custom = GELU()
    layer_torch = torch.nn.GELU(approximate="none")

    output_custom = layer_custom(x)
    output_torch = layer_torch(x)

    assert torch.allclose(output_custom, output_torch, rtol=1e-2, atol=1e-4)


def test_numerical_edge_case_small_values():
    """Test with very small input values near zero."""
    x = torch.randn(8, 128) * 1e-7

    layer_custom = GELU()
    layer_torch = torch.nn.GELU(approximate="none")

    output_custom = layer_custom(x)
    output_torch = layer_torch(x)

    assert torch.allclose(output_custom, output_torch, rtol=1e-2, atol=1e-4)


def test_numerical_edge_case_large_values():
    """Test with large input values."""
    x = torch.randn(8, 128) * 1e3

    layer_custom = GELU()
    layer_torch = torch.nn.GELU(approximate="none")

    output_custom = layer_custom(x)
    output_torch = layer_torch(x)

    assert torch.allclose(output_custom, output_torch, rtol=1e-2, atol=1e-4)


def test_numerical_edge_case_zero_input():
    """Test with zero input."""
    x = torch.zeros(8, 128)

    layer_custom = GELU()
    layer_torch = torch.nn.GELU(approximate="none")

    output_custom = layer_custom(x)
    output_torch = layer_torch(x)

    assert torch.allclose(output_custom, output_torch, rtol=1e-2, atol=1e-4)


def test_numerical_edge_case_negative_input():
    """Test with all negative input values."""
    x = -torch.abs(torch.randn(8, 128))

    layer_custom = GELU()
    layer_torch = torch.nn.GELU(approximate="none")

    output_custom = layer_custom(x)
    output_torch = layer_torch(x)

    assert torch.allclose(output_custom, output_torch, rtol=1e-2, atol=1e-4)


def test_numerical_edge_case_positive_input():
    """Test with all positive input values."""
    x = torch.abs(torch.randn(8, 128))

    layer_custom = GELU()
    layer_torch = torch.nn.GELU(approximate="none")

    output_custom = layer_custom(x)
    output_torch = layer_torch(x)

    assert torch.allclose(output_custom, output_torch, rtol=1e-2, atol=1e-4)


def test_numerical_edge_case_mixed_signs():
    """Test with mixed positive and negative values."""
    x = torch.randn(8, 128)

    layer_custom = GELU()
    layer_torch = torch.nn.GELU(approximate="none")

    output_custom = layer_custom(x)
    output_torch = layer_torch(x)

    assert torch.allclose(output_custom, output_torch, rtol=1e-2, atol=1e-4)


# ============================================================================
# Numerical Stability Tests
# ============================================================================


def test_stability_extreme_positive_values():
    """Test with extreme positive values."""
    x = torch.randn(8, 128) * 1e6

    layer = GELU()
    output = layer(x)

    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()


def test_stability_extreme_negative_values():
    """Test with extreme negative values."""
    x = -torch.abs(torch.randn(8, 128)) * 1e6

    layer = GELU()
    output = layer(x)

    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()


def test_stability_large_batch_size():
    """Test with very large batch size."""
    x = torch.randn(1024, 512)

    layer = GELU()
    output = layer(x)

    assert output.shape == x.shape
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()


def test_stability_repeated_forward_passes():
    """Test accumulation over many forward passes."""
    layer = GELU()
    x = torch.randn(8, 128, requires_grad=True)

    for _ in range(100):
        output = layer(x)
        loss = output.sum()
        loss.backward()

        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

        x.grad.zero_()


# ============================================================================
# Device and Dtype Tests
# ============================================================================


def test_device_cpu():
    """Test layer works on CPU device."""
    device = torch.device("cpu")
    layer = GELU()
    x = torch.randn(8, 128, device=device)

    output = layer(x)

    assert output.device == device


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_device_cuda():
    """Test layer works on CUDA device."""
    device = torch.device("cuda")
    layer = GELU()
    x = torch.randn(8, 128, device=device)

    output = layer(x)

    assert output.device.type == "cuda"


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_dtype_support(dtype):
    """Test layer works with different dtypes."""
    layer = GELU()
    x = torch.randn(8, 128, dtype=dtype)

    output = layer(x)

    assert output.dtype == dtype


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_mixed_precision_cuda(dtype):
    """Test layer works with mixed precision on CUDA."""
    device = torch.device("cuda")
    layer = GELU()
    x = torch.randn(8, 128, dtype=dtype, device=device)

    output = layer(x)

    assert output.dtype == dtype
    assert output.device.type == "cuda"


# ============================================================================
# Benchmark Tests
# ============================================================================


@pytest.mark.benchmark(group="forward_small")
def test_benchmark_forward_small_custom(benchmark):
    """Benchmark custom GELU forward pass with small input (128)."""
    features = 128
    batch_size = 8

    layer = GELU()
    x = torch.randn(batch_size, features)

    benchmark(layer, x)


@pytest.mark.benchmark(group="forward_small")
def test_benchmark_forward_small_pytorch(benchmark):
    """Benchmark PyTorch GELU forward pass with small input (128)."""
    features = 128
    batch_size = 8

    layer = torch.nn.GELU(approximate="none")
    x = torch.randn(batch_size, features)

    benchmark(layer, x)


@pytest.mark.benchmark(group="forward_medium")
def test_benchmark_forward_medium_custom(benchmark):
    """Benchmark custom GELU forward pass with medium input (512)."""
    features = 512
    batch_size = 8

    layer = GELU()
    x = torch.randn(batch_size, features)

    benchmark(layer, x)


@pytest.mark.benchmark(group="forward_medium")
def test_benchmark_forward_medium_pytorch(benchmark):
    """Benchmark PyTorch GELU forward pass with medium input (512)."""
    features = 512
    batch_size = 8

    layer = torch.nn.GELU(approximate="none")
    x = torch.randn(batch_size, features)

    benchmark(layer, x)


@pytest.mark.benchmark(group="forward_large")
def test_benchmark_forward_large_custom(benchmark):
    """Benchmark custom GELU forward pass with large input (4096)."""
    features = 4096
    batch_size = 8

    layer = GELU()
    x = torch.randn(batch_size, features)

    benchmark(layer, x)


@pytest.mark.benchmark(group="forward_large")
def test_benchmark_forward_large_pytorch(benchmark):
    """Benchmark PyTorch GELU forward pass with large input (4096)."""
    features = 4096
    batch_size = 8

    layer = torch.nn.GELU(approximate="none")
    x = torch.randn(batch_size, features)

    benchmark(layer, x)


@pytest.mark.benchmark(group="forward_backward")
def test_benchmark_forward_backward_custom(benchmark):
    """Benchmark custom GELU full training step (forward + backward)."""
    features = 512
    batch_size = 8

    def training_step():
        layer = GELU()
        x = torch.randn(batch_size, features, requires_grad=True)
        output = layer(x)
        loss = output.sum()
        loss.backward()
        return loss

    benchmark(training_step)


@pytest.mark.benchmark(group="forward_backward")
def test_benchmark_forward_backward_pytorch(benchmark):
    """Benchmark PyTorch GELU full training step (forward + backward)."""
    features = 512
    batch_size = 8

    def training_step():
        layer = torch.nn.GELU(approximate="none")
        x = torch.randn(batch_size, features, requires_grad=True)
        output = layer(x)
        loss = output.sum()
        loss.backward()
        return loss

    benchmark(training_step)
