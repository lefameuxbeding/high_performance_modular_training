# High Performance Modular Training

![Python Version](https://img.shields.io/badge/python-3.13-blue.svg)
[![Code Quality](https://github.com/beding/high_performance_modular_training/actions/workflows/code-quality.yaml/badge.svg)](https://github.com/lefameuxbeding/high_performance_modular_training/actions/workflows/code-quality.yaml)

Educational framework for high-performance LLM training.

## Getting Started

This project uses [uv](https://github.com/astral-sh/uv) for fast, reliable Python package management.

### Installation

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install project dependencies
uv sync

# Install pre-commit hooks
uv run pre-commit install
```

### Code Quality

```bash
# Auto-fix linting and formatting issues
uv run ruff check --fix .                              # Fix linting issues
uv run ruff format .                                   # Format code
uv run taplo fmt                                       # Format TOML files

# Check for issues (no auto-fix)
uv run ruff check .                                    # Linting
uv run pyright                                         # Type checking
uv run taplo check                                     # TOML validation
uv run yamllint -c .yamllint .                         # YAML validation
```

### Development

```bash
# Add a new dependency
uv add package-name

# Add a development dependency
uv add --dev package-name

# Update dependencies
uv lock --upgrade
```
