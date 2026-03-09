# Installation

## Requirements

- Python >= 3.10
- polars
- matplotlib

## Install from PyPI

```bash
pip install zeropybench
```

## Install from Source

```bash
git clone https://github.com/CMBSciPol/zeropybench.git
cd zeropybench
pip install .
```

## Development Installation

For development, install with the `dev` dependency group:

```bash
git clone https://github.com/CMBSciPol/zeropybench.git
cd zeropybench
pip install --group dev .
```

This includes additional tools for testing and development.

## JAX Support (Optional)

ZeroPyBench automatically detects and optimizes benchmarking for JAX arrays. To use JAX features, install JAX separately:

```bash
pip install jax
```

## CUDA Support (Optional)

For GPU acceleration with JAX:

```bash
pip install zeropybench --group cuda12
```

or for CUDA 13:

```bash
pip install zeropybench --group cuda13
```

## Troubleshooting

### JAX Installation Issues

If JAX fails to install, you may need to install it separately first:

```bash
pip install jax jaxlib
```

For specific hardware (GPU/TPU), see the [JAX installation guide](https://github.com/google/jax#installation).

### Import Errors

If you get import errors related to `jaxlib`, ensure you have compatible versions of JAX and jaxlib:

```bash
pip install --upgrade jax jaxlib
```
