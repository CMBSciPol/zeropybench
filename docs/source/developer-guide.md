# Development

## Development Setup

1. Clone the repository:
```bash
git clone https://github.com/CMBSciPol/zeropybench.git
cd zeropybench
```

2. Install in development mode:
```bash
pip install --group dev -e .
or
uv sync
```

The dependency groups `cuda12` and `cuda13` can be used to quickly install JAX with cuda support:
```bash
pip install --group dev --group cuda13 -e .
or
uv sync --group cuda13
```

3. Install `pre-commit` (or equivalently `prek`):
```bash
pipx --user install pre-commit
or
uv tool install pre-commit
```
The pre-commit hooks need to be installed (see more details in the section [Code Quality](#code-quality)). After doing so, they will be run at each commit.
```
pre-commit install
```

## Running Tests

The project uses pytest for testing:

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run notebook tests
pytest docs/source/user-guide/

# Run specific test file
pytest tests/test_jax.py
```

The test suite includes:
- **Unit tests**: Individual function testing
- **JAX tests**: JAX-specific benchmarking tests
- **Functional tests**: The documentation notebooks

## Code Quality

The project uses several pre-commit hooks for code quality. They include
- ruff (linting and formatting)
- mypy (type checking)
- and others

The pre-commit will be run against the files that are part of a Git commit. To run against all files
(modified but not added or already committed):
```bash
pre-commit run --all-files
```

## Documentation

Build documentation locally:
The dependencies required to build the documentation have already been installed through `uv sync`. If you prefer using
pip, this extra step is required:
```bash
pip install --group docs .
```

Then, the HTML documentation is built and inspected by:
```
cd docs
make html
firefox build/html/index.html
```

The documentation uses:
- **Sphinx**: Documentation generator
- **MyST**: Markdown support in Sphinx
- **Furo**: Clean, responsive theme
- **nbsphinx**: Jupyter notebook support


## Project Structure

```
zeropybench/
├── src/zeropybench/        # Main package
│   ├── __init__.py            # Public API exports
│   ├── _benchmark.py          # Benchmark class
│   ├── _io.py                 # Read / write utilities
│   ├── _jax.py                # JAX code transformation utilities
│   ├── _plot.py               # Plotting utilities
│   └── _units.py              # Unit conversion utilities
├── tests/                  # Unit test suite
├── docs/                   # Documentation
│   └── source/
│       ├── user-guide/        # User guide and examples
│       ├── api/               # API reference
│       └── developer-guide.md # This file
├── .github/                # CI/CD workflows
├── .pre-commit-config.yaml # Pre-commit hook configuration
└── pyproject.toml          # Project configuration
```

## Contributing

1. Fork the repository
2. Install pre-commit
3. Create a feature branch: `git checkout -b feature-name`
4. Make your changes
5. Add tests for new functionality
6. Ensure all tests pass: `pytest`
7. Submit a pull request

## Release Process

Releases are automated through GitHub Actions:

1. Create a new tag: `git tag v0.x.x`
2. Push the tag: `git push origin v0.x.x`
3. Create a GitHub release
4. The CI will automatically build and publish to PyPI
