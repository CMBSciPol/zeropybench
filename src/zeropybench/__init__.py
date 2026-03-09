from importlib.metadata import version as _version

from ._benchmark import Benchmark, read_benchmark

__all__ = ['Benchmark', 'read_benchmark']
__version__ = _version('zeropybench')
