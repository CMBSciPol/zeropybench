from importlib.metadata import version as _version

from ._benchmark import Benchmark

__all__ = ['Benchmark']
__version__ = _version('zeropybench')
