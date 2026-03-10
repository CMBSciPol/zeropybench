"""Sphinx configuration file for zeropybench documentation."""

import sys
from importlib.metadata import version as get_version
from pathlib import Path

# Add the src directory to the Python path
src_path = Path(__file__).parent.parent.parent / 'src'
sys.path.insert(0, str(src_path))

# Project information
project = 'zeropybench'
copyright = '2025, Pierre Chanial'
author = 'Pierre Chanial'
release = get_version('zeropybench')
version = '.'.join(release.split('.')[:2])  # Major.minor

# General configuration
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'myst_nb',
]

templates_path = ['_templates']
exclude_patterns = []

# HTML output configuration
html_theme = 'furo'
html_static_path = ['_static']
html_title = f'{project} v{version}'
html_theme_options = {
    'light_css_variables': {
        'color-brand-primary': '#2EBF4F',
        'color-brand-content': '#2EBF4F',
    },
    'dark_css_variables': {
        'color-brand-primary': '#34D058',
        'color-brand-content': '#34D058',
    },
}

# autodoc settings
autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'show-inheritance': True,
    'special-members': '__init__',
}
autodoc_typehints = 'description'  # Show type hints in parameter descriptions

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False

# MyST parser configuration
myst_enable_extensions = [
    'colon_fence',
    'deflist',
    'dollarmath',
    'html_admonition',
    'html_image',
    'linkify',
    'replacements',
    'smartquotes',
    'substitution',
    'tasklist',
]

# Intersphinx mapping
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'jax': ('https://jax.readthedocs.io/en/latest/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'polars': ('https://docs.pola.rs/api/python/stable/', None),
}

# myst-nb configuration
nb_execution_mode = 'auto'  # Execute notebooks only if no outputs exist
nb_execution_raise_on_error = False  # Continue building even if there are errors
nb_execution_timeout = 300  # Timeout for notebook execution in seconds
nb_merge_streams = True  # Merge consecutive stdout/stderr outputs


def setup(app):
    app.add_css_file('custom.css')
