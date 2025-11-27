"""
PyBE - Python Bessetti-Erskine
"""

__version__ = "0.1.0"
__author__ = "Robert Simpson"

# Import key functions/classes to make them available at package level
from .bessetti_erskine import gaussian_field, uniform_field

from .beam_pipe import beam_pipe_field, beam_pipe_gaussian_field, beam_pipe_uniform_field

# Optional: define what gets imported with "from PyBE import *"
__all__ = ['gaussian_field', 'uniform_field', 
           'beam_pipe_field', 'beam_pipe_gaussian_field',
           'beam_pipe_uniform_field']