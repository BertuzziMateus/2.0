from imports import *

"""
Backend selector module.
Automatically chooses CuPy (GPU) if available, otherwise falls back to NumPy (CPU).
"""

def get_array_module():
    """Return the array module: CuPy if a CUDA GPU is available, else NumPy."""
    try:
        import cupy as cp

        try:
            # Check whether at least one CUDA-capable GPU exists
            gpu_count = cp.cuda.runtime.getDeviceCount()
            if gpu_count > 0:

                return cp
        except cp.cuda.runtime.CUDARuntimeError:
            pass

        # If something failed, fall back to NumPy
        import numpy as np
        return np

    except ImportError:
        import numpy as np
        return np
