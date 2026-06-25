"""Tesseract — a C++20 deep-learning framework, Python frontend.

Thin re-export layer over the compiled ``tesseract._core`` pybind11 extension
(M4 Track B1 / B-041). Everything here is a shim; all logic lives in the C++
libraries.

Example::

    import numpy as np
    import tesseract as ts

    x = ts.Tensor(np.random.randn(4, 8).astype("float32"))
    model = ts.nn.Sequential([ts.nn.Linear(8, 16), ts.nn.ReLU(),
                              ts.nn.Linear(16, 3)])
    logits = model(x)
"""

from . import _core
from ._core import (
    Tensor,
    DType,
    Device,
    backward,
    ops,
    nn,
    optim,
    models,
    io,
)

float32 = DType.float32
float64 = DType.float64
int64 = DType.int64
int32 = DType.int32

__all__ = [
    "Tensor",
    "DType",
    "Device",
    "backward",
    "ops",
    "nn",
    "optim",
    "models",
    "io",
    "float32",
    "float64",
    "int64",
    "int32",
]

__version__ = "0.1.0"
