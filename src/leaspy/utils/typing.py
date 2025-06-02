"""Define some useful type aliases for static type checks and better input understanding."""

from typing import Any

import torch

__all__ = [
    "KwargsType",
    "IDType",
    "ParamType",
    "FeatureType",
    "DictParams",
    "DictParamsTorch",
]

# Generic dictionary of keyword arguments
KwargsType = dict[str, Any]

# Type for identifier of individuals
IDType = str

# Type for parameters / variables (mostly in dictionary)
ParamType = str

# Type for feature names
FeatureType = str

DictParams = dict[ParamType, Any]
DictParamsTorch = dict[ParamType, torch.Tensor]
