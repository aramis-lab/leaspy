"""Define some useful type aliases for static type checks and better input understanding."""

from typing import *

import torch

# Generic dictionary of keyword arguments
KwargsType = Dict[str, Any]

# Type for identifier of individuals
IDType = str

# Type for parameters / variables (mostly in dictionary)
ParamType = str

# Type for feature names
FeatureType = str

DictParams = Dict[ParamType, Any]
DictParamsTorch = Dict[ParamType, torch.Tensor]
