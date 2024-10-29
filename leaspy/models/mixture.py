import pandas as pd
import torch

from leaspy.models.base import InitializationMethod
from leaspy.models.multivariate import LogisticMultivariateModel
from leaspy.models.obs_models import observation_model_factory

from leaspy.io.data.dataset import Dataset

from leaspy.variables.state import State
from leaspy.variables.specs import (
	NamedVariables,
	ModelParameter,
	PopulationLatentVariable,
	LinkedVariable,
	Hyperparameter,
	SuffStatsRW,
	VariableBaluesR0,
)
from leaspy.variables.distributions import Normal

from leaspy.utils.docs import doc_with_super
from leaspy.utils.functional import Exp, Sqr, Orthobasis, MatMul
from leaspy.utils.typing import KwargsType, DictParams, Optional
from leaspy.utils.weighted_tensor import WeightedTensor, TensorOrWeightedTensor, unsqueeze_rig
ht

from leaspy.exceptions import LeaspyInputError

@doc_with_super()
class MixtureModel(LogisticMultivariateModel):
   """
   Manifold model for multiple variables of interest

   Parameters
   ----------
   name : :obj: `str`
      The name of the model
   base_model: str
      The base model for the mixture
   n_clusters : int
      The number of models in the mixture
   **kwargs
      Hyperparameters of the model

   Attributes
   ----------

   Raises
   ------
   :exc `.LeaspyModelInputError`
      * If `name` is not one of the allowed sub-type 'mixture_linear' or 'mixture_logistic'
      * If hyperparameters are inconsistent
   """
