"""This module defines the distributions used for sampling variables."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, ClassVar, Type

import torch
from torch import Tensor
from torch.autograd import grad
from torch.distributions.mixture_same_family import MixtureSameFamily
from torch.distributions.multivariate_normal import MultivariateNormal

from leaspy.constants import constants
from leaspy.exceptions import LeaspyInputError
from leaspy.utils.distributions import MultinomialDistribution
from leaspy.utils.functional import NamedInputFunction
from leaspy.utils.weighted_tensor import WeightedTensor, sum_dim

__all__ = [
    "StatelessDistributionFamily",
    "StatelessDistributionFamilyFromTorchDistribution",
    "BernoulliFamily",
    "NormalFamily",
    "AbstractWeibullRightCensoredFamily",
    "WeibullRightCensoredFamily",
    "WeibullRightCensoredWithSourcesFamily",
    "SymbolicDistribution",
    "Normal",
    "Bernoulli",
    "WeibullRightCensored",
    "WeibullRightCensoredWithSources",
    # "CategoricalFamily",
    "MixtureNormalFamily",
    "MultivariateNormalFamily"
]


class StatelessDistributionFamily(ABC):
    """Interface to represent stateless distribution families (i.e. no distribution parameters are stored in instance).

    TODO / WIP? allow WeightedTensor for parameters as well?
    (e.g. `batched_deltas = Normal(batched_deltas_mean, ...)` which should be masked at some indices)
    --> mask at latent pop. variable level (`batched_deltas`) or
        directly at model parameter level `batched_deltas_mean`?
    """

    parameters: ClassVar[tuple[str, ...]]

    @classmethod
    @abstractmethod
    def validate_parameters(cls, *params: Any) -> tuple[torch.Tensor, ...]:
        """
        Validate consistency of distribution parameters,
        returning them with out-of-place modifications if needed.
        """

    @classmethod
    def shape(cls, *params_shapes: tuple[int, ...]) -> tuple[int, ...]:
        """
        Shape of distribution samples (without any additional expansion),
        given shapes of distribution parameters.
        """
        # We provide a default implementation which should fit for most cases
        n_params = len(params_shapes)
        if n_params != len(cls.parameters):
            raise LeaspyInputError(
                f"Expecting {len(cls.parameters)} parameters but got {n_params}"
            )
        if n_params == 0:
            raise NotImplementedError(
                "No way to infer shape of samples since no parameter"
            )
        if n_params == 1:
            return params_shapes[0]
        return torch.broadcast_shapes(*params_shapes)

    @classmethod
    @abstractmethod
    def sample(
        cls,
        *params: torch.Tensor,
        sample_shape: tuple[int, ...] = (),
    ) -> torch.Tensor:
        """
        Sample values, given distribution parameters (`sample_shape` is
        prepended to shape of distribution parameters).
        """

    @classmethod
    @abstractmethod
    def mode(cls, *params: torch.Tensor) -> torch.Tensor:
        """
        Mode of distribution (returning first value if discrete ties),
        given distribution parameters.
        """

    @classmethod
    @abstractmethod
    def mean(cls, *params: torch.Tensor) -> torch.Tensor:
        """Mean of distribution (if defined), given distribution parameters."""

    @classmethod
    @abstractmethod
    def stddev(cls, *params: torch.Tensor) -> torch.Tensor:
        """Standard-deviation of distribution (if defined), given distribution parameters."""

    @classmethod
    @abstractmethod
    def _nll(cls, x: WeightedTensor, *params: torch.Tensor) -> WeightedTensor:
        """Negative log-likelihood of value, given distribution parameters."""

    @classmethod
    @abstractmethod
    def _nll_jacobian(cls, x: WeightedTensor, *params: torch.Tensor) -> WeightedTensor:
        """Jacobian w.r.t. value of negative log-likelihood, given distribution parameters."""

    @classmethod
    def _nll_and_jacobian(
        cls,
        x: WeightedTensor,
        *params: torch.Tensor,
    ) -> tuple[WeightedTensor, WeightedTensor]:
        """Negative log-likelihood of value and its jacobian w.r.t. value, given distribution parameters."""
        # not efficient implementation by default
        return cls._nll(x, *params), cls._nll_jacobian(x, *params)

    @classmethod
    def nll(
        cls,
        x: WeightedTensor[float],
        *params: torch.Tensor,
    ) -> WeightedTensor[float]:
        """Negative log-likelihood of value, given distribution parameters."""
        return cls._nll(x, *params)

    @classmethod
    def regularization(
        cls,
        x: torch.Tensor,
        *params: torch.Tensor,
    ) -> WeightedTensor[float]:
        """Negative log-likelihood of value, given distribution parameters."""

        if isinstance (x, Tensor):
            regul = cls._nll(WeightedTensor(x), *params)
        else:
            regul = cls._nll(x, *params)
        return regul

    @classmethod
    def nll_jacobian(
        cls,
        x: WeightedTensor[float],
        *params: torch.Tensor,
    ) -> WeightedTensor[float]:
        """Jacobian w.r.t. value of negative log-likelihood, given distribution parameters."""
        return cls._nll_jacobian(x, *params)

    @classmethod
    def nll_and_jacobian(
        cls,
        x: WeightedTensor[float],
        *params: torch.Tensor,
    ) -> tuple[WeightedTensor[float], WeightedTensor[float]]:
        """Negative log-likelihood of value and its jacobian w.r.t. value, given distribution parameters."""
        return cls._nll_and_jacobian(x, *params)


class StatelessDistributionFamilyFromTorchDistribution(StatelessDistributionFamily):
    """Wrapper to build a `StatelessDistributionFamily` class from an existing torch distribution class."""

    dist_factory: ClassVar[Callable[..., torch.distributions.Distribution]]

    @classmethod
    def validate_parameters(cls, *params: Any) -> tuple[torch.Tensor, ...]:
        """Validate consistency of distribution parameters, returning them with out-of-place modifications if needed.

        Parameters
        ----------
        params : Any
            The parameters to pass to the distribution factory.

        Returns
        -------
        :obj:`tuple` [ :class:`torch.Tensor`, ...] :
            The validated parameters.
        """
        distribution = cls.dist_factory(*params, validate_args=True)
        return tuple(getattr(distribution, parameter) for parameter in cls.parameters)

    @classmethod
    def sample(
        cls,
        *params: torch.Tensor,
        sample_shape: tuple[int, ...] = (),
    ) -> torch.Tensor:
        return cls.dist_factory(*params).sample(sample_shape)

    @classmethod
    def mode(cls, *params: torch.Tensor) -> torch.Tensor:
        """Mode of distribution (returning first value if discrete ties), given distribution parameters.

        Parameters
        ----------
        params : :class:`torch.Tensor`
            The distribution parameters.

        Returns
        -------
        :class:`torch.Tensor` :
            The value of the distribution's mode.
        """
        raise NotImplementedError("Not provided in torch.Distribution interface")

    @classmethod
    def mean(cls, *params: torch.Tensor) -> torch.Tensor:
        """Return the mean of the distribution, if defined.

        Parameters
        ----------
        params : :class:`torch.Tensor`
            The distribution parameters.

        Returns
        -------
        :class:`torch.Tensor` :
            The value of the distribution's mean.
        """
        return cls.dist_factory(*params).mean

    @classmethod
    def stddev(cls, *params: torch.Tensor) -> torch.Tensor:
        """Return the standard-deviation of the distribution.

        Parameters
        ----------
        params : :class:`torch.Tensor`
            The distribution parameters.

        Returns
        -------
        :class:`torch.Tensor` :
            The value of the distribution's standard deviation.
        """
        return cls.dist_factory(*params).stddev

    @classmethod
    def _nll(cls, x: WeightedTensor, *params: torch.Tensor) -> WeightedTensor:
        return WeightedTensor(-cls.dist_factory(*params).log_prob(x.value), x.weight)

    @classmethod
    def _nll_and_jacobian(
        cls,
        x: WeightedTensor,
        *params: torch.Tensor,
    ) -> tuple[WeightedTensor, WeightedTensor]:
        nll = cls._nll(x, *params)
        (nll_grad_value,) = grad(nll.value, (x.value,), create_graph=x.requires_grad)
        return nll, WeightedTensor(nll_grad_value, x.weight)

    @classmethod
    def _nll_jacobian(cls, x: WeightedTensor, *params: torch.Tensor) -> WeightedTensor:
        return cls._nll_and_jacobian(x, *params)[1]


class BernoulliFamily(StatelessDistributionFamilyFromTorchDistribution):
    """Bernoulli family (stateless)."""

    parameters: ClassVar = ("loc",)
    dist_factory: ClassVar = torch.distributions.Bernoulli


class NormalFamily(StatelessDistributionFamilyFromTorchDistribution):
    """Normal / Gaussian family (stateless)."""

    parameters: ClassVar = ("loc", "scale")
    dist_factory: ClassVar = torch.distributions.Normal
    nll_constant_standard: ClassVar = 0.5 * torch.log(2 * torch.tensor(math.pi))

    @classmethod
    def mode(cls, loc: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        """Return the mode of the distribution given the distribution's loc and scale parameters.

        Parameters
        ----------
        loc : :class:`torch.Tensor`
            The distribution loc.

        scale : :class:`torch.Tensor`
            The distribution scale.

        Returns
        -------
        :class:`torch.Tensor` :
            The value of the distribution's mode.
        """
        # `loc`, but with possible broadcasting of shape
        return torch.broadcast_tensors(loc, scale)[0]

    @classmethod
    def mean(cls, loc: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        """Return the mean of the distribution, given the distribution loc and scale parameters.

        Parameters
        ----------
        loc : :class:`torch.Tensor`
            The distribution loc parameters.

        scale : :class:`torch.Tensor`
            The distribution scale parameters.

        Returns
        -------
        :class:`torch.Tensor` :
            The value of the distribution's mean.
        """
        # Hardcode method for efficiency
        # `loc`, but with possible broadcasting of shape
        return torch.broadcast_tensors(loc, scale)[0]

    @classmethod
    def stddev(cls, loc: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        """Return the standard-deviation of the distribution, given loc and scale of the distribution.

        Parameters
        ----------
        loc : :class:`torch.Tensor`
            The distribution loc parameter.

        scale : :class:`torch.Tensor`
            The distribution scale parameter.

        Returns
        -------
        :class:`torch.Tensor` :
            The value of the distribution's standard deviation.
        """
        # Hardcode method for efficiency
        # `scale`, but with possible broadcasting of shape
        return torch.broadcast_tensors(loc, scale)[1]

    @classmethod
    def _nll(
        cls, x: WeightedTensor, loc: torch.Tensor, scale: torch.Tensor
    ) -> WeightedTensor:
        # Hardcode method for efficiency
        return WeightedTensor(
            (
                0.5 * ((x.value - loc) / scale) ** 2
                + torch.log(scale)
                + cls.nll_constant_standard
            ),
            x.weight,
        )

    @classmethod
    def _nll_jacobian(
        cls, x: WeightedTensor, loc: torch.Tensor, scale: torch.Tensor
    ) -> WeightedTensor:
        # Hardcode method for efficiency
        return WeightedTensor((x.value - loc) / scale**2, x.weight)

    @classmethod
    def _nll_and_jacobian(
        cls,
        x: WeightedTensor,
        loc: torch.Tensor,
        scale: torch.Tensor,
    ) -> tuple[WeightedTensor, WeightedTensor]:
        # Hardcode method for efficiency
        z = (x.value - loc) / scale
        nll = 0.5 * z**2 + torch.log(scale) + cls.nll_constant_standard
        return WeightedTensor(nll, x.weight), WeightedTensor(z / scale, x.weight)

    # @classmethod
    # def sample(cls, loc, scale, *, sample_shape = ()):
    #    # Hardcode method for efficiency? (<!> broadcasting)

class MultivariateNormalFamily(StatelessDistributionFamily):
    """Multivariate Normal family with diagonal covariance (stateless)."""

    parameters: ClassVar = ("loc", "scale")  # scale = stddev for each dim
    dist_factory: ClassVar = MultivariateNormal
    nll_constant_standard: ClassVar = 0.5 * torch.log(2 * torch.tensor(math.pi))

    @classmethod
    def multi_dist_factory(cls,
                     loc: torch.Tensor,
                     scale: torch.Tensor,
                     ) -> MultivariateNormal:

        loc = torch.tensor(loc, dtype=torch.float32)
        scale = torch.diag(scale)
        scale = torch.tensor(scale, dtype=torch.float32)

        return MultivariateNormal(loc, scale)
    """
    @classmethod
    def sample(
        cls,
        *params: torch.Tensor,
        sample_shape: tuple[int, ...] = (),
    ) -> torch.Tensor:

        multi_dist = cls.multi_dist_factory(*params)
        print(multi_dist)

        return multi_dist.sample(sample_shape)
    """
    @classmethod
    def mode(cls, loc: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        return loc  # mode = mean for Gaussian

    @classmethod
    def mean(cls, loc: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        return loc

    @classmethod
    def stddev(cls, loc: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        return scale

    @classmethod
    def _nll(
        cls, x: WeightedTensor, loc: torch.Tensor, scale: torch.Tensor
    ) -> WeightedTensor:

        if x.value.ndimension() != loc.ndimension():
            if loc.ndimension()==1:
                x_value_expanded = x.value.unsqueeze(1).repeat(1, loc.shape[0], 1, 1)
                #x_weight_expanded = x.weight.unsqueeze(1).repeat(1, loc.shape[0], 1, 1)
                x = WeightedTensor(x_value_expanded)
            else:
                x_value_expanded = x.value.unsqueeze(1).repeat(1, loc.shape[1], 1, 1)
                x_weight_expanded = x.weight.unsqueeze(1).repeat(1, loc.shape[1], 1, 1)
                x = WeightedTensor(x_value_expanded, x_weight_expanded)

        #print('loc.shape : ', loc.shape)
        z = (x.value - loc) / scale
        LOG_2PI = torch.log(torch.tensor(2 * torch.pi))

        nll = 0.5 * z ** 2 + torch.log(scale) + LOG_2PI
        #nll = 0.5 * torch.sum(z**2 + 2 * torch.log(scale) + torch.log(2 * torch.pi), dim=-1)
        if x.value.ndimension() != nll.ndimension():
            x_value_expanded = x.value.unsqueeze(1).repeat(1, x.shape[0], 1, 1)
            x_weight_expanded = x.weight.unsqueeze(1).repeat(1, x.shape[0], 1, 1)
            x = WeightedTensor(x_value_expanded, x_weight_expanded)

            z = (x.value - loc) / scale
            LOG_2PI = torch.log(torch.tensor(2 * torch.pi))

            nll = 0.5 * z ** 2 + torch.log(scale) + LOG_2PI

        return WeightedTensor(nll, x.weight)

    @classmethod
    def _nll_jacobian(
        cls, x: WeightedTensor, loc: torch.Tensor, scale: torch.Tensor
    ) -> WeightedTensor:
        # Gradient of NLL w.r.t loc (mean), assumes diagonal covariance
        grad = (x.value - loc) / scale**2
        return WeightedTensor(grad, x.weight)

    @classmethod
    def _nll_and_jacobian(
        cls,
        x: WeightedTensor,
        loc: torch.Tensor,
        scale: torch.Tensor,
    ) -> tuple[WeightedTensor, WeightedTensor]:
        z = (x.value - loc) / scale
        LOG_2PI = torch.log(torch.tensor(2 * torch.pi))

        nll = 0.5 * z ** 2 + torch.log(scale) + LOG_2PI
        #nll = 0.5 * torch.sum(z**2 + 2 * torch.log(scale) + torch.log(2 * torch.pi), dim=-1)
        grad = z / scale
        return WeightedTensor(nll, x.weight), WeightedTensor(grad, x.weight)


# class CategoricalFamily(StatelessDistributionFamilyFromTorchDistribution):
#    """
#    Categorical family (stateless).
#    """

#    parameters: ClassVar = ("probs",)
#    dist_factory: ClassVar = torch.distributions.Categorical

#    @classmethod
#    def extract_n_clusters(cls, probs: torch.Tensor) -> int:
#        return probs.size()[0]

#    @classmethod
#    def mixing_probabilities (cls, probs: torch.Tensor) -> torch.Tensor:
#        return torch.tensor([probs])


class MixtureNormalFamily(StatelessDistributionFamily):
    """
    A stateless mixture of univariate normal distributions.

    This class defines a mixture distribution where each component is a univariate
    normal distribution, and the mixture weights are defined by a categorical distribution.

    Parameters:
        - loc (Tensor): Mean of each normal component, one for each cluster.
        - scale (Tensor): Standard deviation of each normal component.
        - probs (Tensor): Probabilities associated with each cluster; must sum to 1.

    The mixture is modeled using `torch.distributions.MixtureSameFamily`, where:
        - The mixing distribution is a `Categorical` defined by `probs`.
        - The component distribution is `Normal(loc, scale)`.
    """

    parameters: ClassVar = ("loc", "scale", "probs")
    nll_constant_standard: ClassVar = 0.5 * torch.log(2 * torch.tensor(math.pi))

    @classmethod
    def dist_factory(
        cls,
        loc: torch.Tensor,
        scale: torch.Tensor,
        probs: torch.Tensor,
    ) -> MixtureSameFamily:
        """
        Construct a MixtureSameFamily distribution of univariate normal components.

        Parameters
        ----------
        loc : :class:`torch.Tensor`
            Mean(s) of the normal components. Shape should be broadcastable with `scale` and `probs`.
        scale : :class:`torch.Tensor`
            Standard deviation(s) of the normal components.
        probs : :class:`torch.Tensor`
            Probabilities of each mixture component. Must sum to 1 along the component axis.

        Returns
        -------
        :class: `.MixtureSameFamily`
            A mixture distribution with categorical mixing and normal components.
        """
        from torch.distributions import Categorical, Normal

        return MixtureSameFamily(
            Categorical(probs),
            Normal(loc, scale),
        )

    @classmethod
    def sample(cls, *params: torch.Tensor, sample_shape: tuple[int, ...] = ()) -> torch.Tensor:
        """
        Draw samples from the mixture of normal distributions.

        Parameters
        ----------
        *params : :class:`torch.Tensor`
            Distribution parameters in the order (loc, scale, probs). These should be
            broadcastable to define a valid MixtureSameFamily distribution.
        sample_shape : :obj:`tuple` of :obj:`int`, optional
            The desired sample shape. Defaults to an empty tuple for a single sample.

        Returns
        -------
        :class:`torch.Tensor`
            A tensor of samples drawn from the specified mixture distribution.
            The shape is `sample_shape + batch_shape`.
        """
        dist = cls.dist_factory(*params)

        return dist.sample(sample_shape)

    @classmethod
    def set_component_distribution(
        cls,
        component_distribution: torch.distributions,
        loc: torch.Tensor,
        scale: torch.Tensor,
    ) -> torch.distributions:
        """
        Ensure that the component distribution is an instance of the :class:`torch.distributions.Normal`.

        Parameters
        ----------
        component_distribution : :class:`torch.distributions.Distribution`
            The distribution object to validate. Must be an instance of `torch.distributions.Normal`.
        loc : :class:`torch.Tensor`
            Mean(s) for the normal distribution.
        scale : :class:`torch.Tensor`
            Standard deviation(s) for the normal distribution.

        Returns
        -------
        :class:`torch.distributions.Distribution`
            The newly set Normal distribution instance.

        Raises
        ------
        :exc:`ValueError`
            If `component_distribution` is not an instance of `torch.distributions.Normal`.
        """
        if not isinstance(
            cls.dist_mixture.component_distribution, torch.distributions.Normal
        ):
            raise ValueError(
                "The Component distribution need to be an "
                "instance of torch.distributions.Normal"
                "Setting the distribution to Normal"
            )
        cls.dist_mixture.component_distribution = torch.distributions.Normal(loc, scale)
        return cls.dist_mixture.component_distribution

    @classmethod
    def extract_probs(cls, *params: Any) -> torch.Tensor:
        """
        Extract the mixture probabilities from the distribution parameters.

        Parameters
        ----------
        *params : :obj:`Any`
            Distribution parameters (expected: loc, scale, probs), passed to `dist_factory`.

        Returns
        -------
        :class:`torch.Tensor`
            A 1D tensor of probabilities for each component in the mixture.
        """
        return cls.dist_factory(*params).mixture_distribution.probs

    @classmethod
    def extract_n_clusters(cls, *params: Any) -> int:
        """
        Return the number of mixture components (i.e., clusters) in the distribution.

        Parameters
        ----------
        *params : :obj:`Any`
            Distribution parameters (expected: loc, scale, probs), passed to `extract_probs`.

        Returns
        -------
        :obj:`int`
            The number of clusters/components in the mixture distribution.
        """
        return cls.extract_probs(*params).size()[0]

    @classmethod
    def extract_cluster_parameters(
        cls,
        which_cluster: int,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        Extract the parameters (probability, mean, standard deviation) for a specific cluster.

        Parameters
        ----------
        which_cluster : :obj:`int`
            The index of the cluster to extract parameters for.

        Returns
        -------
        :obj:`tuple` of :class:`torch.Tensor`
            A tuple containing:
            - prob: Probability of the selected cluster.
            - loc: Mean of the selected normal component.
            - scale: Standard deviation of the selected normal component.
        """
        prob = torch.Tensor(cls.extract_probs)[which_cluster]
        loc = torch.Tensor(cls.mean)[which_cluster]
        scale = torch.Tensor(cls.stddev)[which_cluster]
        return prob, loc, scale

    @classmethod
    def mode(cls, *params: Any) -> torch.Tensor:
        """
        Return the mode of the mixture distribution.

        For a mixture of normal distributions, this returns the mean of the overall
        mixture distribution, which serves as a proxy for the mode.

        Parameters
        ----------
        *params : :obj:`Any`
            Distribution parameters (expected: loc, scale, probs), passed to `dist_factory`.

        Returns
        -------
        :class:`torch.Tensor`
            The mode of the mixture distribution.
        """
        return cls.dist_factory(*params).mean

    @classmethod
    def mean(cls, *params: Any) ->  torch.Tensor:
        """
        Return the mean of the mixture distribution.

        For a mixture of normal distributions, this returns the mean of the overall
        mixture distribution.

        Parameters
        ----------
        *params : :obj:`Any`
            Distribution parameters (expected: loc, scale, probs), passed to `dist_factory`.

        Returns
        -------
        :class:`torch.Tensor`
            The mean of the mixture distribution.
        """
        return cls.dist_factory(*params).mean

    @classmethod
    def stddev(cls, *params: Any) ->  torch.Tensor:
        """
        Return the standard deviation of the mixture distribution.

        For a mixture of normal distributions, this returns the standard deviation
        of the overall mixture distribution.

        Parameters
        ----------
        *params : :obj:`Any`
            Distribution parameters (expected: loc, scale, probs), passed to `dist_factory`.

        Returns
        -------
        :class:`torch.Tensor`
            The standard deviation of the mixture distribution.
        """
        return cls.dist_factory(*params).stddev

    @classmethod
    def _nll(
            cls, x: WeightedTensor, loc: torch.Tensor, scale: torch.Tensor, probs: torch.Tensor,
    ) -> WeightedTensor:
        """
        Compute the element-wise negative log-likelihood (NLL) for a mixture of normal distributions.

        This method handles two scenarios:
        - If `loc` has more than one dimension, it's assumed to represent multiple sources 
        per individual (e.g., shape [n_inds, n_clusters]).
        - If `loc` is one-dimensional, it's treated as a simple mixture (e.g., for global parameters
        like tau or xi), and broadcasting is used accordingly.

        Parameters
        ----------
        x : :class:`.WeightedTensor`
            A tensor-like object containing `value` (data tensor) and `weight` (observation weights).
        loc : :class:`torch.Tensor`
            The means of the normal components. Shape depends on the context:
            - For individual sources: [n_inds, n_clusters]
            - For shared/global parameters: [n_clusters]
        scale : :class:`torch.Tensor`
            The standard deviations of the normal components. Shape should match `loc`.
        probs : :class:`torch.Tensor`
            The mixture probabilities for each component. Not used directly in this method,
            but passed for consistency with the full distribution signature.

        Returns
        -------
        :class:`.WeightedTensor`
            The element-wise negative log-likelihood for each data point, weighted by `x.weight`.
        """

        z_list = []

        if loc.ndim > 1:  # for sources !will need modification if we mess with the sources_std as well
            n_clusters = loc.shape[1]

            for i in range(n_clusters):
                z_cluster = (x.value - loc[:, i]) / scale  # shape: [n_inds, n_sources]
                z_list.append(z_cluster)

            z = torch.stack(z_list, dim=-1)  # shape: [n_inds, n_sources, n_clusters]

        else:  # for tau and xi
            n_clusters = loc.shape[0]

            for i in range(n_clusters):
                #print(scale)
                z_cluster = ((x.value - loc[i]) / scale[i]).squeeze(1)  # shape: [n_inds]
                z_list.append(z_cluster)

            z = torch.stack(z_list, dim=1)  # shape: [n_inds,n_clusters]

        return WeightedTensor((0.5 * z ** 2
                               + torch.log(scale)
                               + cls.nll_constant_standard),x.weight,)

    @classmethod
    def _nll_jacobian(
            cls, x: WeightedTensor, loc: torch.Tensor, scale: torch.Tensor, probs:torch.Tensor,
    ) -> WeightedTensor:
        """
        Compute the Jacobian (gradient w.r.t. `loc`) of the negative log-likelihood 
        for a mixture of normal distributions.
        
        Parameters
        ----------
        x : :class:`.WeightedTensor`
            A tensor-like object with `.value` as the observed data and `.weight` as sample weights.
        loc : :class:`torch.Tensor`
            The means of the normal components. Shape can be:
            - [n_inds, n_clusters] for individual sources
            - [n_clusters] for shared parameters
        scale : :class:`torch.Tensor`
            The standard deviations of the components.
        probs : :class:`torch.Tensor`
            Mixture probabilities. Not used directly here, but included for signature consistency.

        Returns
        -------
        :class:`WeightedTensor`
            A tensor of Jacobian values for each sample and cluster, with shape:
            - [n_inds, n_sources, n_clusters] or [n_inds, n_clusters], depending on input shape.
            The `.weight` is preserved from the input `x`.
        """

        z_list = []

        if loc.ndim > 1:  # for sources !will need modification if we mess with the sources_std as well
            n_clusters = loc.shape[1]

            for i in range(n_clusters):
                z_cluster = (x.value - loc[:, i]) / scale ** 2 # shape: [n_inds, n_sources]
                z_list.append(z_cluster)

            z = torch.stack(z_list, dim=-1)  # shape: [n_inds, n_sources, n_clusters]

        else:  # for tau and xi
            n_clusters = loc.shape[0]

            for i in range(n_clusters):
                z_cluster = ((x.value - loc[i]) / scale[i] ** 2).squeeze(1)  # shape: [n_inds]
                z_list.append(z_cluster)

            z = torch.stack(z_list, dim=1)  # shape: [n_inds,n_clusters]

        return WeightedTensor(z, x.weight)

    @classmethod
    def _nll_and_jacobian(
            cls,
            x: WeightedTensor,
            loc: torch.Tensor,
            scale: torch.Tensor,
            probs: torch.Tensor,
    ) -> tuple[WeightedTensor, WeightedTensor]:
        """
        Compute both the negative log-likelihood (NLL) and its Jacobian w.r.t. `loc`
        for a mixture of normal distributions.

        Parameters
        ----------
        x : :class:`WeightedTensor`
            A tensor-like object containing observed values (`value`) and observation weights (`weight`).
        loc : :class:`torch.Tensor`
            Means of the normal components. Shape:
            - [n_inds, n_clusters] for individual-specific sources.
            - [n_clusters] for global/shared parameters.
        scale : :class:`torch.Tensor`
            Standard deviations of the normal components.
        probs : :class:`torch.Tensor`
            Mixture probabilities (not used in this method, but kept for signature consistency).

        Returns
        -------
        :obj:`tuple` of :class:`WeightedTensor`
            - nll : :class:`WeightedTensor`
                Element-wise negative log-likelihood values.
            - jacobian : :class:`WeightedTensor`
                The Jacobian (gradient) of the NLL with respect to `loc`. Equal to `z / scale`.
        """
        z_list = []

        if loc.ndim > 1:  # for sources !will need modification if we mess with the sources_std as well
            n_clusters = loc.shape[1]

            for i in range(n_clusters):
                z_cluster = (x.value - loc[:, i]) / scale  # shape: [n_inds, n_sources]
                z_list.append(z_cluster)

            z = torch.stack(z_list, dim=-1)  # shape: [n_inds, n_sources, n_clusters]

        else:  # for tau and xi
            n_clusters = loc.shape[0]

            for i in range(n_clusters):
                z_cluster = ((x.value - loc[i]) / scale[i]).squeeze(1)  # shape: [n_inds]
                z_list.append(z_cluster)

            z = torch.stack(z_list, dim=1)  # shape: [n_inds,n_clusters]

        nll = 0.5 * z ** 2 + torch.log(scale) + cls.nll_constant_standard
        return WeightedTensor(nll, x.weight), WeightedTensor(z / scale, x.weight)


class AbstractWeibullRightCensoredFamily(StatelessDistributionFamily):
    dist_weibull: ClassVar = torch.distributions.weibull.Weibull
    precision: float = 0.0001

    @classmethod
    def validate_parameters(cls, *params: Any) -> tuple[torch.Tensor, ...]:
        """Validate consistency of distribution parameters, returning them with out-of-place modifications if needed.

        Parameters
        ----------
        params : Any
            The parameters to pass to the distribution factory.

        Returns
        -------
        :obj:`tuple` [ :class:`torch.Tensor`, ...] :
            The validated parameters.
        """
        raise NotImplementedError("Validate parameters not implemented")

    @classmethod
    def sample(
        cls,
        nu: torch.Tensor,
        rho: torch.Tensor,
        xi: torch.Tensor,
        tau: torch.Tensor,
        sample_shape: tuple[int, ...] = (),
    ) -> torch.Tensor:
        return cls.dist_weibull(nu * torch.exp(-xi), rho).sample(sample_shape) + tau

    @classmethod
    def mode(cls, *params: torch.Tensor) -> torch.Tensor:
        """Return the mode of the distribution (returning first value if discrete ties).

        Parameters
        ----------
        params : :class:`torch.Tensor`
            The distribution parameters.

        Returns
        -------
        :class:`torch.Tensor` :
            The value of the distribution's mode.
        """
        raise NotImplementedError("Mode not implemented")

    @classmethod
    def mean(
        cls,
        nu: torch.Tensor,
        rho: torch.Tensor,
        xi: torch.Tensor,
        tau: torch.Tensor,
    ) -> torch.Tensor:
        """Return the mean of the distribution, if defined.

        Parameters
        ----------
        nu : :class:`torch.Tensor`

        rho : :class:`torch.Tensor`

        xi : :class:`torch.Tensor`

        tau : :class:`torch.Tensor`

        Returns
        -------
        :class:`torch.Tensor` :
            The value of the distribution's mean.
        """
        return cls.dist_weibull(cls._extract_reparametrized_nu(nu, xi), rho).mean + tau

    @staticmethod
    @abstractmethod
    def _extract_reparametrized_nu(nu: torch.Tensor, xi: torch.Tensor) -> torch.Tensor:
        """Reparametrization of nu using individual parameter xi."""

    @classmethod
    def stddev(
        cls,
        nu: torch.Tensor,
        rho: torch.Tensor,
        xi: torch.Tensor,
        tau: torch.Tensor,
    ) -> torch.Tensor:
        """Return the standard-deviation of the distribution, given distribution parameters.

        Parameters
        ----------
        nu : :class:`torch.Tensor`

        rho : :class:`torch.Tensor`

        xi : :class:`torch.Tensor`

        tau : :class:`torch.Tensor`

        Returns
        -------
        :class:`torch.Tensor` :
            The value of the distribution's standard deviation.
        """
        return cls.dist_weibull(
            cls._extract_reparametrized_nu(nu, rho, xi, tau), rho
        ).stddev

    @classmethod
    def _extract_reparametrized_parameters(
        cls,
        x: WeightedTensor,
        nu: torch.Tensor,
        rho: torch.Tensor,
        xi: torch.Tensor,
        tau: torch.Tensor,
        *params: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Construct reparametrized variables
        event_reparametrized_time = cls._extract_reparametrized_event(x.value, tau)
        nu_reparametrized = cls._extract_reparametrized_nu(nu, rho, xi, tau, *params)
        return event_reparametrized_time, x.weight, nu_reparametrized

    @classmethod
    def compute_log_likelihood_hazard(
        cls,
        x: WeightedTensor,
        nu: torch.Tensor,
        rho: torch.Tensor,
        xi: torch.Tensor,
        tau: torch.Tensor,
        *params: torch.Tensor,
    ) -> torch.Tensor:
        event_reparametrized_time, event_bool, nu_reparametrized = (
            cls._extract_reparametrized_parameters(x, nu, rho, xi, tau, *params)
        )
        # Hazard neg log-likelihood only for patient with event not censored
        hazard = torch.where(
            event_reparametrized_time > 0,
            (rho / nu_reparametrized)
            * ((event_reparametrized_time / nu_reparametrized) ** (rho - 1.0)),
            -constants.INFINITY,
        )
        log_hazard = torch.where(hazard > 0, torch.log(hazard), hazard)
        log_hazard = torch.where(event_bool != 0, log_hazard, 0.0)
        return log_hazard

    @classmethod
    def compute_hazard(
        cls,
        x: WeightedTensor,
        nu: torch.Tensor,
        rho: torch.Tensor,
        xi: torch.Tensor,
        tau: torch.Tensor,
        *params: torch.Tensor,
    ) -> torch.Tensor:
        event_reparametrized_time, _, nu_reparametrized = (
            cls._extract_reparametrized_parameters(x, nu, rho, xi, tau, *params)
        )
        # Hazard neg log-likelihood only for patient with event not censored
        hazard = torch.where(
            event_reparametrized_time > 0,
            (rho / nu_reparametrized)
            * ((event_reparametrized_time / nu_reparametrized) ** (rho - 1.0)),
            0.0,
        )
        return hazard

    @classmethod
    def compute_log_survival(
        cls,
        x: torch.Tensor,
        nu: torch.Tensor,
        rho: torch.Tensor,
        xi: torch.Tensor,
        tau: torch.Tensor,
        *params: torch.Tensor,
    ) -> torch.Tensor:
        event_reparametrized_time, _, nu_reparametrized = (
            cls._extract_reparametrized_parameters(x, nu, rho, xi, tau, *params)
        )
        return -(
            (torch.clamp(event_reparametrized_time, min=0.0) / nu_reparametrized) ** rho
        )

    @classmethod
    def compute_predictions(
        cls,
        x: torch.Tensor,
        nu: torch.Tensor,
        rho: torch.Tensor,
        xi: torch.Tensor,
        tau: torch.Tensor,
        *params: torch.Tensor,
    ) -> torch.Tensor:
        nb_events = nu.shape[0]

        # consider that the first time to predict was the last visit and is a reference point
        # and compute the survival S0
        init_log_survival = cls.compute_log_survival(
            WeightedTensor(x.value.min()), nu, rho, xi, tau, *params
        )
        init_survival = torch.exp(init_log_survival.sum(axis=1).expand(nb_events, -1).T)

        if nb_events == 1:
            # when there is only one event, we are interested in the corrected survival S/S0 (Rizopoulos, 2012, p173)
            return (
                torch.exp(cls.compute_log_survival(x, nu, rho, xi, tau, *params))
                / init_survival
            )
        else:
            # When there are multiple event we are interested in the cumulative incidence corrected: CIF/S0
            # see (Andrinopoulou, 2015)
            # Compute for all the possible points till the max
            time = WeightedTensor(
                torch.arange(
                    float(tau),
                    max(float(tau) + cls.precision, x.value.max()),
                    cls.precision,
                    dtype=float,
                )
                .expand(nb_events, -1)
                .T
            )
            log_survival = cls.compute_log_survival(time, nu, rho, xi, tau, *params)
            hazard = cls.compute_hazard(time, nu, rho, xi, tau, *params)
            total_survival = torch.exp(log_survival.sum(axis=1).expand(nb_events, -1).T)
            incidence = total_survival * hazard

            def get_cum_incidence(t, time_ix, incidence_ix):
                # t<tau then the result is 0 as survival is defined to be 1
                index = (time_ix * (time_ix <= t)).argmax() + 1
                return torch.trapezoid(incidence_ix[:index], time_ix[:index])

            list_to_cat = [
                torch.clone(x.value)
                .apply_(lambda t: get_cum_incidence(t, time.value.T[i], incidence.T[i]))
                .T
                for i in range(nb_events)
            ]
            res = torch.cat(list_to_cat).T
            return res / init_survival

    @staticmethod
    def _extract_reparametrized_event(
        event_time: torch.Tensor, tau: torch.Tensor
    ) -> torch.Tensor:
        return event_time - tau

    @staticmethod
    @abstractmethod
    def _extract_reparametrized_nu(
        nu: torch.Tensor,
        rho: torch.Tensor,
        xi: torch.Tensor,
        tau: torch.Tensor,
        *params: torch.Tensor,
    ) -> torch.Tensor:
        """Reparametrization of nu using individual parameter xi"""

    @classmethod
    def _nll(
        cls,
        x: torch.Tensor,
        nu: torch.Tensor,
        rho: torch.Tensor,
        xi: torch.Tensor,
        tau: torch.Tensor,
        *params: torch.Tensor,
    ) -> WeightedTensor:
        """Compute survival neg log-likelihood."""
        log_survival = cls.compute_log_survival(x, nu, rho, xi, tau, *params)
        log_hazard = cls.compute_log_likelihood_hazard(x, nu, rho, xi, tau, *params)

        return WeightedTensor(-1 * (log_survival + log_hazard))

    @classmethod
    def _nll_and_jacobian(
        cls,
        x: WeightedTensor,
        nu: torch.Tensor,
        rho: torch.Tensor,
        xi: torch.Tensor,
        tau: torch.Tensor,
    ) -> tuple[WeightedTensor, WeightedTensor]:
        return cls._nll(x, nu, rho, xi, tau), cls._nll_jacobian(x, nu, rho, xi, tau)

    @classmethod
    def _nll_jacobian(
        cls,
        x: WeightedTensor,
        nu: torch.Tensor,
        rho: torch.Tensor,
        xi: torch.Tensor,
        tau: torch.Tensor,
    ) -> torch.Tensor:
        pass  # WIP
        # Get inputs
        xi_format = xi[:, 0]
        event_rep_time, event_bool, nu_rep = self._extract_reparametrized_parameters(
            x, nu, rho, xi, tau
        )

        # Survival
        log_survival = cls.compute_log_survival(x, nu, rho, xi, tau)

        # Gradients
        grad_xi = rho * log_survival - event_bool * rho
        grad_tau = (rho / nu_rep * torch.exp(xi_format)) * (
            (event_rep_time / nu_rep) ** (rho - 1.0)
        ) + event_bool * (rho - 1) / event_rep_time

        # Normalise as compute on normalised variables
        to_cat = [
            grad_xi,
            grad_tau,
        ]

        grads = torch.cat(to_cat, dim=-1).squeeze(0)

        return grads


class WeibullRightCensoredFamily(AbstractWeibullRightCensoredFamily):
    parameters: ClassVar = ("nu", "rho", "xi", "tau")

    @staticmethod
    def _extract_reparametrized_nu(
        nu: torch.Tensor,
        rho: torch.Tensor,
        xi: torch.Tensor,
        tau: torch.Tensor,
        *params: torch.Tensor,
    ) -> torch.Tensor:
        return torch.exp(-xi) * nu


class WeibullRightCensoredWithSourcesFamily(AbstractWeibullRightCensoredFamily):
    parameters: ClassVar = ("nu", "rho", "xi", "tau", "survival_shifts")

    @staticmethod
    def _extract_reparametrized_nu(nu, rho, xi, tau, survival_shifts):
        return nu * torch.exp(-(xi + (1 / rho) * (survival_shifts)))


@dataclass(frozen=True)
class SymbolicDistribution:
    """Class providing symbolic methods for distribution families."""

    parameters_names: tuple[str, ...]
    dist_family: Type[StatelessDistributionFamily]

    # to hold automatic methods declared in `__post_init__`
    validate_parameters: Callable[..., tuple[torch.Tensor, ...]] = field(
        init=False, repr=False, compare=False
    )
    """Function of named distribution parameters, to validate these parameters."""

    shape: Callable[..., tuple[int, ...]] = field(init=False, repr=False, compare=False)
    """Function of named shapes of distribution parameters, to get shape of distribution samples."""

    mode: Callable[..., torch.Tensor] = field(init=False, repr=False, compare=False)
    """Function of named distribution parameters, to get mode of distribution."""

    mean: Callable[..., torch.Tensor] = field(init=False, repr=False, compare=False)
    """Function of named distribution parameters, to get mean of distribution."""

    stddev: Callable[..., torch.Tensor] = field(init=False, repr=False, compare=False)
    """Function of named distribution parameters, to get std-deviation of distribution."""

    def __post_init__(self):
        if len(self.parameters_names) != len(self.dist_family.parameters):
            raise ValueError(
                f"You provided {len(self.parameters_names)} names for {self.dist_family} parameters, "
                f"while expecting {len(self.dist_family.parameters)}: {self.dist_family.parameters}"
            )
        for bypass_method in {"validate_parameters", "shape", "mode", "mean", "stddev"}:
            object.__setattr__(self, bypass_method, self.get_func(bypass_method))

    def get_func(self, func: str, *extra_args_names: str, **kws):
        """Get keyword-only function from the stateless distribution family."""
        return NamedInputFunction(
            getattr(self.dist_family, func),
            parameters=extra_args_names + self.parameters_names,
            kws=kws or None,
        )

    def get_func_sample(
        self, sample_shape: tuple[int, ...] = ()
    ) -> NamedInputFunction[torch.Tensor]:
        """Factory of symbolic sampling function.

        Parameters
        ----------
        sample_shape : :obj:`tuple` of :obj:`int`, optional
            The shape of the sample.
            Default=().

        Returns
        -------
        :class:`~leaspy.utils.functional.NamedInputFunction` :
            The sample function.
        """
        return self.get_func("sample", sample_shape=sample_shape)

    def get_func_sample_multivariate(
        self, sample_shape: tuple[int, ...] = ()
    ) -> NamedInputFunction[torch.Tensor]:
        """
        Factory of symbolic sampling function.

        Parameters
        ----------
        sample_shape : tuple of int, optional
            The shape of the sample.
            Default=().

        Returns
        -------
        NamedInputFunction :
            The sample function.
        """
        return self.get_func("sample", sample_shape=torch.Size([sample_shape]))

    def get_func_regularization(
        self, value_name: str
    ) -> NamedInputFunction[WeightedTensor[float]]:
        """Factory of symbolic function: state -> negative log-likelihood of value.

        Parameters
        ----------
        value_name : :obj:`str`

        Returns
        -------
        :class:`~leaspy.utils.functional.NamedInputFunction` :
            The named input function to use to compute negative log likelihood.
        """
        return self.get_func("regularization", value_name)

    def get_func_nll(
        self, value_name: str
    ) -> NamedInputFunction[WeightedTensor[float]]:
        """Factory of symbolic function: state -> negative log-likelihood of value.

        Parameters
        ----------
        value_name : :obj:`str`

        Returns
        -------
        :class:`~leaspy.utils.functional.NamedInputFunction` :
            The named input function to use to compute negative log likelihood.
        """
        return self.get_func("nll", value_name)

    def get_func_nll_jacobian(
        self, value_name: str
    ) -> NamedInputFunction[WeightedTensor[float]]:
        """Factory of symbolic function: state -> jacobian w.r.t. value of negative log-likelihood.

        Parameters
        ----------
        value_name : :obj:`str`

        Returns
        -------
        :class:`~leaspy.utils.functional.NamedInputFunction` :
            The named input function to use to compute negative log likelihood jacobian.
        """
        return self.get_func("nll_jacobian", value_name)

    def get_func_nll_and_jacobian(
        self, value_name: str
    ) -> NamedInputFunction[tuple[WeightedTensor[float], WeightedTensor[float]]]:
        """Factory of symbolic function: state -> (negative log-likelihood, its jacobian w.r.t. value).

        Parameters
        ----------
        value_name : :obj:`str`

        Returns
        -------
        :obj:`tuple` [ :class:`~leaspy.utils.functional.NamedInputFunction`, :class:`~leaspy.utils.functional.NamedInputFunction`] :
            The named input functions to use to compute negative log likelihood and its jacobian.
        """
        return self.get_func("nll_and_jacobian", value_name)

    @classmethod
    def bound_to(
        cls,
        dist_family: Type[StatelessDistributionFamily],
    ) -> Callable[..., SymbolicDistribution]:
        """Return a factory to create `SymbolicDistribution` bound to the provided distribution family.

        Parameters
        ----------
        dist_family : :class:`~leaspy.variables.distributions.StatelessDistributionFamily`
            The distribution family to use to create a SymbolicDistribution.

        Returns
        -------
        factory : Callable[..., :class:`~leaspy.variables.distributions.SymbolicDistribution`]
            The factory.
        """

        def factory(*parameters_names: str) -> SymbolicDistribution:
            """Factory of a `SymbolicDistribution`, bounded to the provided distribution family.

            Parameters
            ----------
            *parameters_names : :obj:`str`
                Names, in order, for distribution parameters.

            Returns
            -------
            :class:`~leaspy.variables.distributions.SymbolicDistribution` :
                The symbolic distribution resulting from the factory.
            """
            return SymbolicDistribution(parameters_names, dist_family)

        # Nicer runtime name and docstring for the generated factory function
        factory.__name__ = f"symbolic_{dist_family.__name__}_factory"
        factory.__qualname__ = ".".join(
            factory.__qualname__.split(".")[:-1] + [factory.__name__]
        )
        factory.__doc__ = factory.__doc__.replace(
            "the provided distribution family", f"`{dist_family.__name__}`"
        ).replace(
            "for distribution parameters",
            f"for distribution parameters: {dist_family.parameters}",
        )

        return factory


Normal = SymbolicDistribution.bound_to(NormalFamily)
# Categorical = SymbolicDistribution.bound_to(CategoricalFamily)
MixtureNormal = SymbolicDistribution.bound_to(MixtureNormalFamily)
MultivariateNormal = SymbolicDistribution.bound_to(MultivariateNormalFamily)
Bernoulli = SymbolicDistribution.bound_to(BernoulliFamily)
WeibullRightCensored = SymbolicDistribution.bound_to(WeibullRightCensoredFamily)
WeibullRightCensoredWithSources = SymbolicDistribution.bound_to(
    WeibullRightCensoredWithSourcesFamily
)

# INLINE UNIT TESTS
if __name__ == "__main__":
    print(Normal)
    print(Normal("mean", "std").validate_parameters(mean=0.0, std=1.0))

    nll = Normal("mean", "std").get_func_nll("val")

    args = dict(
        val=WeightedTensor(
            torch.tensor(
                [
                    [-1.0, 0.0, 0.0, 1.0],
                    [0.5, -0.5, -1.0, 0.0],
                ]
            ),
            weight=torch.tensor(
                [
                    [1, 0, 1, 1],
                    [1, 1, 0, 0],
                ]
            ),
        ),
        mean=torch.zeros((2, 4)),
        std=torch.ones(()),
    )

    r_nll = nll(**args)
    print("nll: ", r_nll)
    r_nll_sum_0 = nll.then(sum_dim, dim=1)(**args)
    r_nll_sum_1 = nll.then(sum_dim, dim=1)(**args)
    r_nll_sum_01 = nll.then(sum_dim, dim=(0, 1))(**args)  # MaskedTensor.wsum
    print("nll_sum_0: ", r_nll_sum_0)
    print("nll_sum_1: ", r_nll_sum_1)
    print("nll_sum_01: ", r_nll_sum_01)
    print("nll_sum_0,1: ", sum_dim(r_nll_sum_0, dim=0))
    print("nll_sum_1,0: ", sum_dim(r_nll_sum_1, dim=0))
