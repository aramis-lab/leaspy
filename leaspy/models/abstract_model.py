from __future__ import annotations

import re
import math
from abc import abstractmethod
import copy
import json

import torch
from torch._tensor_str import PRINT_OPTS as torch_print_opts

from leaspy import __version__
from leaspy.models.base import BaseModel
from leaspy.models.noise_models import (
    BaseNoiseModel,
    NoiseModelFactoryInput,
    noise_model_factory,
    export_noise_model,
)
from leaspy.models.utilities import tensor_to_list
from leaspy.io.realizations import (
    VariableType,
    AbstractRealization,
    PopulationRealization,
    IndividualRealization,
    CollectionRealization,
)
from leaspy.io.data.dataset import Dataset

from leaspy.exceptions import LeaspyIndividualParamsInputError, LeaspyModelInputError
from leaspy.utils.typing import (
    FeatureType,
    KwargsType,
    DictParams,
    DictParamsTorch,
    Union,
    List,
    Dict,
    Tuple,
    Iterable,
    Optional,
)


TWO_PI = torch.tensor(2 * math.pi)


#  TODO? refact so to only contain methods needed for the Leaspy api + add another
#  abstract class (interface) on top of it for MCMC fittable models + one for "manifold models"

class AbstractModel(BaseModel):
    """
    Contains the common attributes & methods of the different models.

    Parameters
    ----------
    name : :obj:`str`
        The name of the model.
    noise_model : :obj:`str` or :class:`.BaseNoiseModel`
        The noise model for observations (keyword-only parameter).
    fit_metrics : :obj:`dict`
        Metrics that should be measured during the fit of the model
        and reported back to the user.
    **kwargs
        Hyperparameters for the model

    Attributes
    ----------
    is_initialized : :obj:`bool`
        Indicates if the model is initialized.
    name : :obj:`str`
        The model's name.
    features : :obj:`list` of :obj:`str`
        Names of the model features.
    parameters : :obj:`dict`
        Contains the model's parameters
    noise_model : :class:`.BaseNoiseModel`
        The noise model associated to the model.
    regularization_distribution_factory : function dist params -> :class:`torch.distributions.Distribution`
        Factory of torch distribution to compute log-likelihoods for :term:`regularization` (gaussian by default)
        (Not used anymore)
    fit_metrics : :obj:`dict`
        Contains the metrics that are measured during the fit of the model and reported to the user.
    """

    def __init__(
        self,
        name: str,
        *,
        noise_model: NoiseModelFactoryInput,
        fit_metrics: Optional[Dict[str, float]] = None,
        **kwargs
    ):
        super().__init__(name, **kwargs)
        self.parameters: Optional[KwargsType] = None
        self._noise_model: Optional[BaseNoiseModel] = None

        # load hyperparameters
        self.noise_model = noise_model
        self.load_hyperparameters(kwargs)

        # TODO: dirty hack for now, cf. AbstractFitAlgo
        self.fit_metrics = fit_metrics

    @property
    def noise_model(self) -> BaseNoiseModel:
        return self._noise_model

    @noise_model.setter
    def noise_model(self, model: NoiseModelFactoryInput):
        noise_model = noise_model_factory(model)
        self.check_noise_model_compatibility(noise_model)
        self._noise_model = noise_model

    def check_noise_model_compatibility(self, model: BaseNoiseModel) -> None:
        """
        Raise a :exc:`.LeaspyModelInputError` is the provided noise model
        isn't compatible with the model instance.
        
        This needs to be implemented in subclasses.

        Parameters
        ----------
        model : :class:`.BaseNoiseModel`
            The noise model with which to check compatibility.
        """
        if not isinstance(model, BaseNoiseModel):
            raise LeaspyModelInputError(
                "Expected a subclass of BaselNoiseModel, but received "
                f"a {model.__class__.__name__} instead."
            )

    @abstractmethod
    def to_dict(self) -> KwargsType:
        """
        Export model as a dictionary ready for export.

        Returns
        -------
        KwargsType :
            The model instance serialized as a dictionary.
        """
        return {
            'leaspy_version': __version__,
            'name': self.name,
            'features': self.features,
            'dimension': self.dimension,
            'fit_metrics': self.fit_metrics,  # TODO improve
            'noise_model': export_noise_model(self.noise_model),
            'parameters': {
                k: tensor_to_list(v)
                for k, v in (self.parameters or {}).items()
            }
        }

    def save(self, path: str, **kwargs) -> None:
        """
        Save ``Leaspy`` object as json model parameter file.

        TODO move logic upstream?

        Parameters
        ----------
        path : :obj:`str`
            Path to store the model's parameters.
        **kwargs
            Keyword arguments for :meth:`.AbstractModel.to_dict` child method
            and ``json.dump`` function (default to indent=2).
        """
        from inspect import signature

        export_kws = {k: kwargs.pop(k) for k in signature(self.to_dict).parameters if k in kwargs}
        model_settings = self.to_dict(**export_kws)

        # Default json.dump kwargs:
        kwargs = {'indent': 2, **kwargs}

        with open(path, 'w') as fp:
            json.dump(model_settings, fp, **kwargs)

    def load_parameters(self, parameters: KwargsType) -> None:
        """
        Instantiate or update the model's parameters.

        Parameters
        ----------
        parameters : :obj:`dict` [ :obj:`str`, Any ]
            Contains the model's parameters.
        """
        self.parameters = copy.deepcopy(parameters)

    @abstractmethod
    def load_hyperparameters(self, hyperparameters: KwargsType) -> None:
        """
        Load model's hyperparameters.

        Parameters
        ----------
        hyperparameters : :obj:`dict` [ :obj:`str`, Any ]
            Contains the model's hyperparameters.

        Raises
        ------
        :exc:`.LeaspyModelInputError`
            If any of the consistency checks fail.
        """

    @classmethod
    def _raise_if_unknown_hyperparameters(cls, known_hps: Iterable[str], given_hps: KwargsType) -> None:
        """
        Raises a :exc:`.LeaspyModelInputError` if any unknown hyperparameter is provided to the model.
        """
        # TODO: replace with better logic from GenericModel in the future
        unexpected_hyperparameters = set(given_hps.keys()).difference(known_hps)
        if len(unexpected_hyperparameters) > 0:
            raise LeaspyModelInputError(
                f"Only {known_hps} are valid hyperparameters for {cls.__qualname__}. "
                f"Unknown hyperparameters provided: {unexpected_hyperparameters}."
            )

    def _audit_individual_parameters(self, ips: DictParams) -> KwargsType:
        """
        Perform various consistency and compatibility (with current model) checks
        on an individual parameters dict and outputs qualified information about it.

        TODO? move to IndividualParameters class?

        Parameters
        ----------
        ips : :obj:`dict` [param: str, Any]
            Contains some un-trusted individual parameters.
            If representing only one individual (in a multivariate model) it could be:
                * {'tau':0.1, 'xi':-0.3, 'sources':[0.1,...]}

            Or for multiple individuals:
                * {'tau':[0.1,0.2,...], 'xi':[-0.3,0.2,...], 'sources':[[0.1,...],[0,...],...]}

            In particular, a sources vector (if present) should always be a array_like, even if it is 1D

        Returns
        -------
        ips_info : :obj:`dict`
            * ``'nb_inds'`` : :obj:`int` >= 0
                Number of individuals present.
            * ``'tensorized_ips'`` : :obj:`dict` [ :obj:`str`, :class:`torch.Tensor` ]
                Tensorized version of individual parameters.
            * ``'tensorized_ips_gen'`` : generator
                Generator providing tensorized individual parameters for
                all individuals present (ordered as is).

        Raises
        ------
        :exc:`.LeaspyIndividualParamsInputError`
            If any of the consistency/compatibility checks fail.
        """

        def is_array_like(v):
            # abc.Collection is useless here because set, np.array(scalar) or torch.tensor(scalar)
            # are abc.Collection but are not array_like in numpy/torch sense or have no len()
            try:
                len(v)  # exclude np.array(scalar) or torch.tensor(scalar)
                return hasattr(v, '__getitem__')  # exclude set
            except Exception:
                return False

        # Model supports and needs sources?
        has_sources = (
            hasattr(self, 'source_dimension')
            and isinstance(self.source_dimension, int)
            and self.source_dimension > 0
        )

        # Check parameters names
        expected_parameters = set(['xi', 'tau'] + int(has_sources)*['sources'])
        given_parameters = set(ips.keys())
        symmetric_diff = expected_parameters.symmetric_difference(given_parameters)
        if len(symmetric_diff) > 0:
            raise LeaspyIndividualParamsInputError(
                    f'Individual parameters dict provided {given_parameters} '
                    f'is not compatible for {self.name} model. '
                    f'The expected individual parameters are {expected_parameters}.')

        # Check number of individuals present (with low constraints on shapes)
        ips_is_array_like = {k: is_array_like(v) for k, v in ips.items()}
        ips_size = {k: len(v) if ips_is_array_like[k] else 1 for k, v in ips.items()}

        if has_sources:
            if not ips_is_array_like['sources']:
                raise LeaspyIndividualParamsInputError(
                    f"Sources must be an array_like but {ips['sources']} was provided."
                )

            tau_xi_scalars = all(ips_size[k] == 1 for k in ["tau", "xi"])
            if tau_xi_scalars and (ips_size['sources'] > 1):
                # is 'sources' not a nested array? (allowed iff tau & xi are scalars)
                if not is_array_like(ips['sources'][0]):
                    # then update sources size (1D vector representing only 1 individual)
                    ips_size['sources'] = 1

            # TODO? check source dimension compatibility?

        uniq_sizes = set(ips_size.values())
        if len(uniq_sizes) != 1:
            raise LeaspyIndividualParamsInputError(
                f"Individual parameters sizes are not compatible together. Sizes are {ips_size}."
            )

        # number of individuals present
        n_inds = uniq_sizes.pop()

        # properly choose unsqueezing dimension when tensorizing array_like (useful for sources)
        unsqueeze_dim = -1  # [1,2] => [[1],[2]] (expected for 2 individuals / 1D sources)
        if n_inds == 1:
            unsqueeze_dim = 0  # [1,2] => [[1,2]] (expected for 1 individual / 2D sources)

        # tensorized (2D) version of ips
        t_ips = {k: self._tensorize_2D(v, unsqueeze_dim=unsqueeze_dim) for k, v in ips.items()}

        # construct logs
        return {
            'nb_inds': n_inds,
            'tensorized_ips': t_ips,
            'tensorized_ips_gen': (
                {k: v[i, :].unsqueeze(0) for k, v in t_ips.items()} for i in range(n_inds)
            ),
        }

    @staticmethod
    def _tensorize_2D(x, unsqueeze_dim: int, dtype=torch.float32) -> torch.Tensor:
        """
        Helper to convert a scalar or array_like into an, at least 2D, dtype tensor.

        Parameters
        ----------
        x : scalar or array_like
            Element to be tensorized.
        unsqueeze_dim : :obj:`int`
            Dimension to be unsqueezed (0 or -1).
            Meaningful for 1D array-like only (for scalar or vector
            of length 1 it has no matter).

        Returns
        -------
        :class:`torch.Tensor`, at least 2D

        Examples
        --------
        >>> _tensorize_2D([1, 2], 0) == tensor([[1, 2]])
        >>> _tensorize_2D([1, 2], -1) == tensor([[1], [2])
        """
        # convert to torch.Tensor if not the case
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=dtype)

        # convert dtype if needed
        if x.dtype != dtype:
            x = x.to(dtype)

        # if tensor is less than 2-dimensional add dimensions
        while x.dim() < 2:
            x = x.unsqueeze(dim=unsqueeze_dim)

        # postcondition: x.dim() >= 2
        return x

    def _get_tensorized_inputs(
        self,
        timepoints: torch.Tensor,
        individual_parameters: DictParamsTorch,
        *,
        skip_ips_checks: bool = False,
    ) -> Tuple[torch.Tensor, DictParamsTorch]:
        if not skip_ips_checks:
            # Perform checks on ips and gets tensorized version if needed
            ips_info = self._audit_individual_parameters(individual_parameters)
            n_inds = ips_info['nb_inds']
            individual_parameters = ips_info['tensorized_ips']

            if n_inds != 1:
                raise LeaspyModelInputError(
                    f"Only one individual computation may be performed at a time. {n_inds} was provided."
                )

        # Convert the timepoints (list of numbers, or single number) to a 2D torch tensor
        timepoints = self._tensorize_2D(timepoints, unsqueeze_dim=0)  # 1 individual
        return timepoints, individual_parameters

    # TODO: unit tests? (functional tests covered by api.estimate)
    def compute_individual_trajectory(
        self,
        timepoints,
        individual_parameters: DictParams,
        *,
        skip_ips_checks: bool = False,
    ) -> torch.Tensor:
        """
        Compute scores values at the given time-point(s) given a subject's individual parameters.

        Parameters
        ----------
        timepoints : scalar or array_like[scalar] (:obj:`list`, :obj:`tuple`, :class:`numpy.ndarray`)
            Contains the age(s) of the subject.
        individual_parameters : :obj:`dict`
            Contains the individual parameters.
            Each individual parameter should be a scalar or array_like.
        skip_ips_checks : :obj:`bool` (default: ``False``)
            Flag to skip consistency/compatibility checks and tensorization
            of ``individual_parameters`` when it was done earlier (speed-up).

        Returns
        -------
        :class:`torch.Tensor`
            Contains the subject's scores computed at the given age(s)
            Shape of tensor is ``(1, n_tpts, n_features)``.

        Raises
        ------
        :exc:`.LeaspyModelInputError`
            If computation is tried on more than 1 individual.
        :exc:`.LeaspyIndividualParamsInputError`
            if invalid individual parameters.
        """
        timepoints, individual_parameters = self._get_tensorized_inputs(
            timepoints, individual_parameters, skip_ips_checks=skip_ips_checks
        )
        return self.compute_individual_tensorized(timepoints, individual_parameters)

    # TODO: unit tests? (functional tests covered by api.estimate)
    def compute_individual_ages_from_biomarker_values(
        self,
        value: Union[float, List[float]],
        individual_parameters: DictParams,
        feature: Optional[FeatureType] = None,
    ) -> torch.Tensor:
        """
        For one individual, compute age(s) at which the given features values
        are reached (given the subject's individual parameters).

        Consistency checks are done in the main :term:`API` layer.

        Parameters
        ----------
        value : scalar or array_like[scalar] (:obj:`list`, :obj:`tuple`, :class:`numpy.ndarray`)
            Contains the :term:`biomarker` value(s) of the subject.

        individual_parameters : :obj:`dict`
            Contains the individual parameters.
            Each individual parameter should be a scalar or array_like.

        feature : :obj:`str` (or None)
            Name of the considered :term:`biomarker`.

            .. note::
                Optional for :class:`.UnivariateModel`, compulsory
                for :class:`.MultivariateModel`.

        Returns
        -------
        :class:`torch.Tensor`
            Contains the subject's ages computed at the given values(s).
            Shape of tensor is ``(1, n_values)``.

        Raises
        ------
        :exc:`.LeaspyModelInputError`
            If computation is tried on more than 1 individual.
        """
        value, individual_parameters = self._get_tensorized_inputs(
            value, individual_parameters, skip_ips_checks=False
        )
        return self.compute_individual_ages_from_biomarker_values_tensorized(
            value, individual_parameters, feature
        )

    @abstractmethod
    def compute_individual_ages_from_biomarker_values_tensorized(
        self,
        value: torch.Tensor,
        individual_parameters: DictParamsTorch,
        feature: Optional[FeatureType],
    ) -> torch.Tensor:
        """
        For one individual, compute age(s) at which the given features values are
        reached (given the subject's individual parameters), with tensorized inputs.

        Parameters
        ----------
        value : :class:`torch.Tensor` of shape ``(1, n_values)``
            Contains the :term:`biomarker` value(s) of the subject.

        individual_parameters : DictParamsTorch
            Contains the individual parameters.
            Each individual parameter should be a :class:`torch.Tensor`.

        feature : :obj:`str` (or None)
            Name of the considered :term:`biomarker`.

            .. note::
                Optional for :class:`.UnivariateModel`, compulsory
                for :class:`.MultivariateModel`.

        Returns
        -------
        :class:`torch.Tensor`
            Contains the subject's ages computed at the given values(s).
            Shape of tensor is ``(n_values, 1)``.
        """

    @abstractmethod
    def compute_individual_tensorized(
        self,
        timepoints: torch.Tensor,
        individual_parameters: DictParamsTorch,
        *,
        attribute_type: Optional[str] = None,
    ) -> torch.Tensor:
        """
        Compute the individual values at timepoints according to the model.

        Parameters
        ----------
        timepoints : :class:`torch.Tensor`
            Timepoints tensor of shape ``(n_individuals, n_timepoints)``.

        individual_parameters : :obj:`dict` [ param_name: :obj:`str`, :class:`torch.Tensor` ]
            The tensors are of shape ``(n_individuals, n_dims_param)``.

        attribute_type : :obj:`str` or None
            Flag to ask for :term:`MCMC` attributes instead of model's attributes.

        Returns
        -------
        :class:`torch.Tensor` of shape ``(n_individuals, n_timepoints, n_features)``
        """

    @abstractmethod
    def compute_jacobian_tensorized(
        self,
        timepoints: torch.Tensor,
        individual_parameters: DictParamsTorch,
        *,
        attribute_type: Optional[str] = None,
    ) -> DictParamsTorch:
        """
        Compute the jacobian of the model w.r.t. each individual parameter.

        This function aims to be used in :class:`.ScipyMinimize` to speed up optimization.

        .. note::
            As most of numerical operations are repeated when computing model & jacobian,
            we should create a single method that is able to compute model & jacobian
            "together" (= efficiently) when requested with a flag for instance.

        Parameters
        ----------
        timepoints : :class:`torch.Tensor` of shape ``(n_individuals, n_timepoints)``

        individual_parameters : :obj:`dict` [ param_name: :obj:`str`, :class:`torch.Tensor` ]
            Tensors are of shape ``(n_individuals, n_dims_param)``.

        attribute_type : :obj:`str` or None
            Flag to ask for :term:`MCMC` attributes instead of model's attributes.

        Returns
        -------
        :obj:`dict` [ param_name: :obj:`str`, :class:`torch.Tensor` ] :
            Tensors are of shape ``(n_individuals, n_timepoints, n_features, n_dims_param)``.
        """

    def compute_individual_attachment_tensorized(
        self,
        data: Dataset,
        param_ind: DictParamsTorch,
        *,
        attribute_type: Optional[str] = None,
    ) -> torch.Tensor:
        """
        Compute :term:`attachment` term (per subject).

        Parameters
        ----------
        data : :class:`.Dataset`
            Contains the data of the subjects, in particular the subjects'
            time-points and the mask for nan values & padded visits.

        param_ind : DictParamsTorch
            Contain the individual parameters.

        attribute_type : :obj:`str` or None
            Flag to ask for :term:`MCMC` attributes instead of model's attributes.

        Returns
        -------
        attachment : :class:`torch.Tensor`
            Negative Log-likelihood, shape = ``(n_subjects,)``.
        """
        predictions = self.compute_individual_tensorized(
            data.timepoints, param_ind, attribute_type=attribute_type,
        )
        nll = self.noise_model.compute_nll(data, predictions)
        return nll.sum(dim=tuple(range(1, nll.ndim)))

    def compute_canonical_loss_tensorized(
        self,
        data: Dataset,
        param_ind: DictParamsTorch,
        *,
        attribute_type=None,
    ) -> torch.Tensor:
        """
        Compute canonical loss, which depends on the noise model.

        Parameters
        ----------
        data : :class:`.Dataset`
            Contains the data of the subjects, in particular the subjects'
            time-points and the mask for nan values & padded visits.

        param_ind : :obj:`dict`
            Contain the individual parameters.

        attribute_type : :obj:`str` or None (default)
            Flag to ask for :term:`MCMC` attributes instead of model's attributes.

        Returns
        -------
        loss : :class:`torch.Tensor`
            shape = * (depending on noise-model, always summed over individuals & visits)
        """
        predictions = self.compute_individual_tensorized(
            data.timepoints, param_ind, attribute_type=attribute_type,
        )
        return self.noise_model.compute_canonical_loss(data, predictions)

    def compute_sufficient_statistics(
        self,
        data: Dataset,
        realizations: CollectionRealization,
    ) -> DictParamsTorch:
        """
        Compute sufficient statistics from realizations.

        Parameters
        ----------
        data : :class:`.Dataset`
        realizations : :class:`.CollectionRealization`

        Returns
        -------
        :obj:`dict` [ suff_stat: :obj:`str`, :class:`torch.Tensor`]
        """
        suff_stats = self.compute_model_sufficient_statistics(data, realizations)
        predictions = self.compute_individual_tensorized(
            data.timepoints, realizations.individual.tensors_dict, attribute_type='MCMC'
        )
        noise_suff_stats = self.noise_model.compute_sufficient_statistics(
            data, predictions
        )

        # we add some fake sufficient statistics that are in fact convergence metrics (summed over individuals)
        # TODO proper handling of metrics
        d_regul, _ = self.compute_regularity_individual_parameters(
            realizations.individual.tensors_dict, include_constant=True
        )
        cvg_metrics = {f'nll_regul_{param}': r.sum() for param, r in d_regul.items()}
        cvg_metrics['nll_regul_tot'] = sum(cvg_metrics.values(), 0.)
        cvg_metrics['nll_attach'] = self.noise_model.compute_nll(data, predictions).sum()
        cvg_metrics['nll_tot'] = cvg_metrics['nll_attach'] + cvg_metrics['nll_regul_tot']

        # using `dict` enforces no conflicting name in sufficient statistics
        return dict(suff_stats, **noise_suff_stats, **cvg_metrics)

    @abstractmethod
    def compute_model_sufficient_statistics(
        self,
        data: Dataset,
        realizations: CollectionRealization,
    ) -> DictParamsTorch:
        """
        Compute sufficient statistics from a :class:`.CollectionRealization`.

        Parameters
        ----------
        data : :class:`.Dataset`
        realizations : :class:`.CollectionRealization`

        Returns
        -------
        :obj:`dict` [ suff_stat: :obj:`str`, :class:`torch.Tensor` ]
        """

    def update_parameters_burn_in(
        self,
        data: Dataset,
        sufficient_statistics: DictParamsTorch,
    ) -> None:
        """
        Update model parameters (burn-in phase).

        Parameters
        ----------
        data : :class:`.Dataset`
        sufficient_statistics : :obj:`dict` [ suff_stat: :obj:`str`, :class:`torch.Tensor` ]
        """
        self.update_model_parameters_burn_in(data, sufficient_statistics)
        self.noise_model.update_parameters_from_sufficient_statistics(data, sufficient_statistics)

    @abstractmethod
    def update_model_parameters_burn_in(
        self,
        data: Dataset,
        sufficient_statistics: DictParamsTorch,
    ) -> None:
        """
        Update model parameters (burn-in phase).

        Parameters
        ----------
        data : :class:`.Dataset`
        sufficient_statistics : :obj:`dict` [ suff_stat: :obj:`str`, :class:`torch.Tensor` ]
        """

    def update_parameters_normal(
        self,
        data: Dataset,
        sufficient_statistics: DictParamsTorch,
    ) -> None:
        """
        Update model parameters (after burn-in phase).

        Parameters
        ----------
        data : :class:`.Dataset`
        sufficient_statistics : :obj:`dict` [ suff_stat: :obj:`str`, :class:`torch.Tensor` ]
        """
        self.update_model_parameters_normal(data, sufficient_statistics)
        self.noise_model.update_parameters_from_sufficient_statistics(data, sufficient_statistics)

    @abstractmethod
    def update_model_parameters_normal(
        self,
        data: Dataset,
        sufficient_statistics: DictParamsTorch,
    ) -> None:
        """
        Update model parameters (after burn-in phase).

        Parameters
        ----------
        data : :class:`.Dataset`
        sufficient_statistics : :obj:`dict` [ suff_stat: :obj:`str`, :class:`torch.Tensor` ]
        """

    def get_population_variable_names(self) -> List[str]:
        """
        Get the names of the population variables of the model.

        Returns
        -------
        :obj:`list` of :obj:`str`
        """
        return list(self.get_population_random_variable_information().keys())

    def get_individual_variable_names(self) -> List[str]:
        """
        Get the names of the individual variables of the model.

        Returns
        -------
        :obj:`list` of :obj:`str`
        """
        return list(self.get_individual_random_variable_information().keys())

    @classmethod
    def _serialize_tensor(cls, v, *, indent: str = "", sub_indent: str = "") -> str:
        """Nice serialization of floats, torch tensors (or numpy arrays)."""
        if isinstance(v, (str, bool, int)):
            return str(v)
        if isinstance(v, float) or getattr(v, 'ndim', -1) == 0:
            # for 0D tensors / arrays the default behavior is to print all digits...
            # change this!
            return f'{v:.{1+torch_print_opts.precision}g}'
        if isinstance(v, (list, frozenset, set, tuple)):
            try:
                return cls._serialize_tensor(torch.tensor(list(v)), indent=indent, sub_indent=sub_indent)
            except Exception:
                return str(v)
        if isinstance(v, dict):
            if not len(v):
                return ""
            subs = [
                f"{p} : " + cls._serialize_tensor(vp, indent="  ", sub_indent=" "*len(f"{p} : ["))
                for p, vp in v.items()
            ]
            lines = [indent + _ for _ in "\n".join(subs).split("\n")]
            return "\n" + "\n".join(lines)
        # torch.tensor, np.array, ...
        # in particular you may use `torch.set_printoptions` and `np.set_printoptions` globally
        # to tune the number of decimals when printing tensors / arrays
        v_repr = str(v)
        # remove tensor prefix & possible device/size/dtype suffixes
        v_repr = re.sub(r'^[^\(]+\(', '', v_repr)
        v_repr = re.sub(r'(?:, device=.+)?(?:, size=.+)?(?:, dtype=.+)?\)$', '', v_repr)
        # adjust justification
        return re.sub(r'\n[ ]+([^ ])', rf'\n{sub_indent}\1', v_repr)

    def __str__(self):
        output = "=== MODEL ==="
        output += self._serialize_tensor(self.parameters)

        nm_props = export_noise_model(self.noise_model)
        nm_name = nm_props.pop('name')
        output += f"\nnoise-model : {nm_name}"
        output += self._serialize_tensor(nm_props, indent="  ")

        return output

    def compute_regularity_realization(self, realization: AbstractRealization) -> torch.Tensor:
        """
        Compute regularity term for a :class:`.AbstractRealization` instance.

        Parameters
        ----------
        realization : :class:`.AbstractRealization`

        Returns
        -------
        :class:`torch.Tensor` of the same shape as :attr:`.AbstractRealization.tensor`
        """
        # we do not need to include regularity constant
        # (priors are always fixed at a given iteration)
        if isinstance(realization, PopulationRealization):
            return self.compute_regularity_population_realization(realization)
        if isinstance(realization, IndividualRealization):
            return self.compute_regularity_individual_realization(realization)
        raise LeaspyModelInputError(
            f"Realization {realization} not known, should be 'population' or 'individual'."
        )

    def compute_regularity_population_realization(self, realization: PopulationRealization) -> torch.Tensor:
        """
        Compute regularity term for :class:`.PopulationRealization`.

        Parameters
        ----------
        realization : :class:`.PopulationRealization`

        Returns
        -------
        :class:`torch.Tensor` of the same shape as :attr:`.PopulationRealization.tensor`
        """
        return self.compute_regularity_variable(
            realization.tensor,
            self.parameters[realization.name],
            self.MCMC_toolbox['priors'][f"{realization.name}_std"],
            include_constant=False,
        )

    def compute_regularity_individual_realization(self, realization: IndividualRealization) -> torch.Tensor:
        """
        Compute regularity term for :class:`.IndividualRealization`.

        Parameters
        ----------
        realization : :class:`.IndividualRealization`

        Returns
        -------
        :class:`torch.Tensor` of the same shape as :attr:`.IndividualRealization.tensor`
        """
        return self.compute_regularity_variable(
            realization.tensor,
            self.parameters[f"{realization.name}_mean"],
            self.parameters[f"{realization.name}_std"],
            include_constant=False,
        )

    def compute_regularity_individual_parameters(
        self,
        individual_parameters: DictParamsTorch,
        *,
        include_constant: bool = False,
    ) -> Tuple[DictParamsTorch, DictParamsTorch]:
        """
        Compute the regularity terms (and their gradients if requested), per individual variable of the model.

        Parameters
        ----------
        individual_parameters : :obj:`dict` [ :obj:`str`, :class:`torch.Tensor` ]
            Individual parameters as a dict of tensors of shape ``(n_ind, n_dims_param)``.
        include_constant : :obj:`bool`, optional
            Whether to include a constant term or not.
            Default=False.

        Returns
        -------
        regularity : :obj:`dict` [ param_name: :obj:`str`, :class:`torch.Tensor` ]
            Regularity of the patient(s) corresponding to the given individual parameters.
            Tensors have shape ``(n_individuals)``.

        regularity_grads : :obj:`dict` [ param_name: :obj:`str`, :class:`torch.Tensor` ]
            Gradient of regularity term with respect to individual parameters.
            Tensors have shape ``(n_individuals, n_dims_param)``.
        """
        regularity = {}
        regularity_grads = {}

        for param_name, param_val in individual_parameters.items():
            # priors on this parameter
            priors = dict(
                mean=self.parameters[param_name+"_mean"],
                std=self.parameters[param_name+"_std"]
            )

            # TODO? create a more generic method in model `compute_regularity_variable`?
            # (at least pass the parameter name to this method to compute regularity term for non-Normal priors)
            regularity_param, regularity_grads[param_name] = self.compute_regularity_variable(
                param_val, **priors, include_constant=include_constant, with_gradient=True
            )
            # we sum on the dimension of the parameter (always 1D for now), but regularity term is per individual
            # TODO: shouldn't this summation be done directly in `compute_regularity_variable`
            #  (e.g. multivariate normal)
            regularity[param_name] = regularity_param.sum(dim=1)

        return regularity, regularity_grads

    def compute_regularity_variable(
        self,
        value: torch.Tensor,
        mean: torch.Tensor,
        std: torch.Tensor,
        *,
        include_constant: bool = True,
        with_gradient: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Compute regularity term (Gaussian distribution) and optionally its gradient wrt value.

        .. note::
            TODO: should be encapsulated in a ``RandomVariableSpecification`` class
            together with other specs of RV.

        Parameters
        ----------
        value, mean, std : :class:`torch.Tensor` of same shapes
        include_constant : :obj:`bool` (default ``True``)
            Whether we include or not additional terms constant with respect to `value`.
        with_gradient : :obj:`bool` (default ``False``)
            Whether we also return the gradient of :term:`regularity` term with respect to `value`.

        Returns
        -------
        :class:`torch.Tensor` of same shape than input
        """
        # This is really slow when repeated on tiny tensors (~3x slower than direct formula!)
        #return -self.regularization_distribution_factory(mean, std).log_prob(value)

        y = (value - mean) / std
        neg_loglike = 0.5 * y * y
        if include_constant:
            neg_loglike += 0.5 * torch.log(TWO_PI * std**2)
        if not with_gradient:
            return neg_loglike
        nll_grad = y / std
        return neg_loglike, nll_grad

    @abstractmethod
    def get_population_random_variable_information(self) -> DictParams:
        """
        Return the information on population random variables relative to the model.

        Returns
        -------
        DictParams :
            The information on the population random variables.
        """

    @abstractmethod
    def get_individual_random_variable_information(self) -> DictParams:
        """
        Return the information on individual random variables relative to the model.

        Returns
        -------
        DictParams :
            The information on the individual random variables.
        """

    def get_deterministic_variable_information(self) -> DictParams:
        """
        Return the information on individual random variables relative to the model.

        Returns
        -------
        DictParams :
            The information on the individual random variables.
        """
        return {}

    def _get_population_realizations(self, **init_kws) -> CollectionRealization:
        population_collection = CollectionRealization()
        population_collection.initialize_population(self, **init_kws)
        return population_collection

    @staticmethod
    def time_reparametrization(
        timepoints: torch.Tensor,
        xi: torch.Tensor,
        tau: torch.Tensor,
    ) -> torch.Tensor:
        """
        Tensorized time reparametrization formula.

        .. warning::
            Shapes of tensors must be compatible between them.

        Parameters
        ----------
        timepoints : :class:`torch.Tensor`
            Timepoints to reparametrize.
        xi : :class:`torch.Tensor`
            Log-acceleration of individual(s).
        tau : :class:`torch.Tensor`
            Time-shift(s).

        Returns
        -------
        :class:`torch.Tensor` of same shape as `timepoints`
        """
        return torch.exp(xi) * (timepoints - tau)

    def move_to_device(self, device: torch.device) -> None:
        """
        Move a model and its relevant attributes to the specified :class:`torch.device`.

        Parameters
        ----------
        device : :class:`torch.device`
        """

        # Note that in a model, the only tensors that need offloading to a
        # particular device are in the model.parameters dict as well as in the
        # attributes and MCMC_toolbox['attributes'] objects

        for parameter in self.parameters:
            self.parameters[parameter] = self.parameters[parameter].to(device)

        if hasattr(self, "attributes"):
            self.attributes.move_to_device(device)

        if hasattr(self, "MCMC_toolbox"):
            MCMC_toolbox_attributes = self.MCMC_toolbox.get("attributes", None)
            if MCMC_toolbox_attributes is not None:
                MCMC_toolbox_attributes.move_to_device(device)

        self.noise_model.move_to_device(device)
