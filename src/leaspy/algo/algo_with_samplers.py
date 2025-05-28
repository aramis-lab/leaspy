import warnings
from typing import Optional

from leaspy.exceptions import LeaspyAlgoInputError
from leaspy.io.data import Dataset
from leaspy.samplers import AbstractSampler, sampler_factory
from leaspy.variables.specs import IndividualLatentVariable, PopulationLatentVariable
from leaspy.variables.state import State

from .settings import AlgorithmSettings

__all__ = ["AlgorithmWithSamplersMixin"]


class AlgorithmWithSamplersMixin:
    """Mixin class to use in algorithms that require `samplers`.

    Note that this mixin should be used with a class inheriting from `AbstractAlgo`, which must have `algo_parameters`
    attribute.

    Parameters
    ----------
    settings : :class:`.AlgorithmSettings`
        The specifications of the algorithm as a :class:`.AlgorithmSettings` instance.

        Please note that you can customize the number of memory-less (burn-in) iterations by setting either:
            * `n_burn_in_iter_frac`, such that duration of burn-in phase is a ratio of algorithm `n_iter` (default of 90%)

    Attributes
    ----------
    samplers : :obj:`dict` [:obj:`str`, :class:`~.algo.samplers.abstract_sampler.AbstractSampler` ]
        Dictionary of samplers per each variable

    current_iteration : :obj:`int`, default 0
        Current iteration of the algorithm.
        The first iteration will be 1 and the last one `n_iter`.

    random_order_variables : :obj:`bool` (default True)
        This attribute controls whether we randomize the order of variables at each iteration.
        Article https://proceedings.neurips.cc/paper/2016/hash/e4da3b7fbbce2345d7772b0674a318d5-Abstract.html
        gives a rationale on why we should activate this flag.
    """

    def __init__(self, settings: AlgorithmSettings):
        super().__init__(settings)
        self.samplers: dict[str, AbstractSampler] = None
        self.random_order_variables: bool = self.algo_parameters.get(
            "random_order_variables", True
        )
        self.current_iteration: int = 0
        # Dynamic number of iterations for burn-in phase
        n_burn_in_iter_frac: Optional[float] = self.algo_parameters[
            "n_burn_in_iter_frac"
        ]
        if self.algo_parameters.get("n_burn_in_iter", None) is None:
            if n_burn_in_iter_frac is None:
                raise LeaspyAlgoInputError(
                    "You should NOT have both `n_burn_in_iter_frac` and `n_burn_in_iter` None."
                    "\nPlease set a value for at least one of those settings."
                )
            self.algo_parameters["n_burn_in_iter"] = int(
                n_burn_in_iter_frac * self.algo_parameters["n_iter"]
            )

        elif n_burn_in_iter_frac is not None:
            warnings.warn(
                "`n_burn_in_iter` setting is deprecated in favour of `n_burn_in_iter_frac` - "
                "which defines the duration of the burn-in phase as a ratio of the total number of iterations."
                "\nPlease use the new setting to suppress this warning or explicitly set `n_burn_in_iter_frac=None`."
                "\nNote that while `n_burn_in_iter` is supported "
                "it will always have priority over `n_burn_in_iter_frac`.",
                FutureWarning,
            )

    def _is_burn_in(self) -> bool:
        """
        Check if current iteration is in burn-in (= memory-less) phase.

        Returns
        -------
        bool
        """
        return self.current_iteration <= self.algo_parameters["n_burn_in_iter"]

    def _get_progress_str(self) -> str:
        # The algorithm must define a progress string (thanks to `self.current_iteration`)
        iter_str = super()._get_progress_str()
        if self._is_burn_in():
            iter_str += " (memory-less phase)"
        else:
            iter_str += " (with memory)"
        return iter_str

    def __str__(self):
        out = super().__str__()
        out += "\n= Samplers ="
        for sampler in self.samplers.values():
            out += f"\n    {str(sampler)}"
        return out

    def _initialize_samplers(self, state: State, dataset: Dataset) -> None:
        """
        Instantiate samplers as a dictionary samplers {variable_name: sampler}

        Parameters
        ----------
        state : :class:`.State`
        dataset : :class:`.Dataset`
        """
        self.samplers = {}
        self._initialize_population_samplers(state)
        self._initialize_individual_samplers(state, dataset.n_individuals)

    def _initialize_individual_samplers(self, state: State, n_individuals: int) -> None:
        sampler = self.algo_parameters.get("sampler_ind", None)
        if sampler is None:
            return

        # TODO: per variable and not just per type of variable?
        sampler_kws = self.algo_parameters.get("sampler_ind_params", {})
        for var_name, var in state.dag.sorted_variables_by_type[
            IndividualLatentVariable
        ].items():
            var: IndividualLatentVariable  # for type-hint only
            # remove all properties that are not currently handled by samplers and set default values
            var_kws = dict(
                var.sampling_kws or {},
                name=var_name,
                shape=var.get_prior_shape(state.dag),
            )

            # To enforce a fixed scale for a given var, one should put it in the random var specs
            # But note that for individual variables the model parameters ***_std should always be OK (> 0)
            var_kws.setdefault("scale", var.prior.stddev.call(state))

            self.samplers[var_name] = sampler_factory(
                sampler,
                IndividualLatentVariable,
                n_patients=n_individuals,
                **var_kws,
                **sampler_kws,
            )

    def _initialize_population_samplers(self, state: State) -> None:
        sampler = self.algo_parameters.get("sampler_pop", None)
        if sampler is None:
            return

        # TODO: per variable and not just per type of variable?
        sampler_kws = self.algo_parameters.get("sampler_pop_params", {})
        for var_name, var in state.dag.sorted_variables_by_type[
            PopulationLatentVariable
        ].items():
            var: PopulationLatentVariable  # for type-hint only
            # remove all properties that are not currently handled by samplers and set default values
            var_kws = dict(
                var.sampling_kws or {},
                name=var_name,
                shape=var.get_prior_shape(state.dag),
            )

            # To enforce a fixed scale for a given var, one should put it in the random var specs
            # For instance: for betas & deltas, it is a good idea to define them this way
            # since they'll probably be = 0 just after initialization!
            var_kws.setdefault("scale", state[var_name].abs())
            # TODO: after functional test passed we could change the previous line with the following one (more consistent)
            # var_kws.setdefault("scale", var.prior.stddev.call(state))

            # TODO: mask logic?

            self.samplers[var_name] = sampler_factory(
                sampler, PopulationLatentVariable, **var_kws, **sampler_kws
            )
