from __future__ import annotations
from typing import Callable, TYPE_CHECKING, Dict, Optional, Union, NoReturn

from .realization import (
    AbstractRealization,
    IndividualRealization,
    PopulationRealization,
    DeterministicRealization,
)
from .dict_realizations import DictRealizations
from leaspy.exceptions import LeaspyInputError
if TYPE_CHECKING:
    from leaspy.models.abstract_model import AbstractModel


DictRealizationsType = Union[DictRealizations, Dict[str, AbstractRealization]]


class CollectionRealization(DictRealizations):
    """
    Realizations of population and individual variables, stratified per variable type.

    Parameters
    ----------
    population : DictRealizationsType, optional
    individual : DictRealizationsType, optional
    """
    def __init__(
        self,
        *,
        population: Optional[DictRealizationsType] = None,
        individual: Optional[DictRealizationsType] = None,
        deterministic: Optional[DictRealizationsType] = None,
    ):
        self._population = DictRealizations()
        self._individual = DictRealizations()
        self._deterministic = DictRealizations()
        if population:
            self.population = population
        if individual:
            self.individual = individual
        if deterministic:
            self.deterministic = deterministic

    @property
    def realizations_dict(self) -> Dict[str, AbstractRealization]:
        """Pooled dictionary of realizations, to (almost) provide the common `DictRealizations` interface."""
        return dict(
            **self.population.realizations_dict,
            **self.individual.realizations_dict,
            **self.deterministic.realizations_dict,
        )

    def __setitem__(self, variable_name: str, v) -> NoReturn:
        raise NotImplementedError("Please use the setter from `.population` or `.individual` sub-collections.")

    @property
    def population(self) -> DictRealizations:
        return self._population

    @population.setter
    def population(self, population: DictRealizationsType):
        if isinstance(population, dict):
            self._population = DictRealizations(population)
        else:
            self._population = population
        self._check_unicity_of_variable_names()

    @property
    def individual(self) -> DictRealizations:
        return self._individual

    @property
    def deterministic(self) -> DictRealizations:
        return self._deterministic

    @individual.setter
    def individual(self, individual: DictRealizationsType):
        if isinstance(individual, dict):
            self._individual = DictRealizations(individual)
        else:
            self._individual = individual
        self._check_unicity_of_variable_names()

    def _check_unicity_of_variable_names(self):
        variable_names_intersection = set(self.population.names).intersection(set(self.individual.names))
        if len(variable_names_intersection) != 0:
            raise LeaspyInputError(
                f"All variable names must be distinct in a CollectionRealization instance. "
                f"The following duplicates were found: {variable_names_intersection}."
            )

    def initialize(
        self,
        model: AbstractModel,
        *,
        n_individuals: int,
        skip_variable: Optional[Callable[[dict], bool]] = None,
        **realization_init_kws,
    ) -> None:
        """
        Initialize the CollectionRealization instance from a Model instance.

        Parameters
        ----------
        model : AbstractModel
            The model from which to initialize the collection of realizations.
        n_individuals : int
            The number of individuals in the data.
        skip_variable : Callable or bool, optional
            Whether some variables should be skipped or not.
        **realization_init_kws : dict
            Kwargs for initializing the Realizations.
        """
        self.initialize_population(model, skip_variable=skip_variable, **realization_init_kws)
        self.initialize_individuals(model, n_individuals=n_individuals, skip_variable=skip_variable, **realization_init_kws)
        self.initialize_deterministic(model, n_individuals=n_individuals, skip_variable=skip_variable, **realization_init_kws)

    def initialize_population(
        self,
        model: AbstractModel,
        *,
        skip_variable: Optional[Callable[[dict], bool]] = None,
        **realization_init_kws,
    ) -> None:
        """
        Initialize the population part of the CollectionRealization instance from a Model instance.

        Parameters
        ----------
        model : AbstractModel
            The model from which to initialize the collection of realizations.
        skip_variable : Callable or bool, optional
            Whether some variables should be skipped or not.
        **realization_init_kws : dict
            Kwargs for initializing the Realizations.
        """
        info = model.get_population_random_variable_information()
        for info_variable in info.values():
            if skip_variable is not None and skip_variable(info_variable):
                continue
            name = info_variable["name"]
            realization = PopulationRealization(
                name, info_variable["shape"],
            )
            realization.initialize(model, **realization_init_kws)
            self.population[name] = realization

    def initialize_individuals(
        self,
        model: AbstractModel,
        *,
        n_individuals: int,
        skip_variable: Optional[Callable[[dict], bool]] = None,
        **realization_init_kws,
    ) -> None:
        """
        Initialize the individual part of the CollectionRealization instance from a Model instance.

        Parameters
        ----------
        model : AbstractModel
            The model from which to initialize the collection of realizations.
        n_individuals : int
            The number of individuals in the data.
        skip_variable : Callable or bool, optional
            Whether some variables should be skipped or not.
        **realization_init_kws : dict
            Kwargs for initializing the Realizations.
        """
        info = model.get_individual_random_variable_information()
        for info_variable in info.values():
            if skip_variable is not None and skip_variable(info_variable):
                continue
            name = info_variable["name"]
            realization = IndividualRealization(
                name, info_variable["shape"], n_individuals=n_individuals,
            )
            realization.initialize(model, **realization_init_kws)
            self.individual[name] = realization

    def initialize_deterministic(
        self,
        model: AbstractModel,
        *,
        n_individuals: int,
        skip_variable: Optional[Callable[[dict], bool]] = None,
        **realization_init_kws,
    ) -> None:
        """
        Initialize the deterministic part of the CollectionRealization instance from a Model instance.

        Parameters
        ----------
        model : AbstractModel
            The model from which to initialize the collection of realizations.
        n_individuals : int
            The number of individuals in the data.
        skip_variable : Callable or bool, optional
            Whether some variables should be skipped or not.
        **realization_init_kws : dict
            Kwargs for initializing the Realizations.
        """
        info = model.get_deterministic_variable_information()
        for info_variable in info.values():
            if skip_variable is not None and skip_variable(info_variable):
                continue
            name = info_variable["name"]
            realization = DeterministicRealization(
                name, info_variable["shape"], n_individuals=n_individuals,
            )
            realization.initialize(model, **realization_init_kws)
            self.deterministic[name] = realization

    def clone(self) -> CollectionRealization:
        """
        Deep-copy of the CollectionRealization instance.

        In particular the underlying realizations are cloned and detached.

        Returns
        -------
        CollectionRealization :
            The cloned collection of realizations.
        """
        return CollectionRealization(
            population=self.population.clone(),
            individual=self.individual.clone(),
            deterministic=self.deterministic.clone(),
        )
