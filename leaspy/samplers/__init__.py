from .base import AbstractSampler, AbstractIndividualSampler, AbstractPopulationSampler
from .gibbs import (
    IndividualGibbsSampler,
    PopulationGibbsSampler,
    PopulationFastGibbsSampler,
    PopulationMetropolisHastingsSampler,
)

INDIVIDUAL_SAMPLERS = {"gibbs": IndividualGibbsSampler}
POPULATION_SAMPLERS = {
    "gibbs": PopulationGibbsSampler,
    "fastgibbs": PopulationFastGibbsSampler,
    "metropolis-hastings": PopulationMetropolisHastingsSampler,
}

from .factory import sampler_factory


__all__ = [
    "AbstractSampler",
    "AbstractIndividualSampler",
    "AbstractPopulationSampler",
    "IndividualGibbsSampler",
    "PopulationGibbsSampler",
    "PopulationFastGibbsSampler",
    "PopulationMetropolisHastingsSampler",
    "INDIVIDUAL_SAMPLERS",
    "POPULATION_SAMPLERS",
    "sampler_factory",
]
