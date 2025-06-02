"""This module defines the `AbstractFitAlgo` class used for fitting algorithms."""

from typing import Optional

from leaspy.models import ModelType
from leaspy.utils.typing import DictParamsTorch

from ..base import AlgorithmType, IterativeAlgorithm, ReturnType
from ..settings import AlgorithmSettings, OutputsSettings

__all__ = ["FitAlgorithm"]


class FitAlgorithm(IterativeAlgorithm[ModelType, ReturnType]):
    r"""Abstract class containing common method for all `fit` algorithm classes.

    The algorithm is proven to converge if the sequence `burn_in_step` is positive, with an
    infinite sum :math:`\sum_k \epsilon_k = +\infty` and a finite sum of the squares
    :math:`\sum_k \epsilon_k^2 < \infty` (see following paper).

    `Construction of Bayesian Deformable Models via a Stochastic Approximation Algorithm: A Convergence Study <https://arxiv.org/abs/0706.0787>`_

    Parameters
    ----------
    settings : :class:`~leaspy.algo.AlgorithmSettings`
        The specifications of the algorithm as a :class:`~leaspy.algo.AlgorithmSettings` instance.

    Attributes
    ----------
    algorithm_device : :obj:`str`
        Valid :class:`torch.device`
    current_iteration : :obj:`int`, default 0
        The number of the current iteration.
        The first iteration will be 1 and the last one `n_iter`.
    sufficient_statistics : :obj:`dict` [:obj:`str`, :class:`torch.Tensor`] or None
        Sufficient statistics of the previous step.
        It is None during all the burn-in phase.
    output_manager : :class:`~leaspy.io.logs.fit_output_manager.FitOutputManager`
        Optional output manager of the algorithm
    Inherited attributes
        From :class:`~leaspy.algo.AbstractAlgo`

    See Also
    --------
    :meth:`leaspy.api.Leaspy.fit`
    """

    family = AlgorithmType.FIT

    def __init__(self, settings: AlgorithmSettings):
        super().__init__(settings)
        self.logs = settings.logs
        self.sufficient_statistics: Optional[DictParamsTorch] = None

    def set_output_manager(self, output_settings: OutputsSettings) -> None:
        """Set a :class:`~leaspy.algo.fit.FitOutputManager` object for the run of the algorithm.

        Parameters
        ----------
        output_settings : :class:`~leaspy.algo.OutputsSettings`
            Contains the logs settings for the computation run (console print periodicity, plot periodicity ...)

        Examples
        --------
        >>> from leaspy.algo import AlgorithmSettings, algorithm_factory, OutputsSettings
        >>> algo_settings = AlgorithmSettings("mcmc_saem")
        >>> my_algo = algorithm_factory(algo_settings)
        >>> settings = {
            'path': 'brouillons',
            'print_periodicity': 50,
            'plot_periodicity': 100,
            'save_periodicity': 50
        }
        >>> my_algo.set_output_manager(OutputsSettings(settings))
        """
        if output_settings is not None:
            from .fit_output_manager import FitOutputManager

            self.output_manager = FitOutputManager(output_settings)

    def _get_fit_metrics(self) -> Optional[dict[str, float]]:
        # TODO: finalize metrics handling, a bit dirty to place them in sufficient stats, only with a prefix...
        if self.sufficient_statistics is None:
            return None
        return {
            # (scalars only)
            k: v.item()
            for k, v in self.sufficient_statistics.items()
            if k.startswith("nll_")
        }

    def __str__(self) -> str:
        out = super().__str__()
        # add the fit metrics after iteration number (included the sufficient statistics for now...)
        fit_metrics = self._get_fit_metrics() or {}
        if len(fit_metrics):
            out += "\n= Metrics ="
            for m, v in fit_metrics.items():
                out += f"\n    {m} : {v:.5g}"
        return out
