import warnings

import numpy as np

from leaspy.exceptions import LeaspyAlgoInputError

__all__ = ["AlgoWithAnnealingMixin"]


class AlgoWithAnnealingMixin:
    """
    Mixin to use in algorithms needing `temperature_inv`; inherit from this class first.

    Note that this mixin is to be used with a class inheriting from `AbstractAlgo`
    (and in particular that have a `algo_parameters` attribute)

    Parameters
    ----------
    settings : :class:`.AlgorithmSettings`
        The specifications of the algorithm as a :class:`.AlgorithmSettings` instance.

        Please note that you can customize the number of iterations with annealing by setting either:
        * `annealing.n_iter` directly (deprecated but has priority over following setting, not defined by default)
        * `annealing.n_iter_frac`, such that iterations with annealing is a ratio of algorithm `n_iter` (default of 50%)

    Attributes
    ----------
    annealing_on : bool
        Is annealing activated or not?

    temperature : float >= 1
    temperature_inv : float in ]0, 1]
        Temperature and its inverse when using annealing
    """

    def __init__(self, settings):
        super().__init__(settings)

        self.temperature: float = 1.0
        self.temperature_inv: float = 1.0

        # useful property derived from algo parameters
        self.annealing_on: bool = self.algo_parameters.get("annealing", {}).get(
            "do_annealing", False
        )
        self._annealing_period: int = None
        self._annealing_temperature_decrement: float = None

        if not self.annealing_on:
            return

        # Dynamic number of iterations for annealing
        annealing_n_iter_frac = self.algo_parameters["annealing"]["n_iter_frac"]

        if self.algo_parameters["annealing"].get("n_iter", None) is None:
            if annealing_n_iter_frac is None:
                raise LeaspyAlgoInputError(
                    "You should NOT have both `annealing.n_iter_frac` and `annealing.n_iter` None."
                    "\nPlease set a value for at least one of those settings."
                )

            self.algo_parameters["annealing"]["n_iter"] = int(
                annealing_n_iter_frac * self.algo_parameters["n_iter"]
            )

        elif annealing_n_iter_frac is not None:
            warnings.warn(
                "`annealing.n_iter` setting is deprecated in favour of `annealing.n_iter_frac` - "
                "which defines the duration with annealing as a ratio of the total number of iterations."
                "\nPlease use the new setting to suppress this warning "
                "or explicitly set `annealing.n_iter_frac=None`."
                "\nHowever, note that while `annealing.n_iter` is supported "
                "it will always have priority over `annealing.n_iter_frac`.",
                FutureWarning,
            )

    def __str__(self):
        out = super().__str__()
        if self.annealing_on:
            out += "\n= Annealing =\n"
            out += f"    temperature : {self.temperature:.1f}"
        return out

    def _initialize_annealing(self):
        """
        Initialize annealing, setting initial temperature and number of iterations.
        """
        if not self.annealing_on:
            return

        self.temperature = self.algo_parameters["annealing"]["initial_temperature"]
        self.temperature_inv = 1 / self.temperature

        if not (
            isinstance(self.algo_parameters["annealing"]["n_plateau"], int)
            and self.algo_parameters["annealing"]["n_plateau"] > 0
        ):
            raise LeaspyAlgoInputError(
                "Your `annealing.n_plateau` should be a positive integer"
            )

        if self.algo_parameters["annealing"]["n_plateau"] == 1:
            warnings.warn(
                "You defined `annealing.n_plateau` = 1, so you will stay at initial temperature. "
                "Consider setting `annealing.n_plateau` >= 2 for a true annealing scheme."
            )
            return

        self._annealing_period = self.algo_parameters["annealing"]["n_iter"] // (
            self.algo_parameters["annealing"]["n_plateau"] - 1
        )

        self._annealing_temperature_decrement = (
            self.algo_parameters["annealing"]["initial_temperature"] - 1.0
        ) / (self.algo_parameters["annealing"]["n_plateau"] - 1)
        if self._annealing_temperature_decrement <= 0:
            raise LeaspyAlgoInputError("Your `initial_temperature` should be > 1")

    def _update_temperature(self):
        """
        Update the temperature according to a plateau annealing scheme.
        """
        if not self.annealing_on or self._annealing_period is None:
            return

        if self.current_iteration <= self.algo_parameters["annealing"]["n_iter"]:
            # If we cross a plateau step
            if self.current_iteration % self._annealing_period == 0:
                # Oscillating scheme
                if self.algo_parameters["annealing"].get("oscillations", False):
                    params = self.algo_parameters["annealing"]
                    b = params["range"]
                    c = params["delay"]
                    r = params["period"]
                    k = self.current_iteration
                    kappa = c + 2.0 * float(k) * np.pi / r
                    self.temperature = max(1.0 + b * np.sin(kappa) / kappa, 0.1)

                else:
                    # Decrease temperature linearly
                    self.temperature -= self._annealing_temperature_decrement
                    self.temperature = max(self.temperature, 1)

                self.temperature_inv = 1.0 / self.temperature
