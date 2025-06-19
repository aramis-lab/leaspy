"""This module defines the `` class used for all simulation algorithms."""

from abc import abstractmethod

from leaspy.io.data import Dataset
from leaspy.io.data.data import Data
from leaspy.io.outputs import IndividualParameters
from leaspy.io.outputs.result import Result
from leaspy.models import McmcSaemCompatibleModel, ModelType

from ..base import AlgorithmType, IterativeAlgorithm, ReturnType
from ..settings import OutputsSettings

__all__ = ["SimulateAlgorithm"]


class BaseSimulationAlgorithm(IterativeAlgorithm[ModelType, ReturnType]):
    """
    Wrapper class that delegates to the actual implementation of SimulationAlgorithm.
    """

    def set_output_manager(self, output_settings: OutputsSettings) -> None:
        """Set the output manager.

        This is currently not implemented for simulate.
        """
        pass

    def _run(self, model: McmcSaemCompatibleModel) -> Result:
        """Run the simulation pipeline using a leaspy model.

        This method simulates longitudinal data using the given leaspy model.
        It performs the following steps:
        - Retrieves individual parameters (IP) from fixed effects of the model.
        - Loads the specified Leaspy model.
        - Generates visit ages (timepoints) for each individual (based on specifications
        in visits_type from AlgorithmSettings)
        - Simulates observations at those visit ages.
        - Packages the result into a `Result` object, including simulated data,
        individual parameters, and the model's noise standard deviation.

        Parameters
        ----------
        model : :class:~.models.abstract_model.McmcSaemCompatibleModel
            A Leaspy model object previously trained on longitudinal data.

        Returns
        -------
        result_obj : :class:`Result`
            An object containing:
            - `data`: Simulated longitudinal dataset (`Data` object),
            - `individual_parameters`: The individual parameters used in simulation,
            - `noise_std`: Noise standard deviation used in the simulation.
        """

        # Simulate RE for RM
        individual_parameters_from_model_parameters = (
            self._sample_individual_parameters_from_model_parameters(model)
        )

        self._get_leaspy_model(model)

        dict_timepoints = self._generate_visit_ages(
            individual_parameters_from_model_parameters
        )

        min_spacing = self.param_study.get("min_spacing_between_visits", 1 / 365)

        df_sim = self._generate_dataset(
            model,
            dict_timepoints,
            individual_parameters_from_model_parameters,
            min_spacing_between_visits=min_spacing,
        )

        simulated_data = Data.from_dataframe(df_sim)
        result_obj = Result(
            data=simulated_data,
            individual_parameters=individual_parameters_from_model_parameters,
            noise_std=model.parameters["noise_std"].numpy() * 100,
        )
        return result_obj
