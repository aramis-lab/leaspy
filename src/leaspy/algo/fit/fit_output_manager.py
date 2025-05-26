import re
import time
from pathlib import Path
from typing import Iterable, Optional

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib import colormaps
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D

from leaspy.io.data import Dataset
from leaspy.models import McmcSaemCompatibleModel

from .abstract_fit_algo import AbstractFitAlgo


class FitOutputManager:
    """
    Class used by :class:`~leaspy.algo.AbstractAlgo` (and its child classes) to display & save plots and statistics during algorithm execution.

    Parameters
    ----------
    outputs : :class:`~leaspy.algo.OutputsSettings`
        Initialize the `FitOutputManager` class attributes, like the logs paths, the console print periodicity and so forth.

    Attributes
    ----------
    path_output : :obj:`str`
        Path of the folder containing all the outputs
    path_plot : :obj:`str`
        Path of the subfolder of path_output containing the logs plots
    path_plot_convergence_model_parameters : :obj:`str`
        Path of the first plot of the convergence of the model's parameters (in the subfolder path_plot)
    path_plot_patients : :obj:`str`
        Path of the subfolder of path_plot containing the plot of the reconstruction of the patients' longitudinal
        trajectory by the model
    nb_of_patients_to_plot : :obj:`int`
        Number of patients for whom the reconstructions will be plotted.
    path_save_model_parameters_convergence : :obj:`str`
        Path of the subfolder of path_output containing the progression of the model's parameters convergence
    periodicity_plot : :obj:`int` (default 100)
        Set the frequency of the display of the plots
    periodicity_print : :obj:`int`
        Set the frequency of the display of the statistics
    periodicity_save : :obj:`int`
        Set the frequency of the saves of the model's parameters
    """

    def __init__(self, outputs):
        self.periodicity_print = outputs.print_periodicity
        self.periodicity_save = outputs.save_periodicity
        self.periodicity_plot = outputs.plot_periodicity
        self.nb_of_patients_to_plot = outputs.nb_of_patients_to_plot
        self.plot_sourcewise = outputs.plot_sourcewise
        if outputs.root_path is not None:
            self.path_output = Path(outputs.root_path)
            self.path_plot = Path(outputs.plot_path)
            self.path_plot_patients = Path(outputs.patients_plot_path)
            self.path_save_model_parameters_convergence = Path(
                outputs.parameter_convergence_path
            )
            self.path_plot_convergence_model_parameters = (
                self.path_plot / "convergence_parameters.pdf"
            )
        self.time = time.time()

    def iteration(
        self,
        algo: AbstractFitAlgo,
        model: McmcSaemCompatibleModel,
        data: Dataset,
    ) -> None:
        """
        Call methods to save state of the running computation, display statistics & plots if the current iteration
        is a multiple of `periodicity_print`, `periodicity_plot` or `periodicity_save`

        Parameters
        ----------
        algo : :class:`~leaspy.algo.fit.AbstractFitAlgo`
            The running algorithm
        model : :class:`~leaspy.models.McmcSaemCompatibleModel`
            The model used by the computation
        data : :class:`.Dataset`
            The data used by the computation
        """

        # <!> only `current_iteration` defined for AbstractFitAlgorithm... TODO -> generalize where possible?
        if not hasattr(algo, "current_iteration"):
            # emit a warning?
            return
        iteration = algo.current_iteration

        if self.path_output is None:
            return

        if self.periodicity_print is not None:
            if iteration == 0 or iteration % self.periodicity_print == 0:
                self.print_algo_statistics(algo)
                self.print_model_statistics(model)
                self.print_time()

        if self.periodicity_save is not None:
            if iteration == 0 or iteration % self.periodicity_save == 0:
                self.save_plot_patient_reconstructions(iteration, model, data)
                self.save_model_parameters_convergence(iteration, model)

        if self.periodicity_plot is not None:
            if iteration % self.periodicity_plot == 0:
                self.save_plot_convergence_model_parameters(model)

    def print_time(self):
        """
        Prints the duration since the last periodic point
        """
        current_time = time.time()
        print(f"Duration since last print: {current_time - self.time:.3f}s")
        self.time = current_time

    def print_model_statistics(self, model: McmcSaemCompatibleModel):
        """
        Prints model's statistics

        Parameters
        ----------
        model : :class:`~leaspy.models.McmcSaemCompatibleModel`
            The model used by the computation
        """
        print(model)

    def print_algo_statistics(self, algo: AbstractFitAlgo):
        """
        Prints algorithm's statistics

        Parameters
        ----------
        algo : :class:`~leaspy.algo.fit.AbstractFitAlgo`
            The running algorithm
        """
        print(algo)

    def save_model_parameters_convergence(
        self, iteration: int, model: McmcSaemCompatibleModel
    ) -> None:
        """
        Saves the current state of the model's parameters

        Parameters
        ----------
        iteration : :obj:`int`
            The current iteration
        model : :class:`~.models.abstract_model.McmcSaemCompatibleModel`
            The model used by the computation
        """
        model.state.save(
            str(self.path_save_model_parameters_convergence),
            iteration=iteration,
        )

    def save_plot_convergence_model_parameters(self, model: McmcSaemCompatibleModel):
        """
        Saves figures of the model parameters' convergence in multiple pages of a PDF.

        Parameters
        ----------
        model : :class:`~leaspy.models.McmcSaemCompatibleModel`
            The model used by the computation
        """
        width = 10
        height_per_row = 3.5

        to_skip = {"betas", "sources", "space_shifts", "xi", "tau", "xi_mean"}
        if model.name == "ordinal":
            to_skip.add("deltas")
        params_with_feature_labels = ["g", "v0"]
        params_with_sources = ["mixing_matrix"]
        params_with_events = []
        if model.name == "joint":
            to_skip.add("survival_shifts")
            params_with_sources.append("zeta")
            params_with_events += ["nu", "rho"]

        params_to_plot = list(model.state.tracked_variables - to_skip)

        files_to_plot = self._get_files_related_to_parameters(params_to_plot)
        # To plot related parameters close to each other, we sort the list
        files_to_plot.sort()

        # If plot sourcewise is true, new sourcewise csv files will be created

        if self.plot_sourcewise and hasattr(model, "source_dimension"):
            files_to_plot = [
                file
                for file in files_to_plot
                if not any(file.name.startswith(param) for param in params_with_sources)
            ]
            files_to_plot.extend(
                self._compute_files_sourcewise(
                    params_with_sources, model.source_dimension
                )
            )

        n_plots = len(files_to_plot)
        with PdfPages(self.path_plot_convergence_model_parameters) as pdf:
            for page in range(0, n_plots, 6):
                # 6 plots per page
                _, ax = plt.subplots(3, 2, figsize=(width, 3 * height_per_row))
                ax = ax.flatten()

                for i, file_path in enumerate(files_to_plot[page : page + 6]):
                    parameter_name = file_path.name.split(".csv")[0]
                    parameter_name, index = self._extract_parameter_name_and_index(
                        parameter_name
                    )
                    df_convergence = pd.read_csv(file_path, index_col=0, header=None)
                    ax[i].plot(df_convergence)

                    ax = self._set_title_for_parameter(
                        ax, i, parameter_name, model, index
                    )

                    ax = self._set_legend_for_parameters(
                        ax,
                        i,
                        parameter_name,
                        params_with_feature_labels,
                        params_with_sources,
                        params_with_events,
                        model,
                    )

                plt.tight_layout()
                pdf.savefig()
                plt.close()

    def save_plot_patient_reconstructions(
        self,
        iteration: int,
        model: McmcSaemCompatibleModel,
        data: Dataset,
    ) -> None:
        """
        Saves figures of real longitudinal values and their reconstructions computed by the model for maximum
        5 patients during each iteration.

        Parameters
        ----------
        iteration : :obj:`int`
            The current iteration
        model : :class:`~leaspy.models.McmcSaemCompatibleModel`
            The model used by the computation
        data : :class:`~leaspy.io.data.Dataset`
            The dataset used by the computation
        """
        number_of_patient_plot = min(self.nb_of_patients_to_plot, data.n_individuals)
        individual_parameters_dict = {
            variable: model.state.get_tensor_value(variable)
            for variable in model.individual_variables_names
        }

        colors = colormaps["Dark2"](np.linspace(0, 1, number_of_patient_plot + 2))

        fig, ax = plt.subplots(1, 1)
        ax.set_title(f"Feature trajectory for {number_of_patient_plot} patients")
        ax.set_xlabel("Ages")
        ax.set_ylabel("Normalized Feature Value")

        for i in range(number_of_patient_plot):
            times_patient = data.get_times_patient(i).cpu().detach().numpy()
            true_values_patient = data.get_values_patient(i).cpu().detach().numpy()
            ip_patient = {pn: pv[i] for pn, pv in individual_parameters_dict.items()}

            reconstruction_values_patient = (
                model.compute_individual_trajectory(times_patient, ip_patient)
                .squeeze(0)
                .numpy()
            )
            ax.plot(times_patient, reconstruction_values_patient, c=colors[i])
            ax.plot(
                times_patient,
                true_values_patient,
                c=colors[i],
                linestyle="--",
                marker="o",
            )

            last_time_point = times_patient[-1]
            last_reconstruction_value = reconstruction_values_patient.flatten()[-1]
            ax.text(
                last_time_point,
                last_reconstruction_value,
                data.indices[i],
                color=colors[i],
            )

        min_time, max_time = np.percentile(
            data.timepoints[data.timepoints > 0.0].cpu().detach().numpy(),
            [10, 90],
        )
        timepoints_np = np.linspace(min_time, max_time, 100)
        model_values_np = model.compute_mean_traj(
            torch.tensor(np.expand_dims(timepoints_np, 0))
        )

        for feature in range(model.dimension):
            ax.plot(
                timepoints_np,
                model_values_np[0, :, feature],
                c="gray",
                linewidth=3,
                alpha=0.3,
            )

        line_rec = Line2D([0], [0], label="Reconstructions", color="black")
        line_real = Line2D(
            [0], [0], label="Real feature values", color="black", linestyle="--"
        )
        line_avg = Line2D([0], [0], label="Global avg. features", color="gray")

        handles, labels = ax.get_legend_handles_labels()
        handles.extend([line_rec, line_real, line_avg])

        ax.legend(handles=handles)
        path_iteration = self.path_plot_patients / f"plot_patients_{iteration}.pdf"

        plt.savefig(path_iteration)
        plt.close()

    def _get_files_related_to_parameters(self, parameters: Iterable[str]) -> list[Path]:
        """
        Retrieve the list of file paths related to the given parameters.

        Parameters
        ----------
        parameters : Iterable[:obj:`str`]
            A list of parameter names.

        Returns
        -------
        list[Path]
            A list of file paths that match the parameter names.
        """
        return [
            f
            for f in self.path_save_model_parameters_convergence.iterdir()
            if any(f.name.startswith(param) for param in parameters)
        ]

    def _extract_parameter_name_and_index(
        self, parameter_name: str
    ) -> tuple[Optional[str], Optional[int]]:
        """
         Extract the parameter name and its corresponding index (if applicable) from the given parameter name.

         Parameters
         ----------
         parameter_name : :obj:`str`
             The name of the parameter to extract information from.

         Returns
         -------
         tuple[Optional[:obj:`str`], Optional[:obj:`int`]]
             A tuple where the first element is the parameter name and the second is its index (if applicable).

         Examples
         --------
         >>> _extract_parameter_name_and_index("mixing_matrix_10")
        ('mixing_matrix', 10)
         >>> _extract_parameter_name_and_index("mixing_matrix_1")
         ('mixing_matrix', 1)
         >>> _extract_parameter_name_and_index("mixing_matrix_")
         ('mixing_matrix_', None)
         >>> _extract_parameter_name_and_index("mixing_matrix_10_foo")
        ('mixing_matrix_10_foo', None)
        >>> _extract_parameter_name_and_index("v0")
        ('v0', None)
        """
        if parameter_name == "v0":
            return parameter_name, None
        match = re.search(r"^(.*?)(\d+)$", parameter_name)
        if match:
            return match.group(1).strip("_"), int(match.group(2))
        else:
            return parameter_name, None

    def _set_title_for_parameter(
        self,
        ax: plt.Axes,
        i: int,
        parameter_name: str,
        model: McmcSaemCompatibleModel,
        index: int,
    ) -> plt.Axes:
        """
        Set the title for the plot based on the parameter name and the model's features or other information.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axes object of the plot.
        i : :obj:`int`
            The index of the current plot.
        parameter_name : :obj:`str`
            The name of the parameter being plotted.
        model : :class:`~leaspy.models.McmcSaemCompatibleModel`
            The model containing features and event information.
        index : :obj:`int`
            The index of the specific feature or event, or None if not applicable.

        Returns
        -------
        matplotlib.axes.Axes
            The axes object with the title set.
        """
        if parameter_name == "mixing_matrix":
            ax[i].set_title(parameter_name + " " + model.features[index])
        elif parameter_name == "zeta":
            ax[i].set_title(parameter_name + " " + "event" + " " + str(index + 1))
        elif parameter_name.startswith("sourcewise"):
            ax[i].set_title(
                parameter_name.replace("sourcewise_", "")
                + " "
                + "source"
                + " "
                + str(index)
            )
        else:
            ax[i].set_title(parameter_name)
        return ax

    def _set_legend_for_parameters(
        self,
        ax: plt.Axes,
        i: int,
        parameter_name: str,
        params_with_feature_labels: Iterable[str],
        params_with_sources: Iterable[str],
        params_with_events: Iterable[str],
        model: McmcSaemCompatibleModel,
    ) -> plt.Axes:
        """
        Set the legend for the plot based on the parameter name and the model's features, sources, or events.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axes object of the plot.
        i : :obj:`int`
            The index of the current plot.
        parameter_name : :obj:`str`
            The name of the parameter being plotted.
        params_with_feature_labels : list[:obj:`str`]
            A list of parameters associated with feature labels.
        params_with_sources : list[:obj:`str`]
            A list of parameters associated with sources.
        params_with_events : list[:obj:`str`]
            A list of parameters associated with events.
        model : :class:`~leaspy.models.McmcSaemCompatibleModel`
            The model containing the necessary information for legends.

        Returns
        -------
        matplotlib.axes.Axes
            The axes object with the legend set.
        """
        if parameter_name in params_with_feature_labels:
            ax[i].legend(model.features, loc="best")
        if hasattr(model, "source_dimension") and parameter_name in params_with_sources:
            sources = [
                "Source" + " " + str(i + 1) for i in range(model.source_dimension)
            ]
            ax[i].legend(sources, loc="best")
        if hasattr(model, "nb_events"):
            if parameter_name in params_with_events:
                events = ["Event" + " " + str(i + 1) for i in range(model.nb_events)]
                ax[i].legend(events, loc="best")
            if parameter_name.startswith("sourcewise"):
                if "mixing_matrix" in parameter_name:
                    ax[i].legend(model.features, loc="best")
                if "zeta" in parameter_name:
                    events = [
                        "Event" + " " + str(i + 1) for i in range(model.nb_events)
                    ]
                    ax[i].legend(events, loc="best")
        return ax

    def _compute_files_sourcewise(
        self, params_with_sources: Iterable[str], num_sources: int
    ) -> list[Path]:
        """
        This function processes parameters that have sources and generates new csv files for each source.

        Parameters
        ----------
        params_with_sources : list[:obj:`str`]
            A list of parameters that have associated source data.
        num_sources : :obj:`int`
            The number of sources in model

        Returns
        -------
        list[Path]
            The list of newly created sourcewise files.
        """
        new_files = []
        for param_name in params_with_sources:
            related_files = self._get_files_related_to_parameters([param_name])
            if not related_files:
                continue
            related_files.sort()

            for source_idx in range(num_sources):
                new_files.append(
                    self._concatenate_and_save_sourcewise_parameters(
                        param_name, related_files, source_idx
                    )
                )
        return new_files

    def _concatenate_and_save_sourcewise_parameters(
        self, parameter_name: str, files: Iterable[Path], source_idx: int
    ) -> Path:
        """
        Concatenates data for the specific source and saves it as a csv file.

        Parameters
        ----------
        parameter_name : :obj:`str`
              A sourcewise parameter
        files : Iterable[Path]
              A list of file paths containing sourcewise parameter
        source_idx : :obj:`int`
              The index of the source
        Returns
        -------
        Path
            The path to the saved csv file containing data for the specified source
        """
        df = self._concatenate_parameters(files, source_idx)
        file_name = f"sourcewise_{parameter_name}_{source_idx + 1}.csv"
        file_path = self.path_save_model_parameters_convergence / file_name
        df.to_csv(file_path, header=False)
        return file_path

    def _concatenate_parameters(
        self, files: Iterable[Path], source_idx: int
    ) -> pd.DataFrame:
        """
        Extracts and concatenates a specific column from multiple csv files into a single DataFrame

        Parameters
        ----------
        files : Iterable[Path]
            A list of file paths
        source_idx : :obj: `int`
            The index of the source

        Returns
        -------
        :obj: `pd.DataFrame`
            A DataFrame with each column representing the selected source's data from multiple files
        """
        return pd.concat(
            [
                pd.read_csv(file_path, index_col=0, header=None).iloc[:, source_idx]
                for file_path in files
            ],
            axis=1,
            join="inner",
        )
