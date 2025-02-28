import os

import pytest
from tests import LeaspyTestCase

from leaspy.algo.settings import OutputsSettings
from leaspy.api import Leaspy
from leaspy.io.logs import FitOutputManager


def test_fit_output_manager(tmp_path):
    data = LeaspyTestCase.get_suited_test_data_for_model("logistic_diag_noise")
    leaspy = Leaspy("logistic")
    algo_settings = LeaspyTestCase.get_algo_settings(
        name="mcmc_saem", n_iter=50, seed=0
    )

    output_settings = {
        "path": tmp_path,
        "print_periodicity": 10,
        "save_periodicity": 10,
        "plot_periodicity": 10,
        "plot_sourcewise": False,
        "overwrite_logs_folder": True,
        "nb_of_patients_to_plot": 5,
    }

    logs = OutputsSettings(output_settings)
    manager = FitOutputManager(logs)
    algo_settings.logs = logs
    leaspy.fit(data, settings=algo_settings)

    assert (manager.path_plot_patients / "plot_patients_10.pdf").exists()
    assert (manager.path_plot / "convergence_parameters.pdf").exists()

    files_list = os.listdir(manager.path_save_model_parameters_convergence)
    for parameter in leaspy.model.state._tracked_variables:
        assert any(file.startswith(str(parameter)) for file in files_list)
