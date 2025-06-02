from unittest import skip

import matplotlib.pyplot as plt

from leaspy.io.data.dataset import Dataset
from leaspy.io.logs.visualization.plotting import Plotting
from leaspy.io.outputs.result import Result

from .test_plotter import MatplotlibTestCase


class PlottingTest(MatplotlibTestCase):
    # only check that functions are running, not checking their results
    # TODO? use matplotlib.testing functions to do so?

    # TMP_REMOVE_AT_END = False

    @classmethod
    def setUpClass(cls) -> None:
        # for tmp handling & matplotlib proper backend
        super().setUpClass()
        cls.leaspy = cls.get_hardcoded_model("logistic_diag_noise")
        cls.ips = cls.get_from_personalize_individual_params(
            "data_tiny-individual_parameters.csv"
        )
        _, cls.ips_torch = cls.ips.to_pytorch()
        cls.data = cls.get_suited_test_data_for_model("logistic_diag_noise")
        cls.dataset = Dataset(cls.data)
        cls.result = Result(cls.data, cls.ips_torch)
        cls.inds = ["116", "142", "169"]
        cls.ind = cls.inds[0]

    def setUp(self) -> None:
        with self.assertWarns(FutureWarning):
            self.p = Plotting(self.leaspy.model, self.get_test_tmp_path())

    def test_average_trajectory(self):
        self.p.average_trajectory(save_as="average_trajectory.pdf")
        plt.close()  # TODO? directly in method?
        self.assertHasTmpFile("average_trajectory.pdf")

    @skip(
        "Broken: 'MultivariateModel' object has no attribute 'compute_individual_tensorized'"
    )
    def test_patient_trajectories_with_patient_idx(self):
        rel_path = "patient_trajectories_sub.pdf"
        self.p.patient_trajectories(
            self.data,
            patients_idx=self.inds,
            individual_parameters=self.ips,
            save_as=rel_path,
        )
        plt.close()  # TODO? directly in method?
        self.assertHasTmpFile(rel_path)

    @skip(
        "Broken: 'MultivariateModel' object has no attribute 'compute_individual_tensorized'"
    )
    def test_patient_trajectories(self):
        rel_path = "patient_trajectories.pdf"
        self.p.patient_trajectories(
            self.data,
            individual_parameters=self.ips,
            save_as=rel_path,
        )
        plt.close()  # TODO? directly in method?
        self.assertHasTmpFile(rel_path)

    @skip(
        "Broken: 'MultivariateModel' object has no attribute 'compute_individual_tensorized'"
    )
    def test_patient_trajectories_reparam_with_patient_idx(self):
        rel_path = "patient_trajectories_sub_reparam.pdf"
        self.p.patient_trajectories(
            self.data,
            patients_idx=self.inds,
            individual_parameters=self.ips,
            reparametrized_ages=True,
            save_as=rel_path,
        )
        plt.close()  # TODO? directly in method?
        self.assertHasTmpFile(rel_path)

    @skip(
        "Broken: 'MultivariateModel' object has no attribute 'compute_individual_tensorized'"
    )
    def test_patient_trajectories_reparam(self):
        rel_path = "patient_trajectories_reparam.pdf"
        self.p.patient_trajectories(
            self.data,
            individual_parameters=self.ips,
            reparametrized_ages=True,
            save_as=rel_path,
        )
        plt.close()  # TODO? directly in method?
        self.assertHasTmpFile(rel_path)

    def test_patient_observations_with_patient_idx(self):
        rel_path = "patient_observations_sub.pdf"
        self.p.patient_observations(
            self.data,
            patients_idx=self.inds,
            save_as=rel_path,
        )
        plt.close()  # TODO? directly in method?
        self.assertHasTmpFile(rel_path)

    def test_patient_observations(self):
        rel_path = "patient_observations_all.pdf"
        self.p.patient_observations(self.data, save_as=rel_path)
        plt.close()  # TODO? directly in method?
        self.assertHasTmpFile(rel_path)

    def test_patient_observations_reparametrized(self):
        rel_path = "patient_observations_sub_reparam.pdf"
        self.p.patient_observations(
            self.data,
            patients_idx=self.inds,
            individual_parameters=self.ips,
            save_as=rel_path,
        )
        plt.close()  # TODO? directly in method?
        self.assertHasTmpFile(rel_path)

        rel_path = "patient_observations_sub_reparam_bis.pdf"
        self.p.patient_observations_reparametrized(
            self.data,
            patients_idx=self.inds,
            individual_parameters=self.ips,
            save_as=rel_path,
        )
        plt.close()  # TODO? directly in method?
        self.assertHasTmpFile(rel_path)

        rel_path = "patient_observations_all_reparam.pdf"
        self.p.patient_observations(
            self.data,
            individual_parameters=self.ips,
            save_as=rel_path,
        )
        plt.close()  # TODO? directly in method?
        self.assertHasTmpFile(rel_path)

        rel_path = "patient_observations_all_reparam_bis.pdf"
        self.p.patient_observations_reparametrized(
            self.data,
            individual_parameters=self.ips,
            save_as=rel_path,
        )
        plt.close()  # TODO? directly in method?
        self.assertHasTmpFile(rel_path)
