import os
import unittest
import warnings

from leaspy.io.logs.visualization.plotter import Plotter
from leaspy.io.outputs.result import Result
from leaspy.io.data.dataset import Dataset

from tests import LeaspyTestCase


class MatplotlibTestCase(LeaspyTestCase):

    @classmethod
    def setUpClass(cls) -> None:
        # for tmp handling
        super().setUpClass()

        # can not use the standard matplotlib backend on CI so use a fallback if needed
        cls.set_matplotlib_backend()

    @classmethod
    def set_matplotlib_backend(cls):
        import matplotlib
        if 'macos' in os.environ.get('CI_RUNNER_TAGS', ''):
            # this is needed because in MacOSX CI machine, the following immediately fails severely...
            matplotlib.use('pdf')
            # if CI machine has other matplotlib issues (fonts issues, ...),
            # try deleting the matplotlib cache dir (under .matplotlib/ at user home)
        else:
            try:
                matplotlib.pyplot.subplot(1, 1)
            except Exception:
                matplotlib.use('pdf')


class PlotterTest(MatplotlibTestCase):
    # only check that functions are running, not checking their results
    # TODO? use matplotlib.testing functions to do so?

    # TMP_REMOVE_AT_END = False

    @classmethod
    def setUpClass(cls) -> None:

        # for tmp handling & matplotlib proper backend
        super().setUpClass()

        cls.plotter = Plotter(cls.get_test_tmp_path())
        cls.plotter._show = False  # do not show plots useless for tests!

        cls.leaspy = cls.get_hardcoded_model('logistic_diag_noise')
        cls.ips = cls.get_from_personalize_individual_params('data_tiny-individual_parameters.csv')
        _, cls.ips_torch = cls.ips.to_pytorch()
        cls.data = cls.get_suited_test_data_for_model('logistic_diag_noise')
        cls.dataset = Dataset(cls.data)
        cls.result = Result(cls.data, cls.ips_torch)

        cls.inds = ['116', '142', '169']
        cls.ind = cls.inds[0]

    def setUp(self) -> None:
        warnings.simplefilter('ignore', DeprecationWarning)

    def test_plot_mean_trajectory(self):
        self.plotter.plot_mean_trajectory(self.leaspy.model, save_as='mean_trajectory.pdf')
        self.assertHasTmpFile('mean_trajectory.pdf')

    def test_plot_mean_validity(self):
        rel_path = 'mean_validity.pdf'
        self.plotter.plot_mean_validity(self.leaspy.model, self.result, save_as=rel_path)
        self.assertHasTmpFile(rel_path)

    def test_plot_patient_trajectory(self):
        rel_path = 'patient_trajectory.pdf'
        self.plotter.plot_patient_trajectory(self.leaspy.model, self.result, indices=self.inds,
                                             save_as=rel_path)
        self.assertHasTmpFile(rel_path)

    def test_plot_from_individual_parameters(self):
        rel_path = 'from_individual_parameters.pdf'
        self.plotter.plot_from_individual_parameters(self.leaspy.model, self.ips[self.ind],
                                                     timepoints=self.data.individuals[self.ind].timepoints,
                                                     save_as=rel_path)
        self.assertHasTmpFile(rel_path)

    def test_plot_distribution(self):

        to_plot = ['tau', 'xi'] + [f'sources_{i}' for i in range(self.leaspy.model.source_dimension)]

        for p in to_plot:
            rel_path = f'distribution_{p}.pdf'
            self.plotter.plot_distribution(self.result, p, save_as=rel_path)
            self.assertHasTmpFile(rel_path)

    def test_plot_correlation(self):
        for p1, p2 in [
            ('tau', 'xi'),
            ('xi', 'tau'),
            ('tau', 'sources_0'),
            ('xi', 'sources_1'),
            ('sources_0', 'sources_1'),
        ]:
            rel_path = f'correlation_{p1}_{p2}.pdf'
            self.plotter.plot_correlation(self.result, p1, p2, save_as=rel_path)
            self.assertHasTmpFile(rel_path)

    def test_plot_patients_mapped_on_mean_trajectory(self):
        self.plotter.plot_patients_mapped_on_mean_trajectory(self.leaspy.model, self.result)
        # no save

    ## plots used during convergence (staticmethods)
    def test_plot_error(self):
        rel_path = 'error.pdf'
        Plotter.plot_error(self.get_test_tmp_path(rel_path), dataset=self.dataset, model=self.leaspy.model,
                           param_ind=self.ips_torch)
        self.assertHasTmpFile(rel_path)

    def test_plot_patient_reconstructions(self):
        rel_path = 'patient_reconstructions.pdf'
        self.plotter.plot_patient_reconstructions(self.get_test_tmp_path(rel_path), dataset=self.dataset, model=self.leaspy.model,
                                                  param_ind=self.ips_torch)
        self.assertHasTmpFile(rel_path)

    def test_plot_param_ind(self):
        rel_path = 'param_ind.pdf'
        self.plotter.plot_param_ind(self.get_test_tmp_path(rel_path), param_ind=self.ips_torch.values())
        self.assertHasTmpFile(rel_path)

    @unittest.skip('Done automatically through FitOutputManager...')
    def test_plot_convergence_model_parameters(self):
        #self.plotter.plot_convergence_model_parameters(path_to_logs, path_cvg1, path_cvg2, model=self.leaspy.model)
        pass
