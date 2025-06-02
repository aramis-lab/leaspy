from leaspy.algo import (
    AlgorithmName,
    AlgorithmSettings,
    algorithm_factory,
    get_algorithm_class,
)
from leaspy.algo.fit import TensorMcmcSaemAlgorithm
from tests import LeaspyTestCase


class TestAlgoFactory(LeaspyTestCase):
    def test_algo(self):
        """Test attributes static method"""
        # Test for one name
        settings = AlgorithmSettings("mcmc_saem")
        algo = algorithm_factory(settings)
        self.assertIsInstance(algo, TensorMcmcSaemAlgorithm)

        # Test if raise ValueError if wrong string arg for name
        wrong_arg_exemples = ["mcmc", "blabla"]
        for wrong_arg in wrong_arg_exemples:
            settings.name = wrong_arg
            self.assertRaises(ValueError, algorithm_factory, settings)

    def test_get_class(self):
        algo_class = get_algorithm_class("mcmc_saem")
        self.assertIs(algo_class, TensorMcmcSaemAlgorithm)

        with self.assertRaises(ValueError):
            get_algorithm_class("unknown-algo")

    def test_loading_default_for_all_algos(self):
        # bit of a functional test
        for name in AlgorithmName:
            if name != AlgorithmName.SIMULATE:
                algo_instance = algorithm_factory(AlgorithmSettings(name.value))
                self.assertIsInstance(algo_instance, get_algorithm_class(name))

    def test_auto_burn_in(self):
        for algo_name in ("mcmc_saem", "mode_posterior", "mean_posterior"):
            with self.subTest(algo_name=algo_name):
                default_settings = AlgorithmSettings(algo_name)
                # get & check coherence of default parameters for those algos
                self.assertIsNone(default_settings.parameters["n_burn_in_iter"])
                default_n_iter = default_settings.parameters["n_iter"]
                self.assertIsNotNone(default_n_iter)
                self.assertIsInstance(default_n_iter, int)
                self.assertTrue(default_n_iter > 0)
                self.assertEqual(default_n_iter % 100, 0)
                default_burn_in_frac = default_settings.parameters[
                    "n_burn_in_iter_frac"
                ]
                self.assertIsNotNone(default_burn_in_frac)
                self.assertTrue(0 < default_burn_in_frac < 1, default_burn_in_frac)
                self.assertAlmostEqual((default_burn_in_frac * 100) % 1, 0, places=8)

                # check behavior
                algo = algorithm_factory(default_settings)
                self.assertEqual(
                    algo.algo_parameters["n_burn_in_iter"],
                    int(default_burn_in_frac * default_n_iter),
                )

                settings = AlgorithmSettings(algo_name, n_iter=2100)
                algo = algorithm_factory(settings)
                self.assertEqual(
                    algo.algo_parameters["n_burn_in_iter"],
                    int(default_burn_in_frac * 2100),
                )

                settings = AlgorithmSettings(algo_name, n_burn_in_iter_frac=0.80001)
                algo = algorithm_factory(settings)
                self.assertEqual(
                    algo.algo_parameters["n_burn_in_iter"], int(0.8 * default_n_iter)
                )

                settings = AlgorithmSettings(
                    algo_name, n_iter=1001, n_burn_in_iter_frac=0.8
                )
                algo = algorithm_factory(settings)
                self.assertEqual(algo.algo_parameters["n_burn_in_iter"], 800)

                # priority case, with warning
                settings = AlgorithmSettings(algo_name, n_burn_in_iter=42)
                with self.assertWarns(
                    FutureWarning
                ):  # warn because n_burn_in_iter_frac is not None
                    algo = algorithm_factory(settings)
                self.assertEqual(algo.algo_parameters["n_burn_in_iter"], 42)

                # explicit `n_burn_in_iter_frac=None` (no warning)
                settings = AlgorithmSettings(
                    algo_name, n_burn_in_iter=314, n_burn_in_iter_frac=None
                )
                algo = algorithm_factory(settings)
                self.assertEqual(algo.algo_parameters["n_burn_in_iter"], 314)

                # error case (both n_burn_in_iter_frac & n_burn_in_iter are None)
                settings = AlgorithmSettings(algo_name, n_burn_in_iter_frac=None)
                with self.assertRaises(ValueError):
                    algorithm_factory(settings)

    def test_auto_annealing(self):
        default_settings = AlgorithmSettings("mcmc_saem")
        default_n_iter = default_settings.parameters["n_iter"]

        # get & check coherence of default parameters for those algos
        self.assertFalse(default_settings.parameters["annealing"]["do_annealing"])
        self.assertIsNone(default_settings.parameters["annealing"]["n_iter"])
        default_annealing_iter_frac = default_settings.parameters["annealing"][
            "n_iter_frac"
        ]
        self.assertIsNotNone(default_annealing_iter_frac)
        self.assertTrue(
            0 < default_annealing_iter_frac < 1, default_annealing_iter_frac
        )
        self.assertAlmostEqual((default_annealing_iter_frac * 100) % 1, 0, places=8)

        settings = AlgorithmSettings("mcmc_saem", n_iter=1000)
        algo = algorithm_factory(settings)
        self.assertEqual(algo.algo_parameters["annealing"]["n_iter"], None)

        # also test new partial dictionary behavior for annealing
        settings = AlgorithmSettings(
            "mcmc_saem", n_iter=1001, annealing=dict(do_annealing=True)
        )
        algo = algorithm_factory(settings)
        self.assertEqual(
            algo.algo_parameters["annealing"]["n_iter"],
            int(default_annealing_iter_frac * 1001),
        )

        settings = AlgorithmSettings(
            "mcmc_saem", annealing=dict(do_annealing=True, n_iter_frac=0.40001)
        )
        algo = algorithm_factory(settings)
        self.assertEqual(
            algo.algo_parameters["annealing"]["n_iter"], int(default_n_iter * 0.4)
        )

        settings = AlgorithmSettings(
            "mcmc_saem", n_iter=1000, annealing=dict(do_annealing=True, n_iter_frac=0.3)
        )
        algo = algorithm_factory(settings)
        self.assertEqual(algo.algo_parameters["annealing"]["n_iter"], 300)

        # priority case, with warning
        settings = AlgorithmSettings(
            "mcmc_saem", annealing=dict(do_annealing=True, n_iter=42)
        )
        with self.assertWarns(
            FutureWarning
        ):  # warn because annealing.n_iter_frac is not None
            algo = algorithm_factory(settings)
        self.assertEqual(algo.algo_parameters["annealing"]["n_iter"], 42)

        # explicit `n_burn_in_iter_frac=None` (no warning)
        settings = AlgorithmSettings(
            "mcmc_saem", annealing=dict(do_annealing=True, n_iter=314, n_iter_frac=None)
        )
        algo = algorithm_factory(settings)
        self.assertEqual(algo.algo_parameters["annealing"]["n_iter"], 314)

        # error case (both n_burn_in_iter_frac & n_burn_in_iter are None)
        settings = AlgorithmSettings(
            "mcmc_saem", annealing=dict(do_annealing=True, n_iter_frac=None)
        )
        with self.assertRaises(ValueError):
            algorithm_factory(settings)
