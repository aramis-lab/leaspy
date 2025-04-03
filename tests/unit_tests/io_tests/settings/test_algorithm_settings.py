import json
from pathlib import Path

import torch

from leaspy.algo import AlgorithmSettings, algo_default_data_dir
from tests import LeaspyTestCase


class AlgorithmSettingsTest(LeaspyTestCase):
    @classmethod
    def path_mock_settings_for_loading(cls, name: str):
        return Path(
            cls.get_test_data_path(
                "settings", "algo", "only_algo_settings_load_unittest", f"{name}.json"
            )
        )

    @classmethod
    def path_default_algo(cls, name: str):
        return Path(algo_default_data_dir, f"default_{name}.json")

    def test_default_constructor(self):
        # Default constructor
        name = "scipy_minimize"
        with self.path_default_algo(name).open("r") as fp:
            json_data = json.load(fp)

        settings = AlgorithmSettings(name, algorithm_initialization_method="blabla")
        self.assertEqual(settings.name, name)
        self.assertEqual(settings.parameters, json_data["parameters"])
        self.assertEqual(settings.parameters["use_jacobian"], True)
        self.assertEqual(settings.seed, None)
        self.assertEqual(settings.algorithm_initialization_method, "blabla")

    def test_unknown_algo(self):
        with self.assertRaisesRegex(
            ValueError, "'unknown-algo' is not a valid AlgorithmName"
        ):
            AlgorithmSettings("unknown-algo")

    def test_check_consistency_params(self):
        with self.assertWarnsRegex(UserWarning, "seed"):
            # deterministic, no need for seed
            a = AlgorithmSettings("constant_prediction", seed=42)
        # self.assertIsNone(a.seed)  # set anyway

        with self.assertWarnsRegex(UserWarning, "seed"):
            # bad seed
            a = AlgorithmSettings("mode_posterior", seed="not-castable-to-int")
        self.assertIsNone(a.seed)

        with self.assertWarnsRegex(UserWarning, "model_initialization_method"):
            # `model_initialization_method` is only for fit algo!
            a = AlgorithmSettings(
                "constant_prediction", model_initialization_method="default"
            )
        # self.assertIsNone(a.model_initialization_method)  # set anyway

    def test_set_logs(self):
        a = AlgorithmSettings("constant_prediction")
        with self.assertRaisesRegex(ValueError, "save_periodicity"):
            a.set_logs(save_periodicity=0.5)

        with self.assertRaisesRegex(ValueError, "overwrite_logs_folder"):
            a.set_logs(overwrite_logs_folder="not-a-bool")

        with self.assertWarnsRegex(UserWarning, "ignored_log_param"):
            a.set_logs(
                ignored_log_param=True,
                # to avoid creating directories
                save_periodicity=None,
                plot_periodicity=None,
            )

    def test_jacobian_personalization(self):
        settings = AlgorithmSettings("scipy_minimize", use_jacobian=False)
        self.assertEqual(settings.parameters["use_jacobian"], False)

    def test_constant_prediction_algorithm(self):
        settings = AlgorithmSettings("constant_prediction")
        self.assertEqual(settings.name, "constant_prediction")
        self.assertDictEqual(settings.parameters, {"prediction_type": "last"})

        for prediction_type in ["last", "last_known", "max", "mean"]:
            settings = AlgorithmSettings(
                "constant_prediction", prediction_type=prediction_type
            )
            self.assertEqual(settings.name, "constant_prediction")
            self.assertDictEqual(
                settings.parameters, {"prediction_type": prediction_type}
            )

    def test_lme_fit_algorithm(self):
        settings = AlgorithmSettings("lme_fit")
        self.assertEqual(settings.name, "lme_fit")

    def test_lme_personalize_algorithm(self):
        settings = AlgorithmSettings("lme_personalize")
        self.assertEqual(settings.name, "lme_personalize")

    def test_default_constructor_with_kwargs(self):
        # Default constructor with kwargs
        name = "mcmc_saem"
        with self.path_default_algo(name).open("r") as fp:
            json_data = json.load(fp)

        # also test new partial dictionary behavior for annealing
        settings = AlgorithmSettings(
            name, seed=10, n_iter=2100, annealing={"do_annealing": True}
        )

        json_data["parameters"]["n_iter"] = 2100
        json_data["parameters"]["annealing"]["do_annealing"] = True

        # those 2 "derived" parameters are now set in algo initialization (not in AlgorithmSettings any more)
        # json_data['parameters']['n_burn_in_iter'] = int(0.9*2100)
        # json_data['parameters']['annealing']['n_iter'] = int(0.5*2100)

        self.assertEqual(settings.name, name)
        self.assertEqual(settings.parameters, json_data["parameters"])
        self.assertEqual(settings.seed, 10)
        self.assertEqual(settings.parameters["progress_bar"], True)

    def test_algo_settings_init_with_parameters(self):
        # unknown algo parameters
        with self.assertWarns(UserWarning):
            algo_settings = AlgorithmSettings(
                "mode_posterior", unknown_param_1=1, unknown_param_2={"2": 2}
            )

        # but we set the parameters nonetheless (possibly needed for backward compat / "hidden" params)
        self.assertIn("unknown_param_1", algo_settings.parameters)
        self.assertIn("unknown_param_2", algo_settings.parameters)
        self.assertEqual(algo_settings.parameters["unknown_param_1"], 1)
        self.assertEqual(algo_settings.parameters["unknown_param_2"], {"2": 2})

        # the existing dictionary keys are merged (recursively)
        algo_settings = AlgorithmSettings(
            "mcmc_saem", annealing=dict(do_annealing=True)
        )
        default_mcmc_saem_params = AlgorithmSettings("mcmc_saem").parameters
        default_mcmc_saem_params["annealing"]["do_annealing"] = True
        self.assertEqual(algo_settings.parameters, default_mcmc_saem_params)

        with self.assertRaises(ValueError):
            # we expect a dict!
            algo_settings = AlgorithmSettings("mcmc_saem", annealing=True)

    def check_loaded_algorithm_settings_and_json_data_match(self, path):
        with open(path) as fp:
            json_data = json.load(fp)

        self.assertIn("name", json_data.keys())
        self.assertIn("parameters", json_data.keys())

        settings = AlgorithmSettings.load(path)
        self.assertEqual(settings.name, json_data["name"])
        self.assertEqual(settings.parameters, json_data["parameters"])

        return json_data

    def test_constructor_by_loading_json_bad_dict_param(self):
        # Constructor by loading a json file
        path = self.path_mock_settings_for_loading("mcmc_saem_bad_dict_param")
        with self.assertRaisesRegex(ValueError, "annealing"):
            AlgorithmSettings.load(str(path))

    def test_constructor_by_loading_json_unknown_param(self):
        # Constructor by loading a json file
        path = self.path_mock_settings_for_loading("mcmc_saem_unknown_param")
        with self.assertWarnsRegex(UserWarning, "fake_param"):
            settings = AlgorithmSettings.load(str(path))
        self.assertIn("fake_param", settings.parameters)
        self.assertEqual(settings.parameters["fake_param"], "hello")

    def test_constructor_by_loading_json_partial_dict(self):
        # Constructor by loading a json file
        path = self.path_mock_settings_for_loading("mcmc_saem_partial_dict")

        settings = AlgorithmSettings.load(str(path))

        with self.path_default_algo("mcmc_saem").open("r") as fp:
            default_settings = json.load(fp)

        modified_hyperparams = {
            "seed": 42,
        }
        for k, v in modified_hyperparams.items():
            default_settings[k] = v

        modified_params = {
            "n_iter": 2000,
            "n_burn_in_iter": 1700,
            "sampler_pop": "Fake",
            "sampler_pop_params": {"acceptation_history_length": 50},
            "annealing": {"do_annealing": True},
        }
        for k, v in modified_params.items():
            # maximum 1 level of nesting in test data so the following is correct
            default_settings["parameters"][k] = (
                v
                if not isinstance(v, dict)
                else {**default_settings["parameters"][k], **v}
            )

        for k, v in vars(settings).items():
            if k.startswith("__") or k == "logs":
                continue
            else:
                self.assertEqual(v, default_settings[k], msg=k)

    def test_save(self):
        algo_settings = AlgorithmSettings("mcmc_saem", seed=42)

        # add logs
        path_logs = self.get_test_tmp_path("logs")
        with self.assertWarnsRegex(
            UserWarning, "does not exist"
        ):  # path to be created, with a warning
            algo_settings.set_logs(
                path_logs,
                console_print_periodicity=50,
                save_periodicity=50,
                plot_periodicity=100,
            )

        # save the settings
        path_saved = self.get_test_tmp_path("mcmc_algo_settings_saved.json")
        algo_settings.save(path_saved)

        json_data = self.check_loaded_algorithm_settings_and_json_data_match(path_saved)

        self.assertEqual(
            json_data.keys(),
            {
                "name",
                "seed",
                "algorithm_initialization_method",
                "model_initialization_method",
                "parameters",
                "device",
                # 'logs'
            },
        )
        self.assertEqual(json_data["name"], "mcmc_saem")
        self.assertEqual(json_data["seed"], 42)

    def test_device(self):
        algo_settings_cpu = AlgorithmSettings("mcmc_saem")
        self.assertEqual(algo_settings_cpu.device, "cpu")

        algo_settings_cpu = AlgorithmSettings("mcmc_saem", device="cpu")
        self.assertEqual(algo_settings_cpu.device, "cpu")

        algo_settings_cpu = AlgorithmSettings("mcmc_saem", device=torch.device("cpu"))
        self.assertEqual(algo_settings_cpu.device, "cpu")

        algo_settings_cuda = AlgorithmSettings("mcmc_saem", device="cuda")
        self.assertEqual(algo_settings_cuda.device, "cuda")

        algo_settings_cuda = AlgorithmSettings("mcmc_saem", device=torch.device("cuda"))
        self.assertEqual(algo_settings_cuda.device, "cuda")

        algo_settings_cuda = AlgorithmSettings("mcmc_saem", device="cuda:2")
        self.assertEqual(algo_settings_cuda.device, "cuda")

        algo_settings_cuda = AlgorithmSettings(
            "mcmc_saem", device=torch.device("cuda:2")
        )
        self.assertEqual(algo_settings_cuda.device, "cuda")

        # device is supported only for fit algorithms
        with self.assertWarnsRegex(
            UserWarning, "does not support user-specified devices"
        ):
            _ = AlgorithmSettings("scipy_minimize", device=torch.device("cuda"))
