import shutil
from pathlib import Path

from leaspy.algo import OutputsSettings
from leaspy.io.logs import FitOutputManager
from tests import LeaspyTestCase


class OutputSettingsAndFitOutputManagerTest(LeaspyTestCase):
    def test_constructor_no_save(self):
        logs = OutputsSettings(
            {
                "path": None,
                "print_periodicity": 42,
                "save_periodicity": None,
                "plot_periodicity": None,
                "overwrite_logs_folder": False,
            }
        )

        self.assertIsNone(logs.root_path)
        self.assertEqual(logs.print_periodicity, 42)
        self.assertIsNone(logs.save_periodicity)
        self.assertIsNone(logs.plot_periodicity)

        # only test __init__ method
        fm = FitOutputManager(logs)

    def test_constructor_try_to_plot_without_saving(self):
        with self.assertRaises(Exception):
            logs = OutputsSettings(
                {
                    "path": self.get_test_tmp_path("fake"),
                    "print_periodicity": 42,
                    "save_periodicity": None,
                    "plot_periodicity": 50,
                    "overwrite_logs_folder": False,
                }
            )

    def test_constructor_try_to_plot_not_multiple_of_saving(self):
        with self.assertRaises(Exception):
            logs = OutputsSettings(
                {
                    "path": self.get_test_tmp_path("fake"),
                    "print_periodicity": 42,
                    "save_periodicity": 60,
                    "plot_periodicity": 50,
                    "overwrite_logs_folder": False,
                }
            )

    def test_constructor_try_to_save_without_path_lead_to_default_path(self):
        default_logs_path = Path(OutputsSettings.DEFAULT_LOGS_DIR).resolve()
        try:
            shutil.rmtree(default_logs_path)
        except Exception:
            pass

        with self.assertWarns(UserWarning):
            logs = OutputsSettings(
                {
                    "path": None,
                    "print_periodicity": None,
                    "save_periodicity": 20,
                    "plot_periodicity": 40,
                    "overwrite_logs_folder": False,
                }
            )

        self.assertTrue(default_logs_path.is_dir())

        # only test __init__ method
        fm = FitOutputManager(logs)

        shutil.rmtree(default_logs_path)

        self.assertEqual(Path(logs.root_path), default_logs_path)
        self.assertIsNone(logs.print_periodicity)
        self.assertEqual(logs.save_periodicity, 20)
        self.assertEqual(logs.plot_periodicity, 40)

    def test_constructor_all_ok(self):
        path = Path(self.get_test_tmp_path("fake")).resolve()

        with self.assertWarnsRegex(
            UserWarning, "does not exist. Needed paths will be created"
        ):
            logs = OutputsSettings(
                {
                    "path": str(path),
                    "print_periodicity": 42,
                    "save_periodicity": 23,
                    "plot_periodicity": 46,
                    "overwrite_logs_folder": False,
                }
            )

        self.assertTrue(path.is_dir())
        self.assertEqual(Path(logs.root_path), path)
        self.assertEqual(logs.print_periodicity, 42)
        self.assertEqual(logs.save_periodicity, 23)
        self.assertEqual(logs.plot_periodicity, 46)

        # only test __init__ method
        fm = FitOutputManager(logs)
        self.assertEqual(fm.path_output, Path(logs.root_path))
        self.assertEqual(fm.periodicity_print, 42)
        self.assertEqual(fm.periodicity_save, 23)
        self.assertEqual(fm.periodicity_plot, 46)

    def test_constructor_bad_int(self):
        for bad_val in [-1, 0, "bad_type", ()]:
            with self.subTest(bad_val=bad_val):
                with self.assertWarnsRegex(
                    UserWarning, "The 'print_periodicity' parameter you provided"
                ):
                    logs = OutputsSettings(
                        {
                            "path": None,
                            "print_periodicity": bad_val,
                            "save_periodicity": None,
                            "plot_periodicity": None,
                            "overwrite_logs_folder": False,
                        }
                    )
                self.assertIsNone(logs.print_periodicity)
                fm = FitOutputManager(logs)
                self.assertIsNone(fm.periodicity_print)
