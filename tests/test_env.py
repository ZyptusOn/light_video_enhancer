import os
import tempfile
import unittest
from unittest import mock

from light_video_enhancer import _env


class EnvironmentDiscoveryTests(unittest.TestCase):
    def test_py_launcher_parser_keeps_paths_with_spaces(self):
        output = (
            " -V:3.13 * C:\\Program Files\\Python313\\python.exe\n"
            " -V:3.8 C:\\Users\\demo\\Python38\\python.exe\n"
        )
        self.assertEqual(_env._parse_py_launcher_paths(output), [
            r"C:\Program Files\Python313\python.exe",
            r"C:\Users\demo\Python38\python.exe",
        ])

    @unittest.skipUnless(os.name == "nt", "Windows path layout test")
    def test_glob_discovery_includes_named_conda_env(self):
        with tempfile.TemporaryDirectory(prefix="lve-env-test-") as home:
            python_exe = os.path.join(home, ".conda", "envs", "rife", "python.exe")
            os.makedirs(os.path.dirname(python_exe))
            with open(python_exe, "wb") as handle:
                handle.write(b"test")
            environment = {
                "USERPROFILE": home,
                "LOCALAPPDATA": os.path.join(home, "AppData", "Local"),
                "APPDATA": os.path.join(home, "AppData", "Roaming"),
                "PROGRAMDATA": os.path.join(home, "ProgramData"),
                "ProgramFiles": os.path.join(home, "Program Files"),
                "ProgramFiles(x86)": os.path.join(home, "Program Files (x86)"),
            }
            with mock.patch.dict(os.environ, environment, clear=False):
                self.assertIn(python_exe, _env._glob_python_candidates())

    @unittest.skipUnless(os.name == "nt", "Windows alias test")
    def test_windows_store_alias_is_filtered(self):
        with tempfile.TemporaryDirectory(prefix="lve-env-test-") as root:
            alias = os.path.join(root, "Microsoft", "WindowsApps", "python.exe")
            real = os.path.join(root, "Python", "python.exe")
            os.makedirs(os.path.dirname(alias))
            os.makedirs(os.path.dirname(real))
            for path in (alias, real):
                with open(path, "wb") as handle:
                    handle.write(b"test")
            self.assertEqual(_env._dedupe([alias, real]), [real])


if __name__ == "__main__":
    unittest.main()
