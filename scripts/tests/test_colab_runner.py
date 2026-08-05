# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for colab_runner.py."""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

# Module hack
LIB_PATH = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, os.path.abspath(LIB_PATH))

from scripts import colab_runner  # noqa: E402


class TestGetColabBinary(unittest.TestCase):

  @patch.dict(os.environ, {"COLAB_CLI_PATH": "/custom/colab"})
  @patch("os.path.exists", return_value=True)
  def test_env_var_override(self, mock_exists: MagicMock) -> None:
    binary = colab_runner.get_colab_binary()
    self.assertEqual(binary, "/custom/colab")

  @patch.dict(os.environ, {}, clear=True)
  @patch("shutil.which", return_value="/usr/local/bin/colab")
  def test_path_lookup(self, mock_which: MagicMock) -> None:
    binary = colab_runner.get_colab_binary()
    self.assertEqual(binary, "/usr/local/bin/colab")


  @patch.dict(os.environ, {}, clear=True)
  @patch("shutil.which", return_value=None)
  @patch("os.path.exists", return_value=False)
  def test_not_found_raises(
      self, mock_which: MagicMock, mock_exists: MagicMock
  ) -> None:
    with self.assertRaises(FileNotFoundError):
      colab_runner.get_colab_binary()


class TestColabRunner(unittest.TestCase):

  @patch("subprocess.run")
  def test_provision_and_stop(self, mock_run: MagicMock) -> None:
    runner = colab_runner.ColabRunner(
        session_name="test-session",
        accelerator="v6e1",
        binary_path="/usr/bin/colab",
    )
    runner.provision_session()
    mock_run.assert_called_with(
        ["/usr/bin/colab", "new", "-s", "test-session", "--tpu", "v6e1"],
        check=True,
        text=True,
        capture_output=False,
    )

    runner.stop_session()
    mock_run.assert_called_with(
        ["/usr/bin/colab", "stop", "-s", "test-session"],
        check=False,
        text=True,
        capture_output=False,
    )

  @patch("subprocess.run")
  def test_gpu_provisioning(self, mock_run: MagicMock) -> None:
    runner = colab_runner.ColabRunner(
        session_name="gpu-session",
        accelerator="A100",
        binary_path="/usr/bin/colab",
    )
    runner.provision_session()
    mock_run.assert_called_with(
        ["/usr/bin/colab", "new", "-s", "gpu-session", "--gpu", "A100"],
        check=True,
        text=True,
        capture_output=False,
    )

  @patch("subprocess.run")
  def test_context_manager(self, mock_run: MagicMock) -> None:
    with colab_runner.ColabRunner(
        session_name="ctx-session",
        accelerator="v6e1",
        binary_path="/usr/bin/colab",
    ) as runner:
      self.assertTrue(runner._is_active)
      runner.upload_file("local.txt")
      runner.exec_script("train.py")
      runner.download_file("remote.txt", "local_out.txt")

    self.assertFalse(runner._is_active)
    # Total of 5 subprocess calls executed in sequence:
    # 1. new (provision session)
    # 2. upload_file
    # 3. exec_script
    # 4. download_file
    # 5. stop (cleanup session on context exit)
    self.assertEqual(mock_run.call_count, 5)



if __name__ == "__main__":
  unittest.main()
