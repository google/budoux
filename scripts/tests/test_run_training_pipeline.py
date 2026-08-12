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
"""Tests for the training pipeline runner script."""

import json
import os
import sys
import tempfile
import typing
import unittest
from unittest.mock import MagicMock, patch

import pytest

LIB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, LIB_PATH)

from scripts import run_training_pipeline


class TestRunTrainingPipeline(unittest.TestCase):
  @patch("subprocess.run")
  def test_run_retraining_pipeline_subprocess(self, mock_run: typing.Any) -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
      split_dir = os.path.join(tmp_dir, "splits")
      os.makedirs(split_dir, exist_ok=True)
      with open(os.path.join(split_dir, "knbc_train.txt"), "w") as f:
        f.write("test sentence\n")

      out_model = os.path.join(tmp_dir, "model.json")
      run_training_pipeline.run_retraining_pipeline(
        lang="ja", iterations=10, split_dir=split_dir, out_model=out_model
      )

      self.assertGreaterEqual(mock_run.call_count, 4)

  def test_run_retraining_pipeline_generates_valid_json(self) -> None:
    pytest.importorskip("jax")
    with tempfile.TemporaryDirectory() as tmp_dir:
      split_dir = os.path.join(tmp_dir, "splits")
      os.makedirs(split_dir, exist_ok=True)
      with open(os.path.join(split_dir, "knbc_train.txt"), "w", encoding="utf-8") as f:
        f.write("今日▁は▁良い▁天気▁です。\n")
      with open(os.path.join(split_dir, "knbc_val.txt"), "w", encoding="utf-8") as f:
        f.write("明日▁も▁晴れ▁です。\n")

      out_model = os.path.join(tmp_dir, "model.json")
      run_training_pipeline.run_retraining_pipeline(
        lang="ja",
        iterations=5,
        split_dir=split_dir,
        out_model=out_model,
        weight_factor=1,
      )

      self.assertTrue(os.path.exists(out_model))
      with open(out_model, encoding="utf-8") as f:
        model_data = json.load(f)
      self.assertIsInstance(model_data, dict)

  @patch("scripts.colab_runner.ColabRunner")
  @patch("subprocess.run")
  def test_run_retraining_pipeline_colab(
    self, mock_run: MagicMock, mock_colab_runner_cls: MagicMock
  ) -> None:
    mock_runner = MagicMock()
    mock_colab_runner_cls.return_value.__enter__.return_value = mock_runner

    with tempfile.TemporaryDirectory() as tmp_dir:
      split_dir = os.path.join(tmp_dir, "splits")
      os.makedirs(split_dir, exist_ok=True)
      with open(os.path.join(split_dir, "knbc_train.txt"), "w") as f:
        f.write("test sentence\n")
      out_model = os.path.join(tmp_dir, "model.json")

      def fake_download(remote_path: str, local_path: str) -> None:

        if remote_path == "/content/weights.txt":
          with open(local_path, "w") as f:
            f.write("foo\t1.0\n")

      mock_runner.download_file.side_effect = fake_download

      run_training_pipeline.run_retraining_pipeline(
        lang="ja",
        iterations=10,
        split_dir=split_dir,
        out_model=out_model,
        colab=True,
        accelerator="T4",
      )

      mock_colab_runner_cls.assert_called_once_with(
        session_name="budoux-train-T4", accelerator="T4"
      )
      self.assertTrue(mock_runner.exec_script.called)

      uploaded_remote_paths = [
        call[0][1] for call in mock_runner.upload_file.call_args_list
      ]
      self.assertIn("/content/cleaned.txt", uploaded_remote_paths)
      self.assertIn("/content/train.py", uploaded_remote_paths)

  @patch("scripts.colab_runner.ColabRunner")
  @patch("subprocess.run")
  def test_colab_download_failure_does_not_mask_error(
    self, mock_run: MagicMock, mock_colab_runner_cls: MagicMock
  ) -> None:
    mock_runner = MagicMock()
    mock_colab_runner_cls.return_value.__enter__.return_value = mock_runner
    mock_runner.exec_script.side_effect = RuntimeError("Remote execution failed")

    with tempfile.TemporaryDirectory() as tmp_dir:
      split_dir = os.path.join(tmp_dir, "splits")
      os.makedirs(split_dir, exist_ok=True)
      with open(os.path.join(split_dir, "knbc_train.txt"), "w") as f:
        f.write("test sentence\n")

      out_model = os.path.join(tmp_dir, "model.json")
      with self.assertRaises(RuntimeError) as cm:
        run_training_pipeline.run_retraining_pipeline(
          lang="ja", iterations=10, split_dir=split_dir, out_model=out_model, colab=True
        )
      self.assertEqual(str(cm.exception), "Remote execution failed")


if __name__ == "__main__":
  unittest.main()
