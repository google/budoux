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
from unittest.mock import patch

import pytest

LIB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, LIB_PATH)

from scripts import run_training_pipeline  # noqa: E402


class TestRunTrainingPipeline(unittest.TestCase):

  @patch("subprocess.run")
  def test_run_retraining_pipeline_subprocess(self,
                                              mock_run: typing.Any) -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
      split_dir = os.path.join(tmp_dir, "splits")
      os.makedirs(split_dir, exist_ok=True)
      with open(os.path.join(split_dir, "knbc_train.txt"), "w") as f:
        f.write("test sentence\n")

      out_model = os.path.join(tmp_dir, "model.json")
      run_training_pipeline.run_retraining_pipeline(
          lang="ja", iterations=10, split_dir=split_dir, out_model=out_model)

      self.assertGreaterEqual(mock_run.call_count, 4)

  def test_run_retraining_pipeline_generates_valid_json(self) -> None:
    pytest.importorskip("jax")
    with tempfile.TemporaryDirectory() as tmp_dir:
      split_dir = os.path.join(tmp_dir, "splits")
      os.makedirs(split_dir, exist_ok=True)
      with open(
          os.path.join(split_dir, "knbc_train.txt"), "w",
          encoding="utf-8") as f:
        f.write("今日▁は▁良い▁天気▁です。\n")
      with open(
          os.path.join(split_dir, "knbc_val.txt"), "w", encoding="utf-8") as f:
        f.write("明日▁も▁晴れ▁です。\n")

      out_model = os.path.join(tmp_dir, "model.json")
      run_training_pipeline.run_retraining_pipeline(
          lang="ja",
          iterations=5,
          split_dir=split_dir,
          out_model=out_model,
          weight_factor=1)

      self.assertTrue(os.path.exists(out_model))
      with open(out_model, "r", encoding="utf-8") as f:
        model_data = json.load(f)
      self.assertIsInstance(model_data, dict)


if __name__ == "__main__":
  unittest.main()
