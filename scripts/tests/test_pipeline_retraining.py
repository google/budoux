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
"""Test suite for deterministic model retraining meta-pipeline orchestrator."""

import glob
import json
import os
import sys
import tempfile
import unittest
from typing import Any, Dict
from unittest.mock import patch

import pytest

# Module hack to allow importing scripts and budoux from workspace root
LIB_PATH = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, os.path.abspath(LIB_PATH))

pytest.importorskip("jax")

import budoux  # noqa: E402
from scripts import pipeline_retraining  # noqa: E402

SEP = budoux.utils.SEP


class TestDeterministicSplitCorpus(unittest.TestCase):

  def setUp(self) -> None:
    self.temp_dir_obj = tempfile.TemporaryDirectory()
    self.test_dir = self.temp_dir_obj.name

  def tearDown(self) -> None:
    self.temp_dir_obj.cleanup()

  def test_deterministic_split_80_10_10(self) -> None:
    src_path = os.path.join(self.test_dir, "src.txt")
    lines = [f"line{i}" for i in range(100)]
    with open(src_path, "w", encoding="utf-8") as f:
      f.write("\n".join(lines) + "\n")

    train_p, val_p, test_p = pipeline_retraining.deterministic_split_corpus(
        source_path=src_path,
        output_dir=self.test_dir,
        lang="ja",
        train_ratio=0.80,
        val_ratio=0.10,
        test_ratio=0.10,
    )

    with open(train_p, "r", encoding="utf-8") as f:
      train_lines = [line_str.strip() for line_str in f if line_str.strip()]
    with open(val_p, "r", encoding="utf-8") as f:
      val_lines = [line_str.strip() for line_str in f if line_str.strip()]
    with open(test_p, "r", encoding="utf-8") as f:
      test_lines = [line_str.strip() for line_str in f if line_str.strip()]

    self.assertEqual(len(train_lines), 80)
    self.assertEqual(len(val_lines), 10)
    self.assertEqual(len(test_lines), 10)
    self.assertEqual(train_lines[0], "line42")
    self.assertEqual(set(train_lines + val_lines + test_lines), set(lines))


class TestMergeAndWeightTrainingSources(unittest.TestCase):

  def setUp(self) -> None:
    self.temp_dir_obj = tempfile.TemporaryDirectory()
    self.test_dir = self.temp_dir_obj.name

  def tearDown(self) -> None:
    self.temp_dir_obj.cleanup()

  @patch("os.path.dirname")
  def test_merge_and_weight_training_sources(self, mock_dirname: Any) -> None:
    mock_dirname.return_value = self.test_dir
    base_train = os.path.join(self.test_dir, "knbc_train.txt")
    with open(base_train, "w", encoding="utf-8") as f:
      f.write(f"今日{SEP}は{SEP}天気\n")

    lang_dir = os.path.join(self.test_dir, "..", "data", "finetuning", "ja")
    os.makedirs(lang_dir, exist_ok=True)
    for p in glob.glob(os.path.join(lang_dir, "*.txt")):
      if os.path.isfile(p):
        os.remove(p)

    curated_path = os.path.join(lang_dir, "history.txt")
    with open(curated_path, "w", encoding="utf-8") as f:
      f.write(f"いよいよ{SEP}決戦\n")

    out_path = os.path.join(self.test_dir, "weighted.txt")
    pipeline_retraining.merge_and_weight_training_sources(
        lang="ja",
        output_path=out_path,
        base_train_path=base_train,
        base_scale=1,
        curated_scale=10,
    )

    with open(out_path, "r", encoding="utf-8") as f:
      lines = [line_str.strip() for line_str in f if line_str.strip()]

    # base 1 line * 1 + curated 1 line * 10 = 11 lines
    self.assertEqual(len(lines), 11)


class TestRunBuildCompactModelStep(unittest.TestCase):

  def setUp(self) -> None:
    self.temp_dir_obj = tempfile.TemporaryDirectory()
    self.test_dir = self.temp_dir_obj.name

  def tearDown(self) -> None:
    self.temp_dir_obj.cleanup()

  def test_run_build_compact_model_step(self) -> None:
    weights_p = os.path.join(self.test_dir, "weights.txt")
    with open(weights_p, "w", encoding="utf-8") as f:
      f.write("UW1:a\t1.5\n")

    out_json = os.path.join(self.test_dir, "model.json")
    model_dict = pipeline_retraining.run_build_compact_model_step(
        weights_path=weights_p, model_json_out_path=out_json, scale=1000)
    self.assertIn("UW1", model_dict)
    with open(out_json, "r", encoding="utf-8") as f:
      raw = f.read()
    # verify compact separators (no whitespace after comma or colon)
    self.assertIn('"UW1":{"a":', raw)


class TestRunEvaluationReport(unittest.TestCase):

  @patch("scripts.evaluate_model.evaluate")
  def test_run_evaluation_report(self, mock_eval: Any) -> None:
    mock_eval.return_value = {"accuracy": 0.99, "fscore": 0.98}
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as m:
      model_p = m.name
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as h:
      holdout_p = h.name

    try:
      reports = pipeline_retraining.run_evaluation_report(
          model_path=model_p, holdout_test_path=holdout_p)
      self.assertIn("holdout", reports)
      self.assertEqual(reports["holdout"]["accuracy"], 0.99)
    finally:
      if os.path.exists(model_p):
        os.remove(model_p)
      if os.path.exists(holdout_p):
        os.remove(holdout_p)


class TestRunRetrainingPipeline(unittest.TestCase):

  def setUp(self) -> None:
    self.temp_dir_obj = tempfile.TemporaryDirectory()
    self.test_dir = self.temp_dir_obj.name

  def tearDown(self) -> None:
    self.temp_dir_obj.cleanup()

  @patch("scripts.pipeline_retraining.ensure_base_datasets")
  @patch("scripts.pipeline_retraining.run_encode_step")
  @patch("scripts.pipeline_retraining.run_find_conflicts_step")
  @patch("scripts.pipeline_retraining.run_train_step")
  @patch("scripts.pipeline_retraining.run_build_compact_model_step")
  @patch("scripts.pipeline_retraining.run_evaluation_report")
  def test_run_retraining_pipeline_end_to_end(
      self,
      mock_eval: Any,
      mock_build: Any,
      mock_train: Any,
      mock_conflicts: Any,
      mock_encode: Any,
      mock_ensure: Any,
  ) -> None:
    train_p = os.path.join(self.test_dir, "train.txt")
    val_p = os.path.join(self.test_dir, "val.txt")
    test_p = os.path.join(self.test_dir, "test.txt")
    for p in (train_p, val_p, test_p):
      with open(p, "w", encoding="utf-8") as f:
        f.write("sample\n")

    mock_ensure.return_value = (train_p, val_p, test_p)

    def side_effect_build(weights_path: str,
                          model_json_out_path: str,
                          scale: int = 1000) -> Dict[str, Any]:
      res = {"UW1": {"a": 1000}}
      with open(model_json_out_path, "w", encoding="utf-8") as f:
        json.dump(res, f)
      return res

    mock_build.side_effect = side_effect_build
    mock_eval.return_value = {
        "holdout": {
            "accuracy": 0.99
        },
        "regression": {
            "accuracy": 1.0
        },
    }

    out_model = os.path.join(self.test_dir, "ja.json")
    res = pipeline_retraining.run_retraining_pipeline(
        lang="ja",
        iterations=10,
        output_model=out_model,
        regression_tsv_path=test_p,
    )

    self.assertEqual(res["status"], "completed")
    self.assertTrue(os.path.exists(out_model))


if __name__ == "__main__":
  unittest.main()
