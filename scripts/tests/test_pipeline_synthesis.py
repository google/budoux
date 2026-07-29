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
"""Test suite for data synthesis meta-pipeline orchestrator."""

import os
import sys
import tempfile
import unittest
from typing import Any
from unittest.mock import MagicMock, patch

# Module hack to allow importing scripts and budoux from workspace root
LIB_PATH = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, os.path.abspath(LIB_PATH))

import pytest

pytest.importorskip("google.genai")

import budoux  # noqa: E402
from scripts import pipeline_synthesis  # noqa: E402

SEP = budoux.utils.SEP


class TestHumanSampleReviewGate(unittest.TestCase):

  def test_human_sample_review_gate_auto_accept(self) -> None:
    lines = [f"いよいよ{SEP}はじまる決戦だ。"]
    result = pipeline_synthesis.human_sample_review_gate(
        lines, auto_accept=True)
    self.assertEqual(result, lines)

  def test_human_sample_review_gate_manual_accept(self) -> None:
    lines = [f"いよいよ{SEP}はじまる決戦だ。"]
    mock_input = MagicMock(return_value="y")
    result = pipeline_synthesis.human_sample_review_gate(
        lines, auto_accept=False, input_func=mock_input)
    self.assertEqual(result, lines)

  def test_human_sample_review_gate_manual_reject(self) -> None:
    lines = [f"いよいよ{SEP}はじまる決戦だ。"]
    mock_input = MagicMock(return_value="n")
    result = pipeline_synthesis.human_sample_review_gate(
        lines, auto_accept=False, input_func=mock_input)
    self.assertEqual(result, [])


class TestRegisterSyntheticDataset(unittest.TestCase):

  def setUp(self) -> None:
    self.temp_dir_obj = tempfile.TemporaryDirectory()
    self.test_dir = self.temp_dir_obj.name

  def tearDown(self) -> None:
    self.temp_dir_obj.cleanup()

  @patch("os.path.dirname")
  def test_register_synthetic_dataset(self, mock_dirname: Any) -> None:
    mock_dirname.return_value = self.test_dir
    lines = [f"いよいよ{SEP}はじまる決戦だ。"]
    issue_path = pipeline_synthesis.register_synthetic_dataset(
        lang="ja",
        issue_id="468",
        approved_lines=lines,
    )
    self.assertTrue(os.path.exists(issue_path))


class TestAppendToQualitySuiteAndHistory(unittest.TestCase):

  def setUp(self) -> None:
    self.temp_dir_obj = tempfile.TemporaryDirectory()
    self.test_dir = self.temp_dir_obj.name
    self.quality_path = os.path.join(self.test_dir, "ja.tsv")
    self.history_path = os.path.join(self.test_dir, "history.txt")
    with open(self.quality_path, "w", encoding="utf-8") as f:
      f.write("init\tテストです。\n")

  def tearDown(self) -> None:
    self.temp_dir_obj.cleanup()

  def test_append_to_quality_suite(self) -> None:
    sample = f"いよいよ{SEP}はじまる決戦だ。"
    pipeline_synthesis.append_to_quality_suite(
        lang="ja",
        issue_id="468",
        sample_line=sample,
        quality_path=self.quality_path,
    )
    with open(self.quality_path, "r", encoding="utf-8") as f:
      content = f.read()
    self.assertIn("gh468", content)

  def test_append_to_history(self) -> None:
    samples = [f"いよいよ{SEP}はじまる決戦だ。"]
    pipeline_synthesis.append_to_history(
        lang="ja",
        approved_lines=samples,
        history_path=self.history_path,
    )
    with open(self.history_path, "r", encoding="utf-8") as f:
      content = f.read()
    self.assertIn("いよいよ", content)


class TestRunSynthesisPipeline(unittest.TestCase):

  def setUp(self) -> None:
    self.temp_dir_obj = tempfile.TemporaryDirectory()
    self.test_dir = self.temp_dir_obj.name

  def tearDown(self) -> None:
    self.temp_dir_obj.cleanup()

  @patch("scripts.synthesize_samples.run_agentic_synthesis_pipeline")
  @patch("scripts.pipeline_synthesis.register_synthetic_dataset")
  def test_run_synthesis_pipeline_end_to_end(self, mock_register: Any,
                                             mock_synth: Any) -> None:
    raw_sentence = f"いよいよ{SEP}はじまる決戦だ。"
    mock_synth.return_value = [raw_sentence]
    mock_register.return_value = os.path.join(self.test_dir, "issue_468.txt")

    quality_path = os.path.join(self.test_dir, "ja.tsv")
    history_path = os.path.join(self.test_dir, "history.txt")
    with open(quality_path, "w", encoding="utf-8") as f:
      f.write("init\tテスト\n")

    res = pipeline_synthesis.run_synthesis_pipeline(
        inputs=["いよいよ/はじまる"],
        lang="ja",
        auto_accept=True,
        quality_path=quality_path,
        history_path=history_path,
    )
    self.assertEqual(res["status"], "completed")
    self.assertEqual(res["approved_lines_count"], 1)


if __name__ == "__main__":
  unittest.main()
