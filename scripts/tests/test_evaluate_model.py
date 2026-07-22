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
"""Tests the model evaluation script."""

import json
import os
import sys
import tempfile
import unittest

# module hack to allow importing scripts and budoux from workspace root
LIB_PATH = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, os.path.abspath(LIB_PATH))

import budoux  # noqa: E402
from scripts import evaluate_model  # noqa (module hack)


class TestEvaluateModel(unittest.TestCase):

  def setUp(self) -> None:
    self.temp_dir = tempfile.TemporaryDirectory()
    self.model_path = os.path.join(self.temp_dir.name, 'model.json')
    self.test_data_path = os.path.join(self.temp_dir.name, 'test_data.txt')

    # A simple model that has UW4:A with positive weight (making it break at A)
    # and all other scores 0.
    # Base score: -1000 * 0.5 = -500.
    # At 'A': -500 + 1000 = 500 > 0 (Break).
    # At 'B': -500 + 0 = -500 <= 0 (No Break).
    self.tiny_model = {'UW4': {'A': 1000}}
    with open(self.model_path, 'w', encoding='utf-8') as f:
      json.dump(self.tiny_model, f)

  def tearDown(self) -> None:
    self.temp_dir.cleanup()

  def test_evaluate_model_happy_path(self) -> None:
    # Sentence: BABA
    # Ground truth splits: B▁A▁B▁A (breaks before index 1, 2, 3)
    # Ground truth breaks: [True, True, True]
    # Predictions: B, AB, A (breaks before index 1, 3)
    # Predicted breaks:  [True, False, True]
    # TP: 2 (indices 1, 3)
    # TN: 0
    # FP: 0
    # FN: 1 (index 2)
    # Accuracy: (2+0)/3 = 0.6666666666666666
    # Precision: 2/(2+0) = 1.0
    # Recall: 2/(2+1) = 0.6666666666666666
    # Fscore: 2 * 1.0 * 0.666... / (1.0 + 0.666...) = 0.8
    test_content = f'B{budoux.utils.SEP}A{budoux.utils.SEP}B{budoux.utils.SEP}A\n'
    with open(self.test_data_path, 'w', encoding='utf-8') as f:
      f.write(test_content)

    metrics = evaluate_model.evaluate(self.model_path, self.test_data_path)

    self.assertAlmostEqual(metrics['accuracy'], 2 / 3)
    self.assertAlmostEqual(metrics['precision'], 1.0)
    self.assertAlmostEqual(metrics['recall'], 2 / 3)
    self.assertAlmostEqual(metrics['fscore'], 0.8)

  def test_evaluate_model_literal_slashes(self) -> None:
    # Verify that literal slashes are treated strictly as text characters
    # Sentence: B/A▁B/A
    # Ground truth splits: B/A▁B/A (breaks before index 3)
    # Ground truth breaks: [False, False, True, False, False]
    # Predictions: B/, AB/, A (breaks before index 2, 5 since UW4:A is 1000)
    # Predicted breaks: [False, True, False, False, True]
    # TP: 0
    # TN: 2 (indices 0, 3)
    # FP: 2 (indices 1, 4)
    # FN: 1 (index 2)
    # Accuracy: (0+2)/5 = 0.4
    # Precision: 0/2 = 0.0
    # Recall: 0/1 = 0.0
    # Fscore: 0.0
    test_content = f'B/A{budoux.utils.SEP}B/A\n'
    with open(self.test_data_path, 'w', encoding='utf-8') as f:
      f.write(test_content)

    metrics = evaluate_model.evaluate(self.model_path, self.test_data_path)

    self.assertAlmostEqual(metrics['accuracy'], 0.4)
    self.assertAlmostEqual(metrics['precision'], 0.0)
    self.assertAlmostEqual(metrics['recall'], 0.0)
    self.assertAlmostEqual(metrics['fscore'], 0.0)

  def test_evaluate_model_edge_cases(self) -> None:
    # Verify empty lines are skipped and single characters do not crash
    test_content = '\n\nB\n\nB▁A\n'
    # 'B': length 1 -> 0 transition boundaries (skipped or 0 counts)
    # 'B▁A': length 2 -> 1 transition boundary (index 1).
    # Ground truth: [True]
    # Predictions: B, A (breaks at 'A' since UW4:A is 1000).
    # Predicted breaks: [True]
    # TP: 1, TN: 0, FP: 0, FN: 0
    # Accuracy: 1.0, Precision: 1.0, Recall: 1.0, Fscore: 1.0
    with open(self.test_data_path, 'w', encoding='utf-8') as f:
      f.write(test_content)

    metrics = evaluate_model.evaluate(self.model_path, self.test_data_path)

    self.assertAlmostEqual(metrics['accuracy'], 1.0)
    self.assertAlmostEqual(metrics['precision'], 1.0)
    self.assertAlmostEqual(metrics['recall'], 1.0)
    self.assertAlmostEqual(metrics['fscore'], 1.0)

  def test_evaluate_model_tsv_format(self) -> None:
    tsv_path = os.path.join(self.temp_dir.name, 'test_data.tsv')
    test_content = (
        '# comment line\n'
        f'gh123\tB{budoux.utils.SEP}A{budoux.utils.SEP}B{budoux.utils.SEP}A\n'
        f'gh124\tmeta_info\tB{budoux.utils.SEP}A{budoux.utils.SEP}B{budoux.utils.SEP}A\n'
    )
    with open(tsv_path, 'w', encoding='utf-8') as f:
      f.write(test_content)

    metrics = evaluate_model.evaluate(self.model_path, tsv_path)
    self.assertAlmostEqual(metrics['accuracy'], 2 / 3)
    self.assertAlmostEqual(metrics['precision'], 1.0)
    self.assertAlmostEqual(metrics['recall'], 2 / 3)
    self.assertAlmostEqual(metrics['fscore'], 0.8)


if __name__ == '__main__':
  unittest.main()
