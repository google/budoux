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
"""Test suite for agentic training data synthesis."""

import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

# Module hack to allow importing scripts and budoux from workspace root
LIB_PATH = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, os.path.abspath(LIB_PATH))

import budoux  # noqa: E402
from scripts import synthesize_samples  # noqa: E402

SEP = budoux.utils.SEP  # Canonical character separator '▁'


class TestAlignToBaseParserSplits(unittest.TestCase):

  def test_align_to_base_parser_splits_negative_phrase(self):
    target = synthesize_samples.IntentTarget(
        target_phrase="もはや",
        expected_split="もはや",
        is_positive=False,
        is_reproducible=True)
    mock_parser = MagicMock(spec=budoux.Parser)
    mock_parser.parse.return_value = [
        "彼の", "才能は", "もは", "や", "誰にも", "超えられない。"
    ]

    cands = [f"彼の{SEP}才能は{SEP}もはや{SEP}誰にも超えられない。"]
    result = synthesize_samples.align_to_base_parser_splits(
        cands, target, mock_parser)

    self.assertEqual(len(result), 1)
    expected_sentence = f"彼の{SEP}才能は{SEP}もはや{SEP}誰にも{SEP}超えられない。"
    self.assertEqual(result[0], expected_sentence)

  def test_align_to_base_parser_splits_positive_phrase(self):
    target = synthesize_samples.IntentTarget(
        target_phrase="いよいよはじまる",
        expected_split=f"いよいよ{SEP}はじまる",
        is_positive=True,
        is_reproducible=True,
    )
    mock_parser = MagicMock(spec=budoux.Parser)
    mock_parser.parse.return_value = ["いよいよは", "じまる", "決戦だ。"]

    cands = ["いよいよ▁はじまる決戦だ。"]
    result = synthesize_samples.align_to_base_parser_splits(
        cands, target, mock_parser)

    self.assertEqual(len(result), 1)
    expected_sentence = f"いよいよ{SEP}はじまる{SEP}決戦だ。"
    self.assertEqual(result[0], expected_sentence)

  def test_align_to_base_parser_splits_discards_missing_target(self):
    target = synthesize_samples.IntentTarget(
        target_phrase="もはや",
        expected_split="もはや",
        is_positive=False,
        is_reproducible=True)
    mock_parser = MagicMock(spec=budoux.Parser)
    cands = ["関係のない完全に異なる文章です。"]
    result = synthesize_samples.align_to_base_parser_splits(
        cands, target, mock_parser)
    self.assertEqual(len(result), 0)


class TestParseDirectInput(unittest.TestCase):

  def test_parse_direct_input_positive(self):
    target = synthesize_samples.parse_direct_input("いよいよ/はじまる")
    self.assertEqual(target.target_phrase, "いよいよはじまる")
    self.assertEqual(target.expected_split, f"いよいよ{SEP}はじまる")
    self.assertTrue(target.is_positive)
    self.assertTrue(target.is_reproducible)

  def test_parse_direct_input_negative(self):
    target = synthesize_samples.parse_direct_input("もはや")
    self.assertEqual(target.target_phrase, "もはや")
    self.assertEqual(target.expected_split, "もはや")
    self.assertFalse(target.is_positive)
    self.assertTrue(target.is_reproducible)


class TestVerifyBugReproduction(unittest.TestCase):

  def test_verify_bug_reproduction_detects_bug(self):
    target = synthesize_samples.IntentTarget(
        target_phrase="いよいよはじまる",
        expected_split=f"いよいよ{SEP}はじまる",
        is_positive=True,
    )
    mock_parser = MagicMock(spec=budoux.Parser)
    mock_parser.parse.return_value = ["いよいよは", "じまる"]

    self.assertTrue(
        synthesize_samples.verify_bug_reproduction(target, mock_parser))
    mock_parser.parse.assert_called_once_with("いよいよはじまる")

  def test_verify_bug_reproduction_no_bug(self):
    target = synthesize_samples.IntentTarget(
        target_phrase="もはや", expected_split="もはや", is_positive=False)
    mock_parser = MagicMock(spec=budoux.Parser)
    mock_parser.parse.return_value = ["もはや"]
    self.assertFalse(
        synthesize_samples.verify_bug_reproduction(target, mock_parser))


class TestRunAgenticSynthesisPipeline(unittest.TestCase):

  def test_run_pipeline_raises_without_env_or_client(self):
    mock_parser = MagicMock(spec=budoux.Parser)
    with patch.dict(os.environ, {"GEMINI_API_KEY": ""}, clear=True):
      with self.assertRaises(RuntimeError):
        synthesize_samples.run_agentic_synthesis_pipeline(
            input_str="いよいよ/はじまる", client=None, parser=mock_parser)

  @patch("scripts.synthesize_samples.generate_oversample_candidates")
  @patch("scripts.synthesize_samples.prune_linguistic_anomalies")
  def test_run_agentic_synthesis_pipeline_end_to_end(self, mock_prune,
                                                     mock_generate):
    mock_client = MagicMock()
    mock_parser = MagicMock(spec=budoux.Parser)
    mock_parser.parse.return_value = ["いよいよ", "はじまる", "決戦だ。"]

    raw_sentence = f"いよいよ{SEP}はじまる決戦だ。"
    mock_generate.return_value = [raw_sentence]
    expected_overlay = f"いよいよ{SEP}はじまる{SEP}決戦だ。"
    mock_prune.return_value = [expected_overlay]

    outfile = os.path.join(self.test_dir, "staging_ja.txt")
    lines = synthesize_samples.run_agentic_synthesis_pipeline(
        input_str="いよいよ/はじまる",
        num_candidates=5,
        max_keep=3,
        lang="ja",
        outfile=outfile,
        client=mock_client,
        parser=mock_parser,
    )
    self.assertEqual(lines, [expected_overlay])
    self.assertTrue(os.path.exists(outfile))

  def setUp(self):
    self.temp_dir_obj = tempfile.TemporaryDirectory()
    self.test_dir = self.temp_dir_obj.name

  def tearDown(self):
    self.temp_dir_obj.cleanup()


if __name__ == "__main__":
  unittest.main()
