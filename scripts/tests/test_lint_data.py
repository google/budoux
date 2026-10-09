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
"""Tests for the dataset linter script."""

import os
import sys
import tempfile
import unittest

LIB_PATH = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, os.path.abspath(LIB_PATH))

from scripts import lint_data


class TestLintData(unittest.TestCase):
  def setUp(self) -> None:
    self.temp_dir = tempfile.TemporaryDirectory()

  def tearDown(self) -> None:
    self.temp_dir.cleanup()

  def _create_file(self, filename: str, content: str) -> str:
    path = os.path.join(self.temp_dir.name, filename)
    with open(path, "w", encoding="utf-8") as f:
      f.write(content)
    return path

  def test_clean_data(self) -> None:
    fpath = self._create_file(
      "clean.txt", "今日▁は▁良い▁天気▁です。\n明日▁も▁晴れる▁でしょう。\n"
    )
    errors = lint_data.lint_dataset([fpath])
    self.assertEqual(errors, [])

  def test_consecutive_separators(self) -> None:
    fpath = self._create_file("bad_sep.txt", "今日▁▁は天気です。\n")
    errors = lint_data.lint_dataset([fpath])
    self.assertTrue(any("Consecutive separator" in e for e in errors))

  def test_leading_or_trailing_separator(self) -> None:
    fpath1 = self._create_file("leading.txt", "▁今日は天気です。\n")
    errors1 = lint_data.lint_dataset([fpath1])
    self.assertTrue(any("Leading or trailing" in e for e in errors1))

    fpath2 = self._create_file("trailing.txt", "今日は天気です。▁\n")
    errors2 = lint_data.lint_dataset([fpath2])
    self.assertTrue(any("Leading or trailing" in e for e in errors2))

  def test_invalid_marker_character(self) -> None:
    fpath = self._create_file("invalid_char.txt", "今日▔は天気です。\n")
    errors = lint_data.lint_dataset([fpath])
    self.assertTrue(any("Invalid marker character" in e for e in errors))

  def test_duplicate_lines(self) -> None:
    fpath = self._create_file("dup.txt", "今日▁は▁晴れ。\n今日▁は▁晴れ。\n")
    errors = lint_data.lint_dataset([fpath])
    self.assertTrue(any("Duplicate sentence" in e for e in errors))

  def test_feature_contradiction(self) -> None:
    # Identical 6-character context with conflicting break
    fpath = self._create_file(
      "conflict.txt", "該当する▁方のみ▁入場できます。\n該当する方▁のみ▁入場できます。\n"
    )
    errors = lint_data.lint_dataset([fpath])
    self.assertTrue(any("Contradicting boundary decision" in e for e in errors))
    self.assertTrue(any("Positive (break" in e for e in errors))
    self.assertTrue(any("Negative (no break" in e for e in errors))

  def test_format_only_skips_contradictions(self) -> None:
    fpath = self._create_file(
      "conflict_skip.txt",
      "該当する▁方のみ▁入場できます。\n該当する方▁のみ▁入場できます。\n",
    )
    errors = lint_data.lint_dataset([fpath], check_conflicts=False)
    self.assertEqual(errors, [])

  def test_main_cli_success(self) -> None:
    fpath = self._create_file("cli_clean.txt", "これ▁は▁テスト▁です。\n")
    exit_code = lint_data.main([fpath])
    self.assertEqual(exit_code, 0)

  def test_main_cli_failure(self) -> None:
    fpath = self._create_file("cli_fail.txt", "これ▁▁はエラー。\n")
    exit_code = lint_data.main([fpath])
    self.assertEqual(exit_code, 1)

  def test_main_cli_nonexistent_path(self) -> None:
    with self.assertRaises(FileNotFoundError):
      lint_data.main(["/nonexistent/path/for/sure"])


if __name__ == "__main__":
  unittest.main()
