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
"""Tests the prepare Unicode script."""

import os
import tempfile
import unittest

from scripts import prepare_unicode


class TestPrepareUnicode(unittest.TestCase):
  def test_convert_line_raw_word_mode(self) -> None:
    line = "アリス|は|、|姉|の"
    self.assertEqual(
      prepare_unicode.convert_line(line, line_break=False), "アリス▁は▁、▁姉▁の"
    )
    self.assertEqual(prepare_unicode.convert_line("", line_break=False), "")
    self.assertEqual(
      prepare_unicode.convert_line("|アリス||は|", line_break=False), "アリス▁は"
    )

  def test_convert_line_line_break_mode_punctuation(self) -> None:
    line = "古|腾|堡|计划|电子书|《|爱丽丝|梦游|仙境|》|，|“|你好|！|”"
    self.assertEqual(
      prepare_unicode.convert_line(line, line_break=True),
      "古▁腾▁堡▁计划▁电子书▁《爱丽丝▁梦游▁仙境》，▁“你好！”",
    )

  def test_convert_line_line_break_mode_spaces_and_compound_punct(self) -> None:
    line = "费用|（|包括|法律|费用|）|：（|a|）|分发| |Project| |Gutenberg|™| |作品|。"
    self.assertEqual(
      prepare_unicode.convert_line(line, line_break=True),
      "费用▁（包括▁法律▁费用）：▁（a）▁分发 ▁Project ▁Gutenberg™ ▁作品。",
    )
    # Spurious spaces between CJK punctuation and CJK tokens should be stripped.
    cjk_space_line = "難道|永遠|墜|不|下去|了|嗎|？| |「|不|知道|？|」"
    self.assertEqual(
      prepare_unicode.convert_line(cjk_space_line, line_break=True),
      "難道▁永遠▁墜▁不▁下去▁了▁嗎？▁「不▁知道？」",
    )

  def test_convert_line_strips_zero_width_space(self) -> None:
    line = "罕|\u200b\u200b|见|的|连贯|性|。"
    self.assertEqual(
      prepare_unicode.convert_line(line, line_break=True), "罕▁见▁的▁连贯▁性。"
    )

  def test_attach_punctuation_edge_cases(self) -> None:
    self.assertEqual(prepare_unicode.attach_punctuation([]), [])
    self.assertEqual(prepare_unicode.attach_punctuation([" ", "\u200b"]), [])
    self.assertEqual(prepare_unicode.attach_punctuation(["《", "“"]), ["《“"])
    self.assertEqual(prepare_unicode.attach_punctuation(["。", "”"]), ["。”"])

  def test_process_unicode_splits_and_flavor_mapping(self) -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
      corpus_dir = os.path.join(tmp_dir, "test-corpora")
      bp_dir = os.path.join(
        corpus_dir, "gutenberg", "book1", "out", "google", "breakpoints", "ja"
      )
      os.makedirs(bp_dir, exist_ok=True)
      sample_file = os.path.join(bp_dir, "sample.txt")
      with open(sample_file, "w", encoding="utf-8") as f:
        f.writelines(f"これ|は|第|{i}|番目|の|テスト|文|です|。\n" for i in range(10))
      wrap_file = os.path.join(bp_dir, "sample_wrap.txt")
      with open(wrap_file, "w", encoding="utf-8") as f:
        f.write("無視|さ|れる|行\n")

      outfile = os.path.join(tmp_dir, "unicode_ja_word.txt")
      split_dir = os.path.join(tmp_dir, "splits")

      prepare_unicode.process_unicode(
        lang="ja_word",
        outfile=outfile,
        split_dir=split_dir,
        val_ratio=0.2,
        test_ratio=0.2,
        seed=42,
        corpus_dir=corpus_dir,
      )

      self.assertTrue(os.path.exists(outfile))
      with open(outfile, encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
      self.assertEqual(len(lines), 10)
      self.assertEqual(lines[0], "これ▁は▁第▁0▁番目▁の▁テスト▁文▁です▁。")

      train_path = os.path.join(split_dir, "unicode_ja_word_train.txt")
      val_path = os.path.join(split_dir, "unicode_ja_word_val.txt")
      test_path = os.path.join(split_dir, "unicode_ja_word_test.txt")
      self.assertTrue(os.path.exists(train_path))
      self.assertTrue(os.path.exists(val_path))
      self.assertTrue(os.path.exists(test_path))

      with open(train_path, encoding="utf-8") as f:
        self.assertEqual(len(f.readlines()), 6)
      with open(val_path, encoding="utf-8") as f:
        self.assertEqual(len(f.readlines()), 2)
      with open(test_path, encoding="utf-8") as f:
        self.assertEqual(len(f.readlines()), 2)

  def test_process_unicode_splits_and_lang_mapping(self) -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
      corpus_dir = os.path.join(tmp_dir, "test-corpora")
      bp_dir = os.path.join(
        corpus_dir, "gutenberg", "book1", "out", "google", "breakpoints", "zh-Hant"
      )
      os.makedirs(bp_dir, exist_ok=True)
      sample_file = os.path.join(bp_dir, "sample.txt")
      with open(sample_file, "w", encoding="utf-8") as f:
        f.writelines(f"這|是|第|{i}|個|測試|句子|。\n" for i in range(10))

      outfile = os.path.join(tmp_dir, "unicode_zh-hant.txt")
      split_dir = os.path.join(tmp_dir, "splits")

      prepare_unicode.process_unicode(
        lang="zh-hant",
        outfile=outfile,
        line_break=True,
        split_dir=split_dir,
        val_ratio=0.2,
        test_ratio=0.2,
        seed=42,
        corpus_dir=corpus_dir,
      )

      self.assertTrue(os.path.exists(outfile))
      with open(outfile, encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
      self.assertEqual(len(lines), 10)
      self.assertEqual(lines[0], "這▁是▁第▁0▁個▁測試▁句子。")

      train_path = os.path.join(split_dir, "unicode_zh-hant_train.txt")
      val_path = os.path.join(split_dir, "unicode_zh-hant_val.txt")
      test_path = os.path.join(split_dir, "unicode_zh-hant_test.txt")
      self.assertTrue(os.path.exists(train_path))
      self.assertTrue(os.path.exists(val_path))
      self.assertTrue(os.path.exists(test_path))

      with open(train_path, encoding="utf-8") as f:
        self.assertEqual(len(f.readlines()), 6)
      with open(val_path, encoding="utf-8") as f:
        self.assertEqual(len(f.readlines()), 2)
      with open(test_path, encoding="utf-8") as f:
        self.assertEqual(len(f.readlines()), 2)

  def test_process_unicode_invalid_split_ratios(self) -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
      corpus_dir = os.path.join(tmp_dir, "test-corpora")
      bp_dir = os.path.join(
        corpus_dir, "gutenberg", "book1", "out", "google", "breakpoints", "ja"
      )
      os.makedirs(bp_dir, exist_ok=True)
      with open(os.path.join(bp_dir, "sample.txt"), "w", encoding="utf-8") as f:
        f.write("テスト|文\n")

      with self.assertRaises(ValueError):
        prepare_unicode.process_unicode(
          lang="ja_word",
          outfile=os.path.join(tmp_dir, "out.txt"),
          split_dir=os.path.join(tmp_dir, "splits"),
          val_ratio=0.5,
          test_ratio=0.5,
          corpus_dir=corpus_dir,
        )

  def test_process_unicode_missing_files_exits(self) -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
      corpus_dir = os.path.join(tmp_dir, "test-corpora")
      os.makedirs(corpus_dir, exist_ok=True)
      with self.assertRaises(SystemExit):
        prepare_unicode.process_unicode(
          lang="ja_word",
          outfile=os.path.join(tmp_dir, "out.txt"),
          corpus_dir=corpus_dir,
        )
