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
"""Tests the prepare Wisesight script."""

import os
import tempfile
import unittest

from scripts import prepare_wisesight


class TestPrepareWisesight(unittest.TestCase):
  def test_preprocess_line_standard(self) -> None:
    line = 'วัน|นี้|อากาศ|ดี\n'
    self.assertEqual(prepare_wisesight.preprocess_line(line), 'วัน▁นี้▁อากาศ▁ดี')

  def test_preprocess_line_removes_urls_hashtags_and_emojis(self) -> None:
    line = (
      '✨|อยาก|ไป|ดู| |#|BIFW2018| |ค่ะ| |'
      'https://example.com/foo| |แพง|เว่อร์| |😰|😰|😰์| |❗️|พอ\n'
    )
    self.assertEqual(
      prepare_wisesight.preprocess_line(line), 'อยาก▁ไป▁ดู▁ ▁ค่ะ▁ ▁แพง▁เว่อร์▁ ▁พอ'
    )

  def test_preprocess_line_strips_zero_width_space_and_redundant_borders(self) -> None:
    line = '|ซื้อ|รองพื้น|\u200bคู่|กันแดด| |ค่ะ| #|The|Face|'
    self.assertEqual(prepare_wisesight.preprocess_line(line), 'ซื้อ▁รองพื้น▁คู่▁กันแดด▁ ▁ค่ะ')

  def test_process_wisesight_splits_and_filtering(self) -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
      source_file = os.path.join(tmp_dir, 'sample.label')
      with open(source_file, 'w', encoding='utf-8') as f:
        f.writelines(f'ประโยค|ที่|{i}|สำหรับ|ทดสอบ\n' for i in range(10))
        # Single-chunk or emoji-only lines should be skipped.
        f.write('อร่อย\n')
        f.write('😋|😋|😋\n')

      outfile = os.path.join(tmp_dir, 'source_wisesight.txt')
      split_dir = os.path.join(tmp_dir, 'splits')

      prepare_wisesight.process_wisesight(
        source_filepath=source_file,
        outfile=outfile,
        split_dir=split_dir,
        val_ratio=0.2,
        test_ratio=0.2,
        seed=42,
      )

      self.assertTrue(os.path.exists(outfile))
      with open(outfile, encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
      self.assertEqual(len(lines), 10)
      self.assertEqual(lines[0], 'ประโยค▁ที่▁0▁สำหรับ▁ทดสอบ')

      train_path = os.path.join(split_dir, 'wisesight_train.txt')
      val_path = os.path.join(split_dir, 'wisesight_val.txt')
      test_path = os.path.join(split_dir, 'wisesight_test.txt')
      self.assertTrue(os.path.exists(train_path))
      self.assertTrue(os.path.exists(val_path))
      self.assertTrue(os.path.exists(test_path))

      with open(train_path, encoding='utf-8') as f:
        self.assertEqual(len(f.readlines()), 6)
      with open(val_path, encoding='utf-8') as f:
        self.assertEqual(len(f.readlines()), 2)
      with open(test_path, encoding='utf-8') as f:
        self.assertEqual(len(f.readlines()), 2)

  def test_process_wisesight_invalid_split_ratios(self) -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
      source_file = os.path.join(tmp_dir, 'sample.label')
      with open(source_file, 'w', encoding='utf-8') as f:
        f.write('วัน|นี้|อากาศ|ดี\n')

      with self.assertRaises(ValueError):
        prepare_wisesight.process_wisesight(
          source_filepath=source_file,
          outfile=os.path.join(tmp_dir, 'out.txt'),
          split_dir=os.path.join(tmp_dir, 'splits'),
          val_ratio=0.5,
          test_ratio=0.5,
        )
