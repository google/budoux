# Copyright 2023 Google LLC
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
"""Tests the data encoder script."""

import os
import sys
import typing
import unittest

from budoux import utils

# module hack
LIB_PATH = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, os.path.abspath(LIB_PATH))

from scripts import encode_data  # type: ignore # noqa (module hack)


class TestGetFeature(unittest.TestCase):

  def test_standard(self) -> None:
    feature = encode_data.get_feature('a', 'b', 'c', 'd', 'e', 'f')
    self.assertSetEqual(
        set(feature),
        {
            # Unigram of Words (UW)
            'UW1:a',
            'UW2:b',
            'UW3:c',
            'UW4:d',
            'UW5:e',
            'UW6:f',

            # Bigram of Words (BW)
            'BW1:bc',
            'BW2:cd',
            'BW3:de',

            # Trigram of Words (TW)
            'TW1:abc',
            'TW2:bcd',
            'TW3:cde',
            'TW4:def',
        },
        'Features should be extracted.')

  def test_with_invalid(self) -> None:

    def find_by_prefix(prefix: str, feature: typing.List[str]) -> bool:
      for item in feature:
        if item.startswith(prefix):
          return True
      return False

    feature = encode_data.get_feature('a', 'a', encode_data.INVALID, 'a', 'a',
                                      'a')
    self.assertFalse(
        find_by_prefix('UW3:', feature),
        'Should omit the Unigram feature when the character is invalid.')
    self.assertFalse(
        find_by_prefix('BW2:', feature),
        'Should omit the Bigram feature that covers an invalid character.')


class TestArgParse(unittest.TestCase):

  def test_cmdargs_invalid_option(self) -> None:
    cmdargs = ['-v']
    with self.assertRaises(SystemExit) as cm:
      encode_data.parse_args(cmdargs)
    self.assertEqual(cm.exception.code, 2)

  def test_cmdargs_help(self) -> None:
    cmdargs = ['-h']
    with self.assertRaises(SystemExit) as cm:
      encode_data.parse_args(cmdargs)
    self.assertEqual(cm.exception.code, 0)

  def test_cmdargs_no_source(self) -> None:
    with self.assertRaises(SystemExit) as cm:
      encode_data.parse_args([])
    self.assertEqual(cm.exception.code, 2)

  def test_cmdargs_default(self) -> None:
    cmdargs = ['source.txt']
    output = encode_data.parse_args(cmdargs)
    self.assertEqual(output.source_data, 'source.txt')
    self.assertEqual(output.outfile, encode_data.DEFAULT_OUTPUT_FILENAME)
    self.assertIsNone(output.processes)

  def test_cmdargs_with_outfile(self) -> None:
    cmdargs = ['source.txt', '-o', 'out.txt']
    output = encode_data.parse_args(cmdargs)
    self.assertEqual(output.source_data, 'source.txt')
    self.assertEqual(output.outfile, 'out.txt')
    self.assertIsNone(output.processes)

  def test_cmdargs_with_processes(self) -> None:
    cmdargs = ['source.txt', '--processes', '8']
    output = encode_data.parse_args(cmdargs)
    self.assertEqual(output.source_data, 'source.txt')
    self.assertEqual(output.outfile, encode_data.DEFAULT_OUTPUT_FILENAME)
    self.assertEqual(output.processes, 8)


class TestProcess(unittest.TestCase):

  sentence = '六本木ヒルズでお昼を食べる。'
  sep_indices = {7, 10, 13}

  def test_on_positive_point(self) -> None:
    line = encode_data.process(8, self.sentence, self.sep_indices)
    items = line.split('\t')
    positive = items[0]
    features = set(items[1:])
    self.assertEqual(positive, '-1')
    self.assertIn('UW2:で', features)

  def test_on_negative_point(self) -> None:
    line = encode_data.process(7, self.sentence, self.sep_indices)
    items = line.split('\t')
    positive = items[0]
    features = set(items[1:])
    self.assertEqual(positive, '1')
    self.assertIn('UW3:で', features)


class TestNormalizeInput(unittest.TestCase):

  def test_standard_input(self) -> None:
    source = f'ABC{utils.SEP}DE{utils.SEP}FGHI'
    sentence, sep_indices = encode_data.normalize_input(source)
    self.assertEqual(sentence, 'ABCDEFGHI')
    self.assertEqual(sep_indices, {3, 5, 9})

  def test_with_linebreaks(self) -> None:
    source = f'AB\nCDE{utils.SEP}FG'
    sentence, sep_indices = encode_data.normalize_input(source)
    self.assertEqual(sentence, 'ABCDEFG')
    self.assertEqual(sep_indices, {2, 5, 7})

  def test_doubled_seps(self) -> None:
    source = f'ABC{utils.SEP}{utils.SEP}DE\n\nFG'
    sentence, sep_indices = encode_data.normalize_input(source)
    self.assertEqual(sentence, 'ABCDEFG')
    self.assertEqual(sep_indices, {3, 5, 7})


if __name__ == '__main__':
  unittest.main()
