# Copyright 2021 Google LLC
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
"""Tests the BudouX　CLI."""

import io
import sys
import unittest
from os.path import abspath, dirname, join

# module hack
LIB_PATH = join(dirname(__file__), '..')
sys.path.insert(0, abspath(LIB_PATH))

from budoux import main  # noqa (module hack)

if isinstance(sys.stdin, io.TextIOWrapper) and sys.version_info >= (3, 7):
  sys.stdin.reconfigure(encoding='utf-8')

if isinstance(sys.stdout, io.TextIOWrapper) and sys.version_info >= (3, 7):
  sys.stdout.reconfigure(encoding='utf-8')


class TestCommonOption(unittest.TestCase):

  def test_cmdargs_invalid_option(self) -> None:
    cmdargs = ['-v']
    with self.assertRaises(SystemExit) as cm:
      main.parse_args(cmdargs)

    self.assertEqual(cm.exception.code, 2)

  def test_cmdargs_help(self) -> None:
    cmdargs = ['-h']
    with self.assertRaises(SystemExit) as cm:
      main.parse_args(cmdargs)

    self.assertEqual(cm.exception.code, 0)

  def test_cmdargs_version(self) -> None:
    cmdargs = ['-V']
    with self.assertRaises(SystemExit) as cm:
      main.parse_args(cmdargs)

    self.assertEqual(cm.exception.code, 0)


class TestTextArguments(unittest.TestCase):

  def test_cmdargs_single_text(self) -> None:
    cmdargs = ['これはテストです。']
    output = main._main(cmdargs)

    self.assertEqual(output, "これは\nテストです。")

  def test_cmdargs_single_multiline_text(self) -> None:
    cmdargs = ["これはテストです。\n今日は晴天です。"]
    output = main._main(cmdargs)

    self.assertEqual(output, "これは\nテストです。\n---\n今日は\n晴天です。")

  def test_cmdargs_single_multiline_text_with_delimiter(self) -> None:
    cmdargs = ["これはテストです。\n今日は晴天です。", "-d", "@"]
    output = main._main(cmdargs)

    self.assertEqual(output, "これは\nテストです。\n@\n今日は\n晴天です。")

  def test_cmdargs_single_multiline_text_with_empty_delimiter(self) -> None:
    cmdargs = ["これはテストです。\n今日は晴天です。", "-d", ""]
    output = main._main(cmdargs)

    self.assertEqual(output, "これは\nテストです。\n\n今日は\n晴天です。")

  def test_cmdargs_multi_text(self) -> None:
    cmdargs = ['これはテストです。', '今日は晴天です。']
    with self.assertRaises(SystemExit) as cm:
      main.main(cmdargs)

    self.assertEqual(cm.exception.code, 2)

  def test_cmdargs_single_html(self) -> None:
    cmdargs = ['-H', '今日は<b>とても天気</b>です。']
    output = main._main(cmdargs)

    self.assertEqual(
        output,
        '<span style="word-break: keep-all; overflow-wrap: break-word;">'
        '今日は<b ><wbr>とても<wbr>天気</b>です。</span>')

  def test_cmdargs_multi_html(self) -> None:
    cmdargs = ['-H', '今日は<b>とても天気</b>です。', 'これは<b>テスト</b>です。']
    with self.assertRaises(SystemExit) as cm:
      main._main(cmdargs)

    self.assertEqual(cm.exception.code, 2)

  def test_cmdargs_thres(self) -> None:
    cmdargs = ['--thres', '0', '今日はとても天気です。']
    output_granular = main._main(cmdargs)
    cmdargs = ['--thres', '10000000', '今日はとても天気です。']
    output_whole = main._main(cmdargs)
    self.assertGreater(
        len(output_granular), len(output_whole),
        'Chunks should be more granular when a smaller threshold value is given.'
    )
    self.assertEqual(
        ''.join(output_granular.split('\n')), ''.join(output_whole.split('\n')),
        'The output sentence should be the same regardless of the threshold value.'
    )


class TestStdin(unittest.TestCase):

  def test_cmdargs_blank_stdin(self) -> None:
    with open(
        join(abspath(dirname(__file__)), "in/1.in"),
        "r",
        encoding=sys.getdefaultencoding()) as f:
      sys.stdin = f
      output = main._main([])

    self.assertEqual(output, "")

  def test_cmdargs_text_stdin(self) -> None:
    with open(
        join(abspath(dirname(__file__)), "in/2.in"),
        "r",
        encoding=sys.getdefaultencoding()) as f:
      sys.stdin = f
      output = main._main([])

    self.assertEqual(output, "これは\nテストです。")

  def test_cmdargs_html_stdin(self) -> None:
    with open(
        join(abspath(dirname(__file__)), "in/3.in"),
        "r",
        encoding=sys.getdefaultencoding()) as f:
      sys.stdin = f
      output = main._main(["-H"])

    self.assertEqual(
        output,
        '<span style="word-break: keep-all; overflow-wrap: break-word;">'
        'これは<b ><wbr>テスト</b>です。<wbr>\n'
        '</span>')


if __name__ == '__main__':
  unittest.main()
