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
"Tests the HTML Processor."

import os
import sys
import unittest

# module hack
LIB_PATH = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, os.path.abspath(LIB_PATH))

from budoux import html_processor  # noqa (module hack)


class TestTextContentExtractor(unittest.TestCase):

  def test_output(self) -> None:
    input = '<p><a href="#">Hello</a>, <b>World</b></p>'
    expected = 'Hello, World'
    extractor = html_processor.TextContentExtractor()
    extractor.feed(input)
    self.assertEqual(
        extractor.output, expected,
        'Text content should be extacted from the given HTML string.')


class TestHTMLChunkResolver(unittest.TestCase):

  def test_output(self) -> None:
    input = '<p>ab<b>cde</b>f</p>'
    expected = '<p>ab<b>c<wbr>de</b>f</p>'
    resolver = html_processor.HTMLChunkResolver(['abc', 'def'], '<wbr>')
    resolver.feed(input)
    self.assertEqual(resolver.output, expected,
                     'WBR tags should be inserted as specified by chunks.')

  def test_unpaired(self) -> None:
    input = '<p>abcdef</p></p>'
    expected = '<p>abc<wbr>def</p></p>'
    resolver = html_processor.HTMLChunkResolver(['abc', 'def'], '<wbr>')
    resolver.feed(input)
    self.assertEqual(resolver.output, expected,
                     'Unpaired close tag should not cause errors.')

  def test_nobr(self) -> None:
    input = '<p>ab<nobr>cde</nobr>f</p>'
    expected = '<p>ab<nobr>cde</nobr>f</p>'
    resolver = html_processor.HTMLChunkResolver(['abc', 'def'], '<wbr>')
    resolver.feed(input)
    self.assertEqual(resolver.output, expected,
                     'WBR tags should not be inserted if in NOBR.')

  def test_after_nobr(self) -> None:
    input = '<p>ab<nobr>xy</nobr>abcdef</p>'
    expected = '<p>ab<nobr>xy</nobr>abc<wbr>def</p>'
    resolver = html_processor.HTMLChunkResolver(['abxyabc', 'def'], '<wbr>')
    resolver.feed(input)
    self.assertEqual(resolver.output, expected,
                     'WBR tags should be inserted if after NOBR.')

  def test_img_in_nobr(self) -> None:
    input = '<p>ab<nobr>x<img>y</nobr>abcdef</p>'
    expected = '<p>ab<nobr>x<img>y</nobr>abc<wbr>def</p>'
    resolver = html_processor.HTMLChunkResolver(['abxyabc', 'def'], '<wbr>')
    resolver.feed(input)
    self.assertEqual(resolver.output, expected,
                     'IMG should not affect surrounding NOBR.')


class TestResolve(unittest.TestCase):

  def test_with_simple_text_input(self) -> None:
    chunks = ['abc', 'def']
    html = 'abcdef'
    result = html_processor.resolve(chunks, html)
    expected = '<span style="word-break: keep-all; overflow-wrap: anywhere;">abc\u200bdef</span>'
    self.assertEqual(result, expected)

  def test_with_standard_html_input(self) -> None:
    chunks = ['abc', 'def']
    html = 'ab<a href="http://example.com">cd</a>ef'
    result = html_processor.resolve(chunks, html)
    expected = '<span style="word-break: keep-all; overflow-wrap: anywhere;">ab<a href="http://example.com">c\u200bd</a>ef</span>'
    self.assertEqual(result, expected)

  def test_with_nodes_to_skip(self) -> None:
    chunks = ['abc', 'def', 'ghi']
    html = "a<button>bcde</button>fghi"
    result = html_processor.resolve(chunks, html)
    expected = '<span style="word-break: keep-all; overflow-wrap: anywhere;">a<button>bcde</button>f\u200bghi</span>'
    self.assertEqual(result, expected)

  def test_with_break_before_skip(self) -> None:
    chunks = ['abc', 'def', 'ghi', 'jkl']
    html = "abc<button>defghi</button>jkl"
    result = html_processor.resolve(chunks, html)
    expected = '<span style="word-break: keep-all; overflow-wrap: anywhere;">abc\u200b<button>defghi</button>\u200bjkl</span>'
    self.assertEqual(result, expected)

  def test_with_nothing_to_split(self) -> None:
    chunks = ['abcdef']
    html = 'abcdef'
    result = html_processor.resolve(chunks, html)
    expected = '<span style="word-break: keep-all; overflow-wrap: anywhere;">abcdef</span>'
    self.assertEqual(result, expected)
