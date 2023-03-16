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
    resolver = html_processor.HTMLChunkResolver(['abc', 'def'])
    resolver.feed(input)
    self.assertEqual(resolver.output, expected,
                     'WBR tags should be inserted as specified by chunks.')


class TestResolve(unittest.TestCase):

  def test_with_simple_text_input(self) -> None:
    chunks = ['abc', 'def']
    html = 'abcdef'
    result = html_processor.resolve(chunks, html)
    expected = '<span style="word-break: keep-all; overflow-wrap: break-word;">abc<wbr>def</span>'
    self.assertEqual(result, expected)

  def test_with_standard_html_input(self) -> None:
    chunks = ['abc', 'def']
    html = 'ab<a href="http://example.com">cd</a>ef'
    result = html_processor.resolve(chunks, html)
    expected = '<span style="word-break: keep-all; overflow-wrap: break-word;">ab<a href="http://example.com">c<wbr>d</a>ef</span>'
    self.assertEqual(result, expected)

  def test_with_nodes_to_skip(self) -> None:
    chunks = ['abc', 'def']
    html = "a<button>bcde</button>f"
    result = html_processor.resolve(chunks, html)
    expected = '<span style="word-break: keep-all; overflow-wrap: break-word;">a<button>bcde</button>f</span>'
    self.assertEqual(result, expected)

  def test_with_nothing_to_split(self) -> None:
    chunks = ['abcdef']
    html = 'abcdef'
    result = html_processor.resolve(chunks, html)
    expected = '<span style="word-break: keep-all; overflow-wrap: break-word;">abcdef</span>'
    self.assertEqual(result, expected)
