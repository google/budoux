# Copyright 2024 Google LLC
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
"Tests the HTML Processor consistency across ports."

import json
import os
import sys
import unittest
from typing import List

# module hack
LIB_PATH = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, os.path.abspath(LIB_PATH))

from budoux import html_processor  # noqa (module hack)
from budoux import parser  # noqa (module hack)

class MockParser:
  def __init__(self, chunks: List[str]):
    self.chunks = chunks
  def parse(self, text: str) -> List[str]:
    return self.chunks

class TestHTMLProcessorConsistency(unittest.TestCase):

  def test_shared_cases(self) -> None:
    shared_json_path = os.path.join(os.path.dirname(__file__), 'html_processor_shared_results.json')
    with open(shared_json_path, encoding='utf-8') as f:
      test_cases = json.load(f)

    for case in test_cases:
      with self.subTest(description=case['description']):
        html = case['html']
        expected_inner = case['expected']
        
        # We need to simulate the segmentation results that JS got.
        # JS uses the real Japanese parser, but for these tests we can just
        # provide the chunks that would result in the expected behavior.
        # Actually, let's just use the real parser for "Simple text" etc.
        # But to be safe and port-independent, we should probably mock the parser
        # if the test case specifies chunks. 
        # Our current JSON doesn't specify chunks, it's based on the real Japanese parser.
        
        # Let's extract the "text content" as the Python port currently does.
        text = html_processor.get_text(html)
        # We'll use a real parser for Japanese as the JS port does.
        from budoux import load_default_japanese_parser
        jp_parser = load_default_japanese_parser()
        chunks = jp_parser.parse(text)
        
        # Use resolver directly to get inner HTML
        resolver = html_processor.HTMLChunkResolver(chunks, '\u200b')
        resolver.feed(html)
        self.assertEqual(resolver.output, expected_inner, 
                         f"Failed consistency case: {case['description']}\nInput: {html}\nText: {text}\nChunks: {chunks}")

if __name__ == '__main__':
  unittest.main()
