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
"""HTML processor."""

import json
import os
import typing
from html.parser import HTMLParser

from .utils import SEP

HTMLAttr = typing.List[typing.Tuple[str, typing.Union[str, None]]]
PARENT_CSS_STYLE = 'word-break: keep-all; overflow-wrap: break-word;'
with open(
    os.path.join(os.path.dirname(__file__), 'skip_nodes.json'),
    encoding='utf-8') as f:
  SKIP_NODES: typing.Set[str] = set(json.load(f))


class TextContentExtractor(HTMLParser):
  """An HTML parser to extract text content.

  Attributes:
    output (str): Accumulated text content.
  """
  output = ''

  def handle_data(self, data: str) -> None:
    self.output += data


class HTMLChunkResolver(HTMLParser):
  """An HTML parser to resolve the given HTML string and semantic chunks.

  Attributes:
    output (str): The HTML string to output.
  """
  output = ''

  def __init__(self, chunks: typing.List[str]):
    """Initializes the parser.

    Args:
      chunks (List[str]): The chunks to resolve.
    """
    HTMLParser.__init__(self)
    self.chunks_joined = SEP.join(chunks)
    self.to_skip = False
    self.scan_index = 0

  def handle_starttag(self, tag: str, attrs: HTMLAttr) -> None:
    attr_pairs = []
    for attr in attrs:
      if attr[1] is None:
        attr_pairs.append(' ' + attr[0])
      else:
        attr_pairs.append(' %s="%s"' % (attr[0], attr[1]))
    encoded_attrs = ''.join(attr_pairs)
    self.output += '<%s%s>' % (tag, encoded_attrs)
    self.to_skip = tag.upper() in SKIP_NODES

  def handle_endtag(self, tag: str) -> None:
    self.output += '</%s>' % (tag)
    self.to_skip = False

  def handle_data(self, data: str) -> None:
    for char in data:
      if not char == self.chunks_joined[self.scan_index]:
        if not self.to_skip:
          self.output += '<wbr>'
        self.scan_index += 1
      self.output += char
      self.scan_index += 1


def get_text(html: str) -> str:
  """Gets the text content from the input HTML string.

  Args:
    html (str): Input HTML string.

  Returns:
    The text content.
  """
  text_content_extractor = TextContentExtractor()
  text_content_extractor.feed(html)
  return text_content_extractor.output


def resolve(phrases: typing.List[str], html: str) -> str:
  """Wraps phrases in the HTML string with non-breaking markup.

  Args:
    phrases (List[str]): The phrases included in the HTML string.
    html (str): The HTML string to resolve.

  Returns:
    The HTML string with phrases wrapped in non-breaking markup.
  """
  resolver = HTMLChunkResolver(phrases)
  resolver.feed(html)
  result = '<span style="%s">%s</span>' % (PARENT_CSS_STYLE, resolver.output)
  return result
