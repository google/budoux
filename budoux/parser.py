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
"""BudouX parser to provide semantic chunks."""

import json
import os
import typing
from html.parser import HTMLParser
from .feature_extractor import get_feature
from .utils import Result, SEP

MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')
PARENT_CSS_STYLE = 'word-break: keep-all; overflow-wrap: break-word;'
with open(os.path.join(os.path.dirname(__file__), 'skip_nodes.json')) as f:
  SKIP_NODES: typing.Set[str] = set(json.load(f))

HTMLAttr = typing.List[typing.Tuple[str, typing.Union[str, None]]]


class TextContentExtractor(HTMLParser):
  """An HTML parser to extract text content.

  Attributes:
    output (str): Accumulated text content.
  """
  output = ''

  def handle_data(self, data):
    self.output += data


class HTMLChunkResolver(HTMLParser):
  """An HTML parser to resolve the given HTML string and semantic chunks.

  Attributes:
    output (str): The HTML string to output.
    chunks (str): Chunks concatenated with the separator string.
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

  def handle_starttag(self, tag: str, attrs: HTMLAttr):
    attr_pairs = []
    for attr in attrs:
      if attr[1] is None:
        attr_pairs.append(attr[0])
      else:
        attr_pairs.append('%s="%s"' % (attr[0], attr[1]))
    encoded_attrs = ' '.join(attr_pairs)
    self.output += '<%s %s>' % (tag, encoded_attrs)
    self.to_skip = tag.upper() in SKIP_NODES

  def handle_endtag(self, tag: str):
    self.output += '</%s>' % (tag)
    self.to_skip = False

  def handle_data(self, data: str):
    if self.to_skip:
      self.output += data
      if self.chunks_joined[0] == SEP:
        self.chunks_joined = self.chunks_joined[1 + len(data):]
      else:
        self.chunks_joined = self.chunks_joined[len(data):]
      return
    for char in data:
      if char == self.chunks_joined[0]:
        self.chunks_joined = self.chunks_joined[1:]
        self.output += char
      else:
        self.chunks_joined = self.chunks_joined[2:]
        self.output += '<wbr>' + char


class Parser:
  """BudouX's Parser.

  The main parser object with a variety of class methods to provide semantic
  chunks and markups from the given input string.

  Attributes:
    model: A dict mapping a feature (str) and its score (int).
  """

  def __init__(self, model: typing.Dict[str, int]):
    """Initializes the parser.

    Args:
      model (Dict[str, int]): A dict mapping a feature and its score.
    """
    self.model = model

  def parse(self, sentence: str, thres: int = 1000):
    """Parses the input sentence and returns a list of semantic chunks.

    Args:
      sentence (str): An input sentence.
      thres (int, optional): A score to control the granularity of chunks.

    Returns:
      A list of semantic chunks (List[str]).
    """
    if sentence == '':
      return []
    p1 = Result.UNKNOWN.value
    p2 = Result.UNKNOWN.value
    p3 = Result.UNKNOWN.value
    chunks = [sentence[:3]]
    for i in range(3, len(sentence)):
      feature = get_feature(sentence[i - 3], sentence[i - 2], sentence[i - 1],
                            sentence[i],
                            sentence[i + 1] if i + 1 < len(sentence) else '',
                            sentence[i + 2] if i + 2 < len(sentence) else '',
                            p1, p2, p3)
      score = 0
      for f in feature:
        if not f in self.model:
          continue
        score += self.model[f]
      if score > thres:
        chunks.append(sentence[i])
      else:
        chunks[-1] += sentence[i]
      p = Result.POSITIVE.value if score > 0 else Result.NEGATIVE.value
      p1 = p2
      p2 = p3
      p3 = p
    return chunks

  def translate_html_string(self, html: str, thres: int = 1000):
    """Translates the given HTML string with markups for semantic line breaks.

    Args:
      html (str): An input html string.
      threshold (int, optional): A score to control the granularity of chunks.

    Returns:
      The translated HTML string (str).
    """
    # TODO: Align with the JavaScript API regarding the parent element addition.
    text_content_extractor = TextContentExtractor()
    text_content_extractor.feed(html)
    text_content = text_content_extractor.output
    chunks = self.parse(text_content, thres)
    resolver = HTMLChunkResolver(chunks)
    resolver.feed(html)
    return '<span style="%s">%s</span>' % (PARENT_CSS_STYLE, resolver.output)


def load_default_japanese_parser():
  """Loads a parser equipped with the default Japanese model.

  Returns:
    A parser (:obj:`budoux.Parser`).
  """
  with open(os.path.join(MODEL_DIR, 'ja-knbc.json')) as f:
    model = json.load(f)
  return Parser(model)