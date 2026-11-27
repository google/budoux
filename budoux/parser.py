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
import unicodedata

from .html_processor import HTML_SPEC, translate

MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')
INVISIBLE_FORMAT_CHARS: set[str] = set(HTML_SPEC['invisible_format_chars'])


# Characters outside General_Category M that UAX #29 still keeps in the cluster
# before them: ZWNJ (Extend), Thai and Lao SARA AM (SpacingMark) and the
# halfwidth katakana voiced sound marks (Extend).
_EXTEND_NOT_MARK = frozenset('\u200c\u0e33\u0eb3\uff9e\uff9f')


def _continues_cluster(chars: list[str], offsets: list[int], i: int) -> bool:
  """Checks whether chars[i] continues the grapheme cluster before it.

  A boundary before such a character would split a combining mark (such as a
  decomposed dakuten U+3099), a variation selector, an emoji modifier or tag,
  a ZWJ sequence, a regional indicator pair or CR LF.

  Args:
    chars (list[str]): The visible characters of a sentence.
    offsets (list[int]): The offset of each character in the sentence.
    i (int): An index into chars, greater than 0.

  Returns:
    True if no boundary may be placed before chars[i].
  """
  if offsets[i - 1] + 1 != offsets[i]:
    # An invisible format character in between already ends the cluster.
    return False
  ch = chars[i]
  if ch == '\u200d' or chars[i - 1] == '\u200d':
    return True
  if '\U0001f1e6' <= ch <= '\U0001f1ff':
    # Only the second regional indicator of an adjacent pair continues it.
    j = i - 1
    while (
      j >= 0
      and '\U0001f1e6' <= chars[j] <= '\U0001f1ff'
      and offsets[j] + 1 == offsets[j + 1]
    ):
      j -= 1
    return (i - j) % 2 == 0
  # Below U+0300 only LF continues a cluster, after CR.
  if ch < '\u0300':
    return ch == '\n' and chars[i - 1] == '\r'
  # No combining mark lies among U+4E00-9FFF ideographs.
  if '\u4e00' <= ch <= '\u9fff':
    return False
  if (
    '\U0001f3fb' <= ch <= '\U0001f3ff'
    or '\U000e0020' <= ch <= '\U000e007f'
    or ch in _EXTEND_NOT_MARK
  ):
    return True
  return unicodedata.category(ch)[0] == 'M'


class Parser:
  """BudouX's Parser.

  The main parser object with a variety of class methods to provide semantic
  chunks and markups from the given input string.

  Attributes:
    _model: A dict mapping a feature (str) and its score (int).
  """

  def __init__(self, model: dict[str, dict[str, int]]):
    """Initializes the parser.

    Args:
      model (Dict[str, Dict[str, int]]): A dict mapping a feature and its score.
    """
    self._model: dict[str, dict[str, int]] = {k: dict(v) for k, v in model.items()}
    self._total_score: int = sum(sum(g.values()) for g in self._model.values())

  @property
  def model(self) -> dict[str, dict[str, int]]:
    return self._model

  def parse_boundaries(self, sentence: str) -> list[int]:
    """Parses the input sentence and returns boundary character offsets.

    Invisible format characters (such as U+200B ZERO WIDTH SPACE) are treated as
    transparent in the feature window so repeated segmentation stays idempotent.
    No boundary is placed inside a grapheme cluster, such as before a combining
    mark or a variation selector.

    Args:
      sentence (str): An input sentence.

    Returns:
      A list of character offsets where phrase boundaries occur.
    """
    if sentence == '':
      return []

    uw1 = self._model.get('UW1', {})
    uw2 = self._model.get('UW2', {})
    uw3 = self._model.get('UW3', {})
    uw4 = self._model.get('UW4', {})
    uw5 = self._model.get('UW5', {})
    uw6 = self._model.get('UW6', {})
    bw1 = self._model.get('BW1', {})
    bw2 = self._model.get('BW2', {})
    bw3 = self._model.get('BW3', {})
    tw1 = self._model.get('TW1', {})
    tw2 = self._model.get('TW2', {})
    tw3 = self._model.get('TW3', {})
    tw4 = self._model.get('TW4', {})

    chars: list[str] = []
    offsets: list[int] = []
    for idx, ch in enumerate(sentence):
      if ch in INVISIBLE_FORMAT_CHARS:
        continue
      chars.append(ch)
      offsets.append(idx)

    boundaries: list[int] = []
    n = len(chars)
    for i in range(1, n):
      score = -self._total_score
      if i > 2:
        score += 2 * uw1.get(chars[i - 3], 0)
        score += 2 * tw1.get(chars[i - 3] + chars[i - 2] + chars[i - 1], 0)
      if i > 1:
        score += 2 * uw2.get(chars[i - 2], 0)
        score += 2 * bw1.get(chars[i - 2] + chars[i - 1], 0)
        score += 2 * tw2.get(chars[i - 2] + chars[i - 1] + chars[i], 0)
      score += 2 * uw3.get(chars[i - 1], 0)
      score += 2 * uw4.get(chars[i], 0)
      score += 2 * bw2.get(chars[i - 1] + chars[i], 0)
      if i + 1 < n:
        score += 2 * uw5.get(chars[i + 1], 0)
        score += 2 * bw3.get(chars[i] + chars[i + 1], 0)
        score += 2 * tw3.get(chars[i - 1] + chars[i] + chars[i + 1], 0)
      if i + 2 < n:
        score += 2 * uw6.get(chars[i + 2], 0)
        score += 2 * tw4.get(chars[i] + chars[i + 1] + chars[i + 2], 0)

      if score > 0 and not _continues_cluster(chars, offsets, i):
        boundaries.append(offsets[i])
    return boundaries

  def parse(self, sentence: str) -> list[str]:
    """Parses the input sentence and returns a list of semantic chunks.

    Args:
      sentence (str): An input sentence.

    Returns:
      A list of semantic chunks (List[str]).
    """
    if sentence == '':
      return []
    boundaries = self.parse_boundaries(sentence)
    chunks: list[str] = []
    start = 0
    for boundary in boundaries:
      chunks.append(sentence[start:boundary])
      start = boundary
    chunks.append(sentence[start:])
    return chunks

  def translate_html_string(self, html: str, wrap: bool = True) -> str:
    """Translates the given HTML string with markups for semantic line breaks.

    Args:
      html (str): An input html string.
      wrap (bool, optional): Whether to wrap the output in a span element with
        inline styles. Defaults to True.

    Returns:
      The translated HTML string (str).
    """
    return translate(self, html, wrap=wrap)


def load_default_japanese_parser() -> Parser:
  """Loads a parser equipped with the default Japanese model.

  Returns:
    A parser (:obj:`budoux.Parser`).
  """
  with open(os.path.join(MODEL_DIR, 'ja.json'), encoding='utf-8') as f:
    model = json.load(f)
  return Parser(model)


def load_default_simplified_chinese_parser() -> Parser:
  """Loads a parser equipped with the default Simplified Chinese model.

  Returns:
    A parser (:obj:`budoux.Parser`).
  """
  with open(os.path.join(MODEL_DIR, 'zh-hans.json'), encoding='utf-8') as f:
    model = json.load(f)
  return Parser(model)


def load_default_traditional_chinese_parser() -> Parser:
  """Loads a parser equipped with the default Traditional Chinese model.

  Returns:
    A parser (:obj:`budoux.Parser`).
  """
  with open(os.path.join(MODEL_DIR, 'zh-hant.json'), encoding='utf-8') as f:
    model = json.load(f)
  return Parser(model)


def load_default_thai_parser() -> Parser:
  """Loads a parser equipped with the default Thai model.

  Returns:
    A parser (:obj:`budoux.Parser`).
  """
  with open(os.path.join(MODEL_DIR, 'th.json'), encoding='utf-8') as f:
    model = json.load(f)
  return Parser(model)
