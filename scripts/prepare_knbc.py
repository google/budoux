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
"""Prepares a dataset from the KNBC corpus.

Before running this script, you need to download the KNBC corpus by running:

$ curl -o knbc.tar.bz2 https://nlp.ist.i.kyoto-u.ac.jp/kuntt/KNBC_v1.0_090925_utf8.tar.bz2
$ tar -xf knbc.tar.bz2

Now you should have a directory named `KNBC_v1.0_090925_utf8`.
Run the following to generate a dataset named `source_knbc.txt`.

$ python scripts/prepare_knbc.py KNBC_v1.0_090925_utf8 -o source_knbc.txt
"""

import argparse
import os
import typing
from html.parser import HTMLParser

from budoux import utils


class KNBCHTMLParser(HTMLParser):
  """Parses the HTML files in the KNBC corpus to collect chunks.

  Attributes:
    chunks: The collected chunks.
    row: The current row index.
    col: The current column index.
    current_word: The current word to process.
    on_split_row: Whether the scan is on the splitting row.
    split_tab: Whether to split by tags in addition to Bunsetsu.
  """

  BUNSETSU_SPLIT_ID = 'bnst-kugiri'
  TAG_SPLIT_ID = 'tag-kugiri'

  def __init__(self, split_tab: bool = False) -> None:
    """Initializes the HTML parser for the KNBC corpus.

    Args:
      split_tab: Split by tags in addition to Bunsetsu. (default: False)
    """
    super().__init__()
    self.chunks = ['']
    self.row = 0
    self.col = 0
    self.current_word = ''
    self.on_split_row = False
    self.split_tab = split_tab

  def handle_starttag(
      self, tag: str,
      attributes: typing.List[typing.Tuple[str, typing.Optional[str]]]) -> None:
    if tag == 'tr':
      self.row += 1
      self.col = 0
      self.current_word = ''
      self.on_split_row = False

    if tag == 'td':
      self.col += 1
      for name, value in attributes:
        if (name == 'id' and value == self.BUNSETSU_SPLIT_ID) or (
            self.split_tab and name == 'id' and value == self.TAG_SPLIT_ID):
          self.on_split_row = True

  def handle_endtag(self, tag: str) -> None:
    if tag != 'tr':  # Skip all tags but TR.
      return None
    if self.row < 3:  # Skip the first two rows.
      return None
    if self.on_split_row:
      return self.chunks.append('')
    if self.col == 5:
      self.chunks[-1] += self.current_word

  def handle_data(self, data: str) -> None:
    if self.col == 1:
      self.current_word = data


def break_before_sequence(chunks: typing.List[str],
                          sequence: str) -> typing.List[str]:
  """Breaks chunks before a specified character sequence appears.

  Args:
    chunks (List[str]): Chunks to break.
    sequence (str): A character sequence to break chunks before.

  Returns:
    Processed chunks.
  """
  chunks = utils.SEP.join(chunks).replace(sequence,
                                          utils.SEP + sequence).split(utils.SEP)
  chunks = [chunk for chunk in chunks if len(chunk) > 0]
  return chunks


def postprocess(chunks: typing.List[str]) -> typing.List[str]:
  """Applies some processes to modify the extracted chunks.

  Args:
    chunks (List[str]): Source chunks.

  Returns:
    Processed chunks.
  """
  chunks = break_before_sequence(chunks, '（')
  chunks = break_before_sequence(chunks, 'もら')
  return chunks


def parse_args() -> argparse.Namespace:
  DEFAULT_OUT_PATH = 'source.txt'
  parser = argparse.ArgumentParser(
      description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
  parser.add_argument('source_dir', help='Path to the KNBC corpus directory.')
  parser.add_argument(
      '-o',
      '--outfile',
      help=f'File path to the output dataset. (default: {DEFAULT_OUT_PATH})',
      default=DEFAULT_OUT_PATH)
  return parser.parse_args()


def main() -> None:
  args = parse_args()
  source_dir = args.source_dir
  outfile = args.outfile
  html_dir = os.path.join(source_dir, 'html')
  with open(outfile, 'w') as f:
    for file in sorted(os.listdir(html_dir)):
      if file[-11:] != '-morph.html':
        continue
      parser = KNBCHTMLParser(split_tab=False)
      data = open(os.path.join(html_dir, file)).read()
      parser.feed(data)
      chunks = parser.chunks
      chunks = postprocess(chunks)
      if len(chunks) < 2:
        continue
      f.write(utils.SEP.join(chunks) + '\n')
  print('\033[92mTraining data is output to: %s\033[0m' % (outfile))


if __name__ == '__main__':
  main()
