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
"""Prepares a dataset from the Wisesight corpus.

Before running this script, you need to download the Wisesight corpus by running:

$ curl -o wisesight-1000-samples-tokenised.label https://raw.githubusercontent.com/PyThaiNLP/wisesight-sentiment/master/word-tokenization/wisesight-1000-samples-tokenised.label
$ curl -o wisesight-160-samples-tokenised.label https://raw.githubusercontent.com/PyThaiNLP/wisesight-sentiment/master/word-tokenization/wisesight-160-samples-tokenised.label

Then run this command as follows over each file.

$ python scripts/prepare_wisesight.py wisesight-1000-samples-tokenised.label -o source_train.txt
$ python scripts/prepare_wisesight.py wisesight-160-samples-tokenised.label -o source_val.txt
"""
import argparse
import re

import regex


def parse_args() -> argparse.Namespace:
  DEFAULT_OUT_PATH = 'source.txt'
  parser = argparse.ArgumentParser(
      description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
  parser.add_argument(
      'source_filepath', help='Path to a Wisesight corpus label file.')
  parser.add_argument(
      '-o',
      '--outfile',
      help=f'File path to the output dataset. (default: {DEFAULT_OUT_PATH})',
      default=DEFAULT_OUT_PATH)
  return parser.parse_args()


def main() -> None:
  args = parse_args()
  source_filepath = args.source_filepath
  target_filepath = args.outfile

  with open(target_filepath, 'w') as outfile:
    with open(source_filepath) as infile:
      for line in infile:
        line = line.strip()
        line = re.sub(r'https?://[^ ]+', '', line)  # Remove URLs
        line = re.sub(r'#[^ ]+', '', line)  # Remove hashtags
        line = regex.compile(r'\p{Emoji_Presentation=Yes}+').sub(
            '', line)  # Remove emojis
        line = re.sub(r'\|+', '|', line)  # Remove consecutive separators
        line = re.sub(r'(\|\s)*\|$', '', line)  # Remove redundant spaces
        outfile.write(line.replace('|', '▁') + '\n')  # Replace the separators.
  print('\033[92mTraining data is output to: %s\033[0m' % (target_filepath))


if __name__ == '__main__':
  main()
