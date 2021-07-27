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
"""Encodes the training data with extracted features."""

import argparse
from context import feature_extractor


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('source_data',
    help='''File path of the source training data to extract features.''')
  parser.add_argument('-o', '--outfile',
    help='''Output file path for the encoded training data.
            (default: encoded_data.txt)''',
    default='encoded_data.txt')
  args = parser.parse_args()
  source_filename = args.source_data
  train_data_filename = args.outfile
  feature_extractor.process(source_filename, train_data_filename)
  print('\033[92mEncoded training data is output to: %s\033[0m' % (
        train_data_filename))


if __name__ == '__main__':
  main()
