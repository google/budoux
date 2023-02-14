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
"""Translates a model JSON file to another format, such as ICU Resource Bundle.

Example usage:

$ python translate_model.py --format=icu model.json > icurb.txt

You can also use this script to update the model files older than v0.5.0 to make
it work with the latest version.

$ python translate_model.py --format=json old-model.json > new-model.json
"""

import argparse
import itertools
import json
import typing

ArgList = typing.Optional[typing.List[str]]


def translate_icu(model: typing.Dict[str, typing.Dict[str, int]]) -> str:
  """Translates a model to the ICU Resource Bundle format.

  The output is intended to update the data in:
  https://github.com/unicode-org/icu/blob/main/icu4c/source/data/brkitr/adaboost/jaml.txt

  Args:
    model: A model.
  Returns:
    A model string formatted in the ICU Resource Bundle format.
  """
  indent = '    '
  output = 'jaml {\n'
  for group_name, members in model.items():
    output += f'{indent}{group_name}Keys {{\n'
    for key in members.keys():
      output += f'{indent}{indent}"{key}",\n'
    output += f'{indent}}}\n'
    output += f'{indent}{group_name}Values:intvector {{\n'
    for val in members.values():
      output += f'{indent}{indent}{val},\n'
    output += f'{indent}}}\n'
  output += '}'
  return output


def normalize(
    model: typing.Dict[str,
                       typing.Any]) -> typing.Dict[str, typing.Dict[str, int]]:
  """Updates a model to the latest format. Does nothing if it's updated already.

  Args:
    model: A model.
  Returns:
    An updated model.
  """
  is_old_format = all([isinstance(v, int) for v in model.values()])
  if is_old_format:
    output = {}
    sorted_items = sorted(model.items(), key=lambda x: x[0])
    groups = itertools.groupby(sorted_items, key=lambda x: x[0].split(':')[0])
    for group in groups:
      output[group[0]] = dict(
          (item[0].split(':')[-1], item[1]) for item in group[1])
    return output
  try:
    assert (all([
        isinstance(v, int)
        for groups in model.values()
        for v in groups.values()
    ])), 'Scores should be integers'
  except (AssertionError, AttributeError) as e:
    raise Exception('Unsupported model format:', e)
  else:
    return model


def main() -> None:
  DEFAULT_FORMAT = 'json'
  parser = argparse.ArgumentParser(
      description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
  parser.add_argument(
      'model', help='File path for the JSON format model file.', type=str)
  parser.add_argument(
      '--format',
      help=f'Target format (default: {DEFAULT_FORMAT})',
      type=str,
      default=DEFAULT_FORMAT,
      choices={DEFAULT_FORMAT, 'icu'})
  args = parser.parse_args()
  model_path: str = args.model
  format: str = args.format
  with open(model_path) as f:
    model = json.load(f)
  model = normalize(model)
  if format == 'json':
    print(json.dumps(model, ensure_ascii=False, separators=(',', ':')))
  elif format == 'icu':
    print(translate_icu(model))
  else:
    pass


if __name__ == '__main__':
  main()
