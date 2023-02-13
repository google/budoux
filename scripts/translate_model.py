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
"""Translates a model JSON file to another format, such as ICU Resource Bundle."""

import argparse
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


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
      'model', help='File path for the JSON format model file.', type=str)
  args = parser.parse_args()
  model_path: str = args.model
  with open(model_path) as f:
    model = json.load(f)
  print(translate_icu(model))


if __name__ == '__main__':
  main()
