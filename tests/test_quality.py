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
"""Quality regression test."""

import os
import sys
import unittest

# module hack
LIB_PATH = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, os.path.abspath(LIB_PATH))

from scripts.evaluate_model import evaluate  # noqa (module hack)


class TestQuality(unittest.TestCase):

  def test_ja(self) -> None:
    model_path = os.path.join(LIB_PATH, 'budoux', 'models', 'ja.json')
    quality_path = os.path.join(os.path.dirname(__file__), 'quality', 'ja.tsv')
    res = evaluate(model_path, quality_path)
    errors = res['errors']
    self.assertEqual(
        len(errors), 0, 'Failing sentences:\n{}'.format('\n'.join(
            [f'expected:{err[0]}\tactual:{err[1]}' for err in errors])))
