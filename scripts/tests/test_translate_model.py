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
"""Tests the model translator script."""

import os
import sys
import unittest

# module hack
LIB_PATH = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, os.path.abspath(LIB_PATH))

from scripts import translate_model  # type: ignore # noqa (module hack)


class TestTranslate(unittest.TestCase):

  def test_icu(self) -> None:
    model = {'a': {'x': 12, 'y': 88}, 'b': {'x': 47, 'z': 13}}
    expect = '''
jaml {
    aKeys {
        "x",
        "y",
    }
    aValues:intvector {
        12,
        88,
    }
    bKeys {
        "x",
        "z",
    }
    bValues:intvector {
        47,
        13,
    }
}
'''.strip()
    result = translate_model.translate_icu(model)
    self.assertEqual(result, expect)
