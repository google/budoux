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
"""Tests the model build script."""

import os
import sys
import unittest

# module hack
LIB_PATH = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, os.path.abspath(LIB_PATH))

from scripts import build_model  # noqa (module hack)


class TestAggregateScores(unittest.TestCase):

  def test_standard(self) -> None:
    weights = [
        'AB:x\t2.893\n', 'BC:y\t0.123\n', 'AB:y\t2.123\n', 'BC:y\t1.234\n'
    ]
    model = build_model.aggregate_scores(weights)
    self.assertDictEqual(model, {
        'AB': {
            'x': 2.893,
            'y': 2.123
        },
        'BC': {
            'y': 1.357
        }
    }, 'should group scores by feature type.')

  def test_blank_line(self) -> None:
    weights = [
        '\n', 'AB:x\t2.893\n', 'BC:y\t0.123\n', '\n', 'AB:y\t2.123\n',
        'BC:y\t1.234\n'
    ]
    model = build_model.aggregate_scores(weights)
    self.assertDictEqual(model, {
        'AB': {
            'x': 2.893,
            'y': 2.123
        },
        'BC': {
            'y': 1.357
        }
    }, 'should skip blank lines.')


class TestRoundModel(unittest.TestCase):

  def test_standard(self) -> None:
    model = {
        'AB': {
            'x': 1.0002,
            'y': 4.1237,
        },
        'BC': {
            'z': 2.1111,
        }
    }
    model_rounded = build_model.round_model(model, 1000)
    self.assertDictEqual(model_rounded, {
        'AB': {
            'x': 1000,
            'y': 4123
        },
        'BC': {
            'z': 2111
        }
    }, 'should scale and round scores to integer.')

  def test_insignificant_score(self) -> None:
    model = {
        'AB': {
            'x': 0.0009,
            'y': 4.1237,
        },
        'BC': {
            'z': 2.1111,
        }
    }
    model_rounded = build_model.round_model(model, 1000)
    self.assertDictEqual(model_rounded, {
        'AB': {
            'y': 4123
        },
        'BC': {
            'z': 2111
        }
    }, 'should remove insignificant scores lower than 1.')
