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
"""Tests the finetune script."""

import os
import sys
import unittest
import tempfile
from collections import OrderedDict
from jax import numpy as jnp

# module hack
LIB_PATH = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, os.path.abspath(LIB_PATH))

from scripts import finetune  # noqa (module hack)


class TestArgParse(unittest.TestCase):

  def test_cmdargs_invalid_option(self) -> None:
    cmdargs = ['-v']
    with self.assertRaises(SystemExit) as cm:
      finetune.parse_args(cmdargs)
    self.assertEqual(cm.exception.code, 2)

  def test_cmdargs_help(self) -> None:
    cmdargs = ['-h']
    with self.assertRaises(SystemExit) as cm:
      finetune.parse_args(cmdargs)
    self.assertEqual(cm.exception.code, 0)

  def test_cmdargs_no_data(self) -> None:
    with self.assertRaises(SystemExit) as cm:
      finetune.parse_args([])
    self.assertEqual(cm.exception.code, 2)

  def test_cmdargs_no_base_model(self) -> None:
    with self.assertRaises(SystemExit) as cm:
      finetune.parse_args(['encoded.txt'])
    self.assertEqual(cm.exception.code, 2)

  def test_cmdargs_default(self) -> None:
    cmdargs = ['encoded.txt', 'model.json']
    output = finetune.parse_args(cmdargs)
    self.assertEqual(output.train_data, 'encoded.txt')
    self.assertEqual(output.base_model, 'model.json')


class TestLoadModel(unittest.TestCase):

  def setUp(self) -> None:
    self.model_file_path = tempfile.NamedTemporaryFile().name
    with open(self.model_file_path, 'w') as f:
      f.write('{"UW1": {"a": 12, "b": 23}, "TW3": {"xyz": 47}}')

  def test_extracted_keys(self) -> None:
    result = finetune.load_model(self.model_file_path).keys
    self.assertListEqual(result, ['UW1:a', 'UW1:b', 'TW3:xyz'])

  def test_value_variance(self) -> None:
    result = finetune.load_model(self.model_file_path).weights.var()
    self.assertAlmostEqual(float(result), 1, places=5)

  def test_value_mean(self) -> None:
    result = finetune.load_model(self.model_file_path).weights.sum()
    self.assertAlmostEqual(float(result), 0, places=5)

  def test_value_order(self) -> None:
    result = finetune.load_model(self.model_file_path).weights.tolist()
    self.assertGreater(result[1], result[0])
    self.assertGreater(result[2], result[1])


class TestLoadDataset(unittest.TestCase):

  def setUp(self) -> None:
    self.entries_file_path = tempfile.NamedTemporaryFile().name
    with open(self.entries_file_path, 'w') as f:
      f.write(('1\tfoo\tbar\n'
               '-1\tfoo\n'
               '1\tfoo\tbar\tbaz\n'
               '1\tbar\tfoo\n'
               '-1\tbaz\tqux\n'))
    self.model = finetune.NormalizedModel(['foo', 'bar'], jnp.array([23, -37]))

  def test_y(self) -> None:
    result = finetune.load_dataset(self.entries_file_path, self.model)
    expected = [True, False, True, True, False]
    self.assertListEqual(result.Y.tolist(), expected)

  def test_x(self) -> None:
    result = finetune.load_dataset(self.entries_file_path, self.model)
    expected = [[1, 1], [1, -1], [1, 1], [1, 1], [-1, -1]]
    self.assertListEqual(result.X.tolist(), expected)

class TestFit(unittest.TestCase):
  def test_health(self) -> None:
    w = jnp.array([.9, .5, 0])
    X = jnp.array([[-1, 1, 1]])
    # The current result is x.dot(w) = -.4 < 0 => i.e. False.
    # This tests if the method can learn a new weight that inverses the result.
    Y = jnp.array([True])
    dataset = finetune.Dataset(X, Y)
    w = finetune.fit(w, dataset)
    self.assertGreater(X.dot(w).tolist()[0], 0)  # x.dot(w) > 0 => True.
