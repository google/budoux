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
"""Tests the training script."""

import math
import os
import sys
import tempfile
import typing
import unittest

import numpy as np
from jax import numpy as jnp

# module hack
LIB_PATH = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, os.path.abspath(LIB_PATH))

from scripts import train  # noqa (module hack)


class TestArgParse(unittest.TestCase):

  def test_cmdargs_invalid_option(self) -> None:
    cmdargs = ['-v']
    with self.assertRaises(SystemExit) as cm:
      train.parse_args(cmdargs)
    self.assertEqual(cm.exception.code, 2)

  def test_cmdargs_help(self) -> None:
    cmdargs = ['-h']
    with self.assertRaises(SystemExit) as cm:
      train.parse_args(cmdargs)
    self.assertEqual(cm.exception.code, 0)

  def test_cmdargs_no_data(self) -> None:
    with self.assertRaises(SystemExit) as cm:
      train.parse_args([])
    self.assertEqual(cm.exception.code, 2)

  def test_cmdargs_default(self) -> None:
    cmdargs = ['encoded.txt']
    output = train.parse_args(cmdargs)
    self.assertEqual(output.encoded_train_data, 'encoded.txt')
    self.assertEqual(output.output, train.DEFAULT_OUTPUT_NAME)
    self.assertEqual(output.log, train.DEFAULT_LOG_NAME)
    self.assertEqual(output.feature_thres, train.DEFAULT_FEATURE_THRES)
    self.assertEqual(output.iter, train.DEFAULT_ITERATION)
    self.assertEqual(output.out_span, train.DEFAULT_OUT_SPAN)
    self.assertEqual(output.val_data, None)

  def test_cmdargs_full(self) -> None:
    cmdargs = [
        'encoded.txt', '-o', 'out.txt', '--log', 'foo.log', '--feature-thres',
        '100', '--iter', '10', '--out-span', '50', '--val-data',
        'val_encoded.txt'
    ]
    output = train.parse_args(cmdargs)
    self.assertEqual(output.encoded_train_data, 'encoded.txt')
    self.assertEqual(output.output, 'out.txt')
    self.assertEqual(output.log, 'foo.log')
    self.assertEqual(output.feature_thres, 100)
    self.assertEqual(output.iter, 10)
    self.assertEqual(output.out_span, 50)
    self.assertEqual(output.val_data, 'val_encoded.txt')


class TestPreprocess(unittest.TestCase):

  def test_standard_setup(self) -> None:
    train_data_path = tempfile.NamedTemporaryFile().name
    val_data_path = tempfile.NamedTemporaryFile().name
    with open(train_data_path, 'w') as f:
      f.write(('1\tfoo\tbar\n'
               '-1\tfoo\n'
               '1\tfoo\tbar\tbaz\n'
               '1\tbar\tfoo\n'
               '-1\tbaz\tqux\n'))
    with open(val_data_path, 'w') as f:
      f.write(('1\tbar\tbaz\n'
               '-1\txyz\n'
               '1\tabc\tqux\tfoo\n'))
    train_dataset, features, val_dataset = train.preprocess(
        train_data_path, 1, val_data_path)

    self.assertEqual(features, ['foo', 'bar', 'baz'])
    # The training input X and the target Y should look like below:
    # Y    X(foo bar baz)
    # 1      1   1   0
    # -1     1   0   0
    # 1      1   1   1
    # 1      1   1   0
    # -1     0   0   1
    self.assertEqual(train_dataset.Y.tolist(), [True, False, True, True, False])
    self.assertEqual(train_dataset.X_rows.tolist(), [0, 0, 1, 2, 2, 2, 3, 3, 4])
    self.assertEqual(train_dataset.X_cols.tolist(), [0, 1, 0, 0, 1, 2, 1, 0, 2])

    # The validation input X and the target Y should look like below:
    # Y    X(foo bar baz)
    # 1      0   1   1
    # -1     0   0   0
    # 1      1   0   0
    self.assertIsInstance(val_dataset, train.Dataset)
    if isinstance(val_dataset, train.Dataset):
      self.assertEqual(val_dataset.Y.tolist(), [True, False, True])
      self.assertEqual(val_dataset.X_rows.tolist(), [0, 0, 2])
      self.assertEqual(val_dataset.X_cols.tolist(), [1, 2, 0])
    else:
      raise ValueError('val_dataset is not an instance of Dataset.')

  def teset_no_val(self) -> None:
    train_data_path = tempfile.NamedTemporaryFile().name
    with open(train_data_path, 'w') as f:
      f.write(('1\tfoo\tbar\n'
               '-1\tfoo\n'
               '1\tfoo\tbar\tbaz\n'
               '1\tbar\tfoo\n'
               '-1\tbaz\tqux\n'))
    train_dataset, features, val_dataset = train.preprocess(
        train_data_path, 1)
    self.assertEqual(features, ['foo', 'bar', 'baz'])
    self.assertEqual(train_dataset.Y.tolist(), [True, False, True, True, False])
    self.assertEqual(train_dataset.X_rows.tolist(), [0, 0, 1, 2, 2, 2, 3, 3, 4])
    self.assertEqual(train_dataset.X_cols.tolist(), [0, 1, 0, 0, 1, 2, 1, 0, 2])
    self.assertIsNone(val_dataset)


class TestPred(unittest.TestCase):

  def test_standard_setup(self) -> None:
    X = np.array([
        [1, 1, 0],
        [1, 0, 1],
        [0, 1, 0],
        [0, 0, 1],
    ])
    phis = np.array([0.4, 0.2, -0.3])
    N = X.shape[0]
    rows, cols = np.where(X == 1)
    res = train.pred(phis, rows, cols, N)
    expected = [
        0.4 + 0.2 - (-0.3) > 0,
        0.4 - 0.2 + (-0.3) > 0,
        -0.4 + 0.2 - (-0.3) > 0,
        -0.4 - 0.2 + (-0.3) > 0,
    ]
    self.assertEqual(res.tolist(), expected)


class TestGetMetrics(unittest.TestCase):

  def test_standard_setup(self) -> None:
    pred = np.array([0, 0, 1, 0, 0], dtype=bool)
    target = np.array([1, 0, 1, 1, 1], dtype=bool)
    result = train.get_metrics(pred, target)
    self.assertEqual(result.tp, 1)
    self.assertEqual(result.tn, 1)
    self.assertEqual(result.fp, 0)
    self.assertEqual(result.fn, 3)
    self.assertEqual(result.accuracy, 2 / 5)
    p = 1 / 1
    r = 1 / 4
    self.assertEqual(result.precision, p)
    self.assertEqual(result.recall, r)
    self.assertEqual(result.fscore, 2 * p * r / (p + r))


class TestUpdate(unittest.TestCase):
  X = np.array([
      [1, 0, 1, 0],
      [0, 1, 0, 0],
      [0, 0, 0, 0],
      [1, 0, 0, 0],
      [0, 1, 1, 0],
  ])

  def test_standard_setup1(self) -> None:
    rows, cols = np.where(self.X == 1)
    M = self.X.shape[-1]
    Y = np.array([1, 1, 0, 0, 1], dtype=bool)
    w = np.array([0.1, 0.3, 0.1, 0.1, 0.4])
    scores = jnp.zeros(M)
    new_w, new_scores, best_feature_index, added_score = train.update(
        w, scores, rows, cols, Y)
    self.assertFalse(w.argmax() == 0)
    self.assertTrue(new_w.argmax() == 0)
    self.assertFalse(scores.argmax() == 1)
    self.assertTrue(new_scores.argmax() == 1)
    self.assertEqual(best_feature_index, 1)
    self.assertTrue(added_score > 0)


class TestFit(unittest.TestCase):

  def test_fit(self) -> None:
    weights_file_path = tempfile.NamedTemporaryFile().name
    log_file_path = tempfile.NamedTemporaryFile().name
    # Prepare a dataset that the 2nd feature (= the 2nd col in X) perfectly
    # correlates with Y in a negative way.
    X = jnp.array([
        [0, 1, 1, 1],
        [1, 1, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 1],
    ])
    Y = jnp.array([0, 0, 1, 1])
    rows, cols = jnp.where(X == 1)
    dataset = train.Dataset(rows, cols, Y)
    features = ['a', 'b', 'c']
    iters = 5
    out_span = 2
    scores = train.fit(dataset, dataset, features, iters, weights_file_path,
                       log_file_path, out_span)
    with open(weights_file_path) as f:
      weights = [
          line.split('\t') for line in f.read().splitlines() if line.strip()
      ]
    top_feature = weights[0][0]
    self.assertEqual(
        top_feature, 'b', msg='The most effective feature should be selected.')
    self.assertEqual(
        len(weights),
        iters,
        msg='The number of lines should equal to the iteration count.')

    with open(log_file_path) as f:
      log = [line.split('\t') for line in f.read().splitlines() if line.strip()]
    self.assertEqual(
        len(log),
        math.ceil(iters / out_span) + 1,
        msg='The number of lines should equal to the ceil of iteration / out_span plus one for the header'
    )
    self.assertEqual(
        len(set(len(line) for line in log)),
        1,
        msg='The header and the body should have the same number of columns.')

    model: typing.Dict[str, float] = {}
    for weight in weights:
      model.setdefault(weight[0], 0)
      model[weight[0]] += float(weight[1])
    self.assertEqual(scores.shape[0], len(features))
    loaded_scores = [model.get(feature, 0) for feature in features]
    self.assertTrue(np.all(np.isclose(scores, loaded_scores)))
    os.remove(weights_file_path)
    os.remove(log_file_path)


class TestExtractFeatures(unittest.TestCase):

  def test_with_standard_setup(self) -> None:
    entries_file_path = tempfile.NamedTemporaryFile().name
    with open(entries_file_path, 'w') as f:
      f.write(('1\tfoo\tbar\n'
               '-1\tfoo\n'
               '1\tfoo\tbar\tbaz\n'
               '1\tbar\tfoo\n'
               '-1\tbaz\tqux\n'))
    result = train.extract_features(entries_file_path, 1)
    self.assertEqual(result, ['foo', 'bar', 'baz'])

  def test_with_redundant_breaks(self) -> None:
    entries_file_path = tempfile.NamedTemporaryFile().name
    with open(entries_file_path, 'w') as f:
      f.write(('1\tfoo\tbar\n'
               '-1\tfoo\n'
               '1\tfoo\tbar\tbaz\n\n'
               '1\tbar\tfoo\n'
               '\n'
               '-1\tbaz\tqux\n'))
    result = train.extract_features(entries_file_path, 1)
    self.assertEqual(result, ['foo', 'bar', 'baz'])


class TestLoadDataset(unittest.TestCase):

  def test_with_standard_setup(self) -> None:
    entries_file_path = tempfile.NamedTemporaryFile().name
    with open(entries_file_path, 'w') as f:
      f.write(('1\tfoo\tbar\n'
               '-1\tfoo\n'
               '1\tfoo\tbar\tbaz\n'
               '1\tbar\tfoo\n'
               '-1\tbaz\tqux\n'))
    result = train.load_dataset(entries_file_path, {
        'foo': 0,
        'bar': 1,
        'baz': 2
    })
    self.assertEqual(result.X_rows.tolist(), [0, 0, 1, 2, 2, 2, 3, 3, 4])
    self.assertEqual(result.X_cols.tolist(), [0, 1, 0, 0, 1, 2, 1, 0, 2])
    self.assertEqual(result.Y.tolist(), [True, False, True, True, False])

  def test_with_redundant_breaks(self) -> None:
    entries_file_path = tempfile.NamedTemporaryFile().name
    with open(entries_file_path, 'w') as f:
      f.write(('1\tfoo\tbar\n'
               '-1\tfoo\n'
               '1\tfoo\tbar\tbaz\n\n'
               '1\tbar\tfoo\n'
               '\n'
               '-1\tbaz\tqux\n'))
    result = train.load_dataset(entries_file_path, {
        'foo': 0,
        'bar': 1,
        'baz': 2
    })
    self.assertEqual(result.X_rows.tolist(), [0, 0, 1, 2, 2, 2, 3, 3, 4])
    self.assertEqual(result.X_cols.tolist(), [0, 1, 0, 0, 1, 2, 1, 0, 2])
    self.assertEqual(result.Y.tolist(), [True, False, True, True, False])


if __name__ == '__main__':
  unittest.main()
