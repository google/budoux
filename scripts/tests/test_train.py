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
import unittest
from pathlib import Path

import numpy as np
import numpy.typing as npt

# module hack
LIB_PATH = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, os.path.abspath(LIB_PATH))

from scripts import train  # type: ignore # noqa (module hack)

ENTRIES_FILE_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), 'entries_test.txt'))
WEIGHTS_FILE_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), 'weights_test.txt'))
LOG_FILE_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), 'train_test.log'))


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
    self.assertEqual(output.chunk_size, None)

  def test_cmdargs_full(self) -> None:
    cmdargs = [
        'encoded.txt', '-o', 'out.txt', '--log', 'foo.log', '--feature-thres',
        '100', '--iter', '10', '--chunk-size', '1000', '--out-span', '50'
    ]
    output = train.parse_args(cmdargs)
    self.assertEqual(output.encoded_train_data, 'encoded.txt')
    self.assertEqual(output.output, 'out.txt')
    self.assertEqual(output.log, 'foo.log')
    self.assertEqual(output.feature_thres, 100)
    self.assertEqual(output.iter, 10)
    self.assertEqual(output.chunk_size, 1000)
    self.assertEqual(output.out_span, 50)


class TestTrain(unittest.TestCase):

  def setUp(self) -> None:
    Path(WEIGHTS_FILE_PATH).touch()
    Path(LOG_FILE_PATH).touch()
    with open(ENTRIES_FILE_PATH, 'w') as f:
      f.write((
          ' 1\tA\tC\n'  # the first column represents the label (-1 / 1).
          '-1\tA\tB\n'  # the rest columns represents the associated features.
          ' 1\tA\tC\n'
          '-1\tA\n'
          ' 1\tA\tC\n'))

  def test_pred(self) -> None:
    X: npt.NDArray[np.bool_] = np.array([
        [True, False, True, False],
        [False, True, False, True],
    ])
    phis = {
        1: 8.0,  # Weights Feature #1 by 8.
        2: 2.0,  # Weights Feature #2 by 2.
    }
    # Since Feature #1 (= the 2nd col in X) wins, the prediction should be:
    # [
    #   False,
    #   True,
    # ]
    pred = train.pred(phis, X)
    self.assertListEqual(pred.tolist(), [False, True])

  def test_preprocess(self) -> None:
    freq_thres = 0
    X, Y, features = train.preprocess(ENTRIES_FILE_PATH, freq_thres)
    self.assertListEqual(features, ['A', 'C', 'B'],
                         'Features should be ordered by frequency.')

    self.assertListEqual(
        X.tolist(),
        [
            # A    C     B      BIAS
            [True, True, False, True],
            [True, False, True, True],
            [True, True, False, True],
            [True, False, False, True],
            [True, True, False, True],
        ],
        'X should represent the entry features with a bias column.')

    self.assertListEqual(Y.tolist(), [
        True,
        False,
        True,
        False,
        True,
    ], 'Y should represent the entry labels.')

    freq_thres = 4
    X, Y, features = train.preprocess(ENTRIES_FILE_PATH, freq_thres)
    self.assertListEqual(
        features, ['A'],
        'Features with smaller frequency than the threshold should be filtered.'
    )

    self.assertListEqual(
        X.tolist(),
        [
            # A    BIAS
            [True, True],
            [True, True],
            [True, True],
            [True, True],
            [True, True],
        ],
        'X should represent the filtered entry features with a bias column.')

    self.assertListEqual(
        Y.tolist(), [
            True,
            False,
            True,
            False,
            True,
        ], 'Y should represent the entry labels even some labels are filtered.')

  def test_split_dataset(self) -> None:
    N = 100
    X = np.random.rand(N, 2)
    Y = np.arange(N)
    split_ratio = .8
    X_train, X_test, Y_train, Y_test = train.split_dataset(X, Y, split_ratio)
    self.assertAlmostEqual(X_train.shape[0], N * split_ratio)
    self.assertAlmostEqual(X_test.shape[0], N * (1 - split_ratio))
    self.assertAlmostEqual(X_train.shape[1], 2)
    self.assertAlmostEqual(X_test.shape[1], 2)
    self.assertAlmostEqual(Y_train.shape[0], N * split_ratio)
    self.assertAlmostEqual(Y_test.shape[0], N * (1 - split_ratio))

  def test_fit(self) -> None:
    # Prepare a dataset that the 2nd feature (= the 2nd col in X) perfectly
    # correlates with Y in a negative way.
    X: npt.NDArray[np.bool_] = np.array([
        [False, True, True, False],
        [True, True, False, True],
        [False, False, True, False],
        [True, False, False, True],
    ])
    Y: npt.NDArray[np.bool_] = np.array([
        False,
        False,
        True,
        True,
    ])
    features = ['a', 'b', 'c']
    iters = 5
    out_span = 2
    train.fit(X, Y, X, Y, features, iters, WEIGHTS_FILE_PATH, LOG_FILE_PATH,
              out_span)
    with open(WEIGHTS_FILE_PATH) as f:
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

    with open(LOG_FILE_PATH) as f:
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

  def test_fit_chunk(self) -> None:
    # Prepare a dataset that the 2nd feature (= the 2nd col in X) perfectly
    # correlates with Y in a negative way.
    X: npt.NDArray[np.bool_] = np.array([
        [False, True, True, False],
        [True, True, False, True],
        [False, False, True, False],
        [True, False, False, True],
    ])
    Y: npt.NDArray[np.bool_] = np.array([
        False,
        False,
        True,
        True,
    ])
    features = ['a', 'b', 'c']
    iters = 5
    out_span = 2
    chunk_size = 2
    train.fit(X, Y, X, Y, features, iters, WEIGHTS_FILE_PATH, LOG_FILE_PATH,
              out_span, chunk_size)
    with open(WEIGHTS_FILE_PATH) as f:
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

    with open(LOG_FILE_PATH) as f:
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

    train.fit(X, Y, X, Y, features, iters, WEIGHTS_FILE_PATH, LOG_FILE_PATH,
              out_span, 2)

  def tearDown(self) -> None:
    os.remove(WEIGHTS_FILE_PATH)
    os.remove(LOG_FILE_PATH)
    os.remove(ENTRIES_FILE_PATH)


if __name__ == '__main__':
  unittest.main()
