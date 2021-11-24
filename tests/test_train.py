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

import os
import sys
import unittest
from pathlib import Path
import numpy as np

LIB_PATH = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, os.path.abspath(LIB_PATH))

from scripts import train

ENTRIES_FILE_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), 'entries_test.txt'))
WEIGHTS_FILE_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), 'weights_test.txt'))
LOG_FILE_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), 'train_test.log'))


class TestTrain(unittest.TestCase):

  def setUp(self):
    Path(WEIGHTS_FILE_PATH).touch()
    Path(LOG_FILE_PATH).touch()
    with open(ENTRIES_FILE_PATH, 'w') as f:
      f.write((
          ' 1\tA\tC\n'  # the first column represents the label (-1 / 1).
          '-1\tA\tB\n'  # the rest cols represents the associated features.
          ' 1\tA\tC\n'
          '-1\tA\n'
          ' 1\tA\tC\n'))

  def test_pred(self):
    X = np.array([
        [True, False, True, False],
        [False, True, False, True],
    ])
    phis = {
        1: 8,  # Weights Feature #1 by 8.
        2: 2,  # Weights Feature #2 by 2.
    }
    # Since Feature #1 (= the 2nd col in X) wins, the prediction should be:
    # [
    #   False,
    #   True,
    # ]
    pred = train.pred(phis, X)
    self.assertListEqual(pred.tolist(), [False, True])

  def test_preprocess(self):
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

    self.assertListEqual(Y.tolist(), [
        True,
        False,
        True,
        False,
        True,
    ], 'Y should represent the entry labels even filtered.')

  def test_split_dataset(self):
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

  def test_fit(self):
    # Prepare a dataset that the 2nd feature (= the 2nd col in X) perfectly
    # correlates with Y in a negative way.
    X = np.array([
        [False, True, True, False],
        [True, True, False, True],
        [False, False, True, False],
        [True, False, False, True],
    ])
    Y = np.array([
        False,
        False,
        True,
        True,
    ])
    features = ['a', 'b', 'c']
    iters = 1
    train.fit(X, Y, features, iters, WEIGHTS_FILE_PATH, LOG_FILE_PATH)
    with open(WEIGHTS_FILE_PATH) as f:
      weights = f.read().splitlines()
    top_feature = weights[0].split('\t')[0]
    self.assertEqual(
        top_feature, 'b', msg='The most effective feature should be selected.')

  def tearDown(self):
    os.remove(WEIGHTS_FILE_PATH)
    os.remove(LOG_FILE_PATH)
    os.remove(ENTRIES_FILE_PATH)


if __name__ == '__main__':
  unittest.main()
