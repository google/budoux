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
"""Runs training and exports the learned weights to build a model."""

import argparse
import typing
from collections import Counter

import numpy as np
import numpy.typing as npt

EPS = np.finfo(float).eps  # type: np.floating[typing.Any]


def preprocess(
    entries_filename: str, feature_thres: int
) -> typing.Tuple[npt.NDArray[np.bool_], npt.NDArray[np.bool_],
                  typing.List[str]]:
  """Loads entries and translates them into NumPy arrays.

  Args:
    entries_filename (str): A file path to the entries file.
    feature_thres (str): A threshold to filter out features whose frequency is
      below the given value.

  Returns:
    X (numpy.ndarray): Input entries.
    Y (numpy.ndarray): Output labels.
    features (List[str]): Effective features.
  """
  with open(entries_filename) as f:
    entries = [row.strip().split('\t') for row in f.read().splitlines()]
  print('#entries:\t%d' % (len(entries)))

  features_counter: typing.Counter[str] = Counter()
  for entry in entries:
    features_counter.update(entry[1:])
  features = [
      item[0]
      for item in features_counter.most_common()
      if item[1] > feature_thres
  ]
  print('#features:\t%d' % (len(features)))
  feature_index = dict([(feature, i) for i, feature in enumerate(features)])

  M = len(features) + 1
  N = len(entries)
  Y: npt.NDArray[np.bool_] = np.zeros(N, dtype=bool)
  X: npt.NDArray[np.bool_] = np.zeros((N, M), dtype=bool)

  for i, entry in enumerate(entries):
    Y[i] = entry[0] == '1'
    for col in entry[1:]:
      if col in feature_index:
        X[i, feature_index[col]] = True
    X[i, -1] = True  # add a bias column.
  return X, Y, features


def pred(phis: typing.Dict[int, float],
         X: npt.NDArray[np.bool_]) -> npt.NDArray[np.bool_]:
  """Predicts the output from the given classifiers and input entries.

  Args:
    phis (Dict[int, float]): Classifiers represented as a mapping from the
      feature index to its score.
    X (numpy.ndarray): Input entries.

  Returns:
    A list of inferred labels.
  """
  alphas: npt.NDArray[np.float64]
  y: npt.NDArray[np.int64]

  alphas = np.array(list(phis.values()))
  y = 2 * (X[:, list(phis.keys())]
           == True) - 1  # noqa (cannot replace `==` with `is`)
  return y.dot(alphas) > 0


def split_dataset(
    X: npt.NDArray[typing.Any],
    Y: npt.NDArray[typing.Any],
    split_ratio: float = 0.9
) -> typing.Tuple[npt.NDArray[typing.Any], npt.NDArray[typing.Any],
                  npt.NDArray[typing.Any], npt.NDArray[typing.Any]]:
  """Splits given entries and labels to training and testing datasets.

  Args:
    X (numpy.ndarray): Entries to split.
    Y (numpy.ndarray): Labels to split.
    split_ratio (float, optional): The ratio to hold for the training dataset.

  Returns:
    X_train (numpy.ndarray): Training entries.
    X_test (numpy.ndarray): Testing entries.
    Y_train (numpy.ndarray): Training labels.
    Y_test (numpy.ndarray): Testing labels.
  """
  N, _ = X.shape
  np.random.seed(0)
  indices = np.random.permutation(N)
  X_train = X[indices[:int(N * split_ratio)]]
  X_test = X[indices[int(N * split_ratio):]]
  Y_train = Y[indices[:int(N * split_ratio)]]
  Y_test = Y[indices[int(N * split_ratio):]]
  return X_train, X_test, Y_train, Y_test


def fit(X_train: npt.NDArray[np.bool_],
        Y_train: npt.NDArray[np.bool_],
        X_test: npt.NDArray[np.bool_],
        Y_test: npt.NDArray[np.bool_],
        features: typing.List[str],
        iters: int,
        weights_filename: str,
        log_filename: str,
        chunk_size: typing.Optional[int] = None) -> typing.Dict[int, float]:
  """Trains an AdaBoost classifier.

  Args:
    X_train (numpy.ndarray): Training entries.
    Y_train (numpy.ndarray): Training labels.
    X_test (numpy.ndarray): Testing entries.
    Y_test (numpy.ndarray): Testing labels.
    features (List[str]): Features, which correspond to the columns of entries.
    iters (int): A number of training iterations.
    weights_filename (str): A file path to write the learned weights.
    log_filename (str): A file path to log the accuracy along with training.
    chunk_size (Optional[int]): A chunk size to split training entries into chunks for memory reduction
      when calculating AdaBoost's weighted training error.

  Returns:
    phi (Dict[int, float]): Leanred child classifiers.
  """
  with open(weights_filename, 'w') as f:
    f.write('')
  with open(log_filename, 'w') as f:
    f.write('')
  print('Outputting learned weights to %s ...' % (weights_filename))

  phis: typing.Dict[int, float] = dict()

  assert (X_train.shape[1] == X_test.shape[1]
         ), 'Training and test entries should have the same number of features.'
  assert (X_train.shape[1] - 1 == len(features)
         ), 'The training data should have the same number of features + BIAS.'
  assert (X_train.shape[0] == Y_train.shape[0]
         ), 'Training entries and labels should have the same number of items.'
  assert (X_test.shape[0] == Y_test.shape[0]
         ), 'Testing entries and labels should have the same number of items.'

  N_train, M_train = X_train.shape
  w = np.ones(N_train) / N_train

  for t in range(iters):
    print('=== %s ===' % (t))
    if chunk_size is None:
      res: npt.NDArray[np.float64] = w.dot(Y_train[:, None] ^ X_train) / w.sum()
    else:
      res = np.zeros(M_train)
      for i in range(0, N_train, chunk_size):
        Y_train_chunk = Y_train[i:i + chunk_size]
        X_train_chunk = X_train[i:i + chunk_size]
        w_chunk = w[i:i + chunk_size]
        res += w_chunk.dot(Y_train_chunk[:, None] ^ X_train_chunk)
      res = res / w.sum()
    err = 0.5 - np.abs(res - 0.5)
    m_best = int(err.argmin())
    pol_best = res[m_best] < 0.5
    err_min = err[m_best]
    print('min error:\t', err_min)
    print('best tree:\t', m_best)
    alpha = np.log((1 - err_min) / (err_min + EPS))
    phis.setdefault(m_best, 0)
    phis[m_best] += alpha if pol_best else -alpha
    miss = Y_train ^ X_train[:, m_best]
    if not pol_best:
      miss = ~(miss)
    w = w * np.exp(alpha * miss)
    with open(weights_filename, 'a') as f:
      feature = features[m_best] if m_best < len(features) else 'BIAS'
      f.write('%s\t%.3f\n' % (feature, alpha if pol_best else -alpha))
    acc_train = (pred(phis, X_train) == Y_train).mean()
    acc_test = (pred(phis, X_test) == Y_test).mean()
    print('training accuracy:\t', acc_train)
    print('testing accuracy:\t', acc_test)
    with open(log_filename, 'a') as f:
      f.write('%.5f\t%.5f\n' % (acc_train, acc_test))
  return phis


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
      'encoded_train_data', help='File path for the encoded training data.')
  parser.add_argument(
      '-o',
      '--output',
      help='Output file path for the learned weights. (default: weights.txt)',
      default='weights.txt')
  parser.add_argument(
      '--log',
      help='Output file path for the training log. (default: train.log)',
      default='train.log')
  parser.add_argument(
      '--feature-thres',
      help='Threshold value of the minimum feature frequency. (default: 10)',
      default=10)
  parser.add_argument(
      '--iter',
      help='Number of iterations for training. (default: 10000)',
      default=10000)
  parser.add_argument(
      '--chunk-size',
      help='A chunk size to split training entries into chunks for memory reduction when calculating AdaBoost\'s weighted training error.'
  )

  return parser.parse_args()


def main() -> None:
  args = parse_args()
  train_data_filename = args.encoded_train_data
  weights_filename = args.output
  log_filename = args.log
  feature_thres = int(args.feature_thres)
  iterations = int(args.iter)
  chunk_size = int(args.chunk_size) if args.chunk_size is not None else None

  X, Y, features = preprocess(train_data_filename, feature_thres)
  X_train, X_test, Y_train, Y_test = split_dataset(X, Y)
  fit(X_train, Y_train, X_test, Y_test, features, iterations, weights_filename,
      log_filename, chunk_size)

  print('Training done. Export the model by passing %s to build_model.py' %
        (weights_filename))


if __name__ == '__main__':
  main()
