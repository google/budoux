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
from functools import partial
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
from jax import device_put, jit

EPS = np.finfo(float).eps  # type: np.floating[typing.Any]
DEFAULT_OUTPUT_NAME = 'weights.txt'
DEFAULT_LOG_NAME = 'train.log'
DEFAULT_FEATURE_THRES = 10
DEFAULT_ITERATION = 10000
DEFAULT_OUT_SPAN = 100
ArgList = typing.Optional[typing.List[str]]


class Result(NamedTuple):
  tp: int
  tn: int
  fp: int
  fn: int
  accuracy: float
  precision: float
  recall: float
  fscore: float


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
    entries = [
        row.strip().split('\t') for row in f.read().splitlines() if row.strip()
    ]
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
    indices = [feature_index[col] for col in entry[1:] if col in feature_index]
    X[i, indices] = True
  X[:, -1] = True  # add a bias column.
  return X, Y, features


@jit
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

  alphas = jnp.array(list(phis.values()))
  y = 2 * (
      X[:, list(phis.keys())] == True  # noqa (cannot replace `==` with `is`)
  ) - 1
  result: npt.NDArray[np.bool_] = y.dot(alphas) > 0
  return result


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


def get_metrics(pred: npt.NDArray[np.bool_],
                actual: npt.NDArray[np.bool_]) -> Result:
  tp = np.sum(np.logical_and(pred == 1, actual == 1))
  tn = np.sum(np.logical_and(pred == 0, actual == 0))
  fp = np.sum(np.logical_and(pred == 1, actual == 0))
  fn = np.sum(np.logical_and(pred == 0, actual == 1))
  accuracy = (tp + tn) / (tp + tn + fp + fn)
  precision = tp / (tp + fp)
  recall = tp / (tp + fn)
  return Result(
      tp=tp,
      tn=tn,
      fp=fp,
      fn=fn,
      accuracy=accuracy,
      precision=precision,
      recall=recall,
      fscore=2 * precision * recall / (precision + recall),
  )


@jit
def update_weight(
    w: npt.NDArray[np.float64], YX: npt.NDArray[np.bool_]
) -> typing.Tuple[npt.NDArray[np.float64], float, int, bool]:
  res = w.dot(YX)
  err = 0.5 - jnp.abs(res - 0.5)
  m_best = err.argmin()
  pol_best = res.at[m_best].get() < 0.5
  err_min = err.at[m_best].get()
  alpha = jnp.log((1 - err_min) / (err_min + EPS))
  w = w * jnp.exp(alpha * (YX[:, m_best] == pol_best))
  w = w / w.sum()
  return w, alpha, m_best, pol_best


@partial(jit, static_argnames=['chunk_size'])
def update_weight_chunk(
    w: npt.NDArray[np.float64], YX: npt.NDArray[np.bool_],
    chunk_size: int) -> typing.Tuple[npt.NDArray[np.float64], float, int, bool]:
  N, M = YX.shape
  res = jnp.zeros(M)
  for chunk in range(0, N, chunk_size):
    res += w[chunk:chunk + chunk_size].dot(YX[chunk:chunk + chunk_size])
  err = 0.5 - jnp.abs(res - 0.5)
  m_best = err.argmin()
  pol_best = res.at[m_best].get() < 0.5
  err_min = err.at[m_best].get()
  alpha = jnp.log((1 - err_min) / (err_min + EPS))
  w = w * jnp.exp(alpha * (YX[:, m_best] == pol_best))
  w = w / w.sum()
  return w, alpha, m_best, pol_best


def fit(X_train: npt.NDArray[np.bool_],
        Y_train: npt.NDArray[np.bool_],
        X_test: npt.NDArray[np.bool_],
        Y_test: npt.NDArray[np.bool_],
        features: typing.List[str],
        iters: int,
        weights_filename: str,
        log_filename: str,
        out_span: int,
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
    out_span (int): Iteration span to output metics and weights.
    chunk_size (Optional[int]): A chunk size to split training entries for
      memory efficiency.

  Returns:
    phi (Dict[int, float]): Learned child classifiers.
  """
  with open(weights_filename, 'w') as f:
    f.write('')
  with open(log_filename, 'w') as f:
    f.write(
        'iter\ttrain_accuracy\ttrain_precision\ttrain_recall\ttrain_fscore\t'
        'test_accuracy\ttest_precision\ttest_recall\ttest_fscore\n')
  print('Outputting learned weights to %s ...' % (weights_filename))

  phis: typing.Dict[int, float] = dict()
  phi_buffer: typing.List[typing.Tuple[str, float]] = []

  assert (X_train.shape[1] == X_test.shape[1]
         ), 'Training and test entries should have the same number of features.'
  assert (X_train.shape[1] - 1 == len(features)
         ), 'The training data should have the same number of features + BIAS.'
  assert (X_train.shape[0] == Y_train.shape[0]
         ), 'Training entries and labels should have the same number of items.'
  assert (X_test.shape[0] == Y_test.shape[0]
         ), 'Testing entries and labels should have the same number of items.'

  X_train = device_put(X_train)
  Y_train = device_put(Y_train)
  X_test = device_put(X_test)
  Y_test = device_put(Y_test)
  N_train, _ = X_train.shape
  w = jnp.ones(N_train) / N_train
  YX_train = Y_train[:, None] ^ X_train

  def output_progress(t: int) -> None:
    with open(weights_filename, 'a') as f:
      f.write('\n'.join('%s\t%.6f' % p for p in phi_buffer) + '\n')
    phi_buffer.clear()
    pred_train = pred(phis, X_train)
    pred_test = pred(phis, X_test)
    metrics_train = get_metrics(pred_train, Y_train)
    metrics_test = get_metrics(pred_test, Y_test)
    print('=== %s ===' % t)
    print()
    print('train accuracy:\t%.5f' % metrics_train.accuracy)
    print('train prec.:\t%.5f' % metrics_train.precision)
    print('train recall:\t%.5f' % metrics_train.recall)
    print('train fscore:\t%.5f' % metrics_train.fscore)
    print()
    print('test accuracy:\t%.5f' % metrics_test.accuracy)
    print('test prec.:\t%.5f' % metrics_test.precision)
    print('test recall:\t%.5f' % metrics_test.recall)
    print('test fscore:\t%.5f' % metrics_test.fscore)
    print()
    with open(log_filename, 'a') as f:
      f.write('%d\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\n' % (
          t,
          metrics_train.accuracy,
          metrics_train.precision,
          metrics_train.recall,
          metrics_train.fscore,
          metrics_test.accuracy,
          metrics_test.precision,
          metrics_test.recall,
          metrics_test.fscore,
      ))

  for t in range(iters):
    if chunk_size:
      w, alpha, m_best, pol_best = update_weight_chunk(w, YX_train, chunk_size)
    else:
      w, alpha, m_best, pol_best = update_weight(w, YX_train)
    w.block_until_ready()
    m_best = int(m_best)
    alpha_signed = alpha if pol_best else -alpha
    phis.setdefault(m_best, 0)
    phis[m_best] += alpha_signed
    feature = features[m_best] if m_best < len(features) else 'BIAS'
    phi_buffer.append((feature, alpha_signed))
    if (t + 1) % out_span == 0:
      output_progress(t + 1)
  if len(phi_buffer) > 0:
    output_progress(t + 1)

  return phis


def parse_args(test: ArgList = None) -> argparse.Namespace:
  """Parses commandline arguments.

  Args:
    test (typing.Optional[typing.List[str]], optional): Commandline args for
      testing. Defaults to None.

  Returns:
    argparse.Namespace: Parsed data of args.
  """
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
      'encoded_train_data', help='File path for the encoded training data.')
  parser.add_argument(
      '-o',
      '--output',
      help=f'Output file path for the learned weights. (default: {DEFAULT_OUTPUT_NAME})',
      type=str,
      default=DEFAULT_OUTPUT_NAME)
  parser.add_argument(
      '--log',
      help=f'Output file path for the training log. (default: {DEFAULT_LOG_NAME})',
      type=str,
      default=DEFAULT_LOG_NAME)
  parser.add_argument(
      '--feature-thres',
      help=f'Threshold value of the minimum feature frequency. (default: {DEFAULT_FEATURE_THRES})',
      type=int,
      default=DEFAULT_FEATURE_THRES)
  parser.add_argument(
      '--iter',
      help=f'Number of iterations for training. (default: {DEFAULT_ITERATION})',
      type=int,
      default=DEFAULT_ITERATION)
  parser.add_argument(
      '--out-span',
      help=f'Iteration span to output metrics and weights. (default: {DEFAULT_OUT_SPAN})',
      type=int,
      default=DEFAULT_OUT_SPAN)
  parser.add_argument(
      '--chunk-size',
      type=int,
      help='A chunk size to split training entries for memory efficiency. (default: None)',
      default=None)
  if test is None:
    return parser.parse_args()
  else:
    return parser.parse_args(test)


def main() -> None:
  args = parse_args()
  train_data_filename: str = args.encoded_train_data
  weights_filename: str = args.output
  log_filename: str = args.log
  feature_thres = int(args.feature_thres)
  iterations = int(args.iter)
  out_span = int(args.out_span)
  chunk_size = int(args.chunk_size) if args.chunk_size is not None else None

  X, Y, features = preprocess(train_data_filename, feature_thres)
  X_train, X_test, Y_train, Y_test = split_dataset(X, Y)
  del X, Y
  fit(X_train, Y_train, X_test, Y_test, features, iterations, weights_filename,
      log_filename, out_span, chunk_size)

  print('Training done. Export the model by passing %s to build_model.py' %
        (weights_filename))


if __name__ == '__main__':
  main()
