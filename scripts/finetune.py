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
"""Finetunes the model with the given training data."""

import argparse
import array
import typing
import json
from collections import OrderedDict
from jax import numpy as jnp
from jax import Array, jit, grad


class NormalizedModel(typing.NamedTuple):
  keys: typing.List[str]
  weights: Array


class Dataset(typing.NamedTuple):
  X: Array
  Y: Array


def load_model(file_path: str) -> NormalizedModel:
  """Loads a model as a pair of a features list and a normalized weight vector.
  
  Args:
    file_path: A file path for the model JSON file.

  Returns:
    A normalized model, which is a pair of a list of feature identifiers and a
    normalized weight vector.
  """
  with open(file_path) as f:
    model = json.load(f)
  model_flat = OrderedDict()
  for category in model:
    for item in model[category]:
      model_flat['%s:%s' % (category, item)] = model[category][item]
  weights = jnp.array(list(model_flat.values()))
  weights = weights / weights.std()
  weights = weights - weights.mean()
  keys = list(model_flat.keys())
  return NormalizedModel(keys, weights)


def load_dataset(file_path: str, model: NormalizedModel) -> Dataset:
  """Loads a dataset from the given file path.
  
  Args:
    file_path: A file path for the encoded data file.
    model: A normalized model.

  Returns:
    A dataset of inputs (X) and outputs (Y).
  """
  xs = []
  ys = array.array('B')
  with open(file_path) as f:
    for row in f:
      cols = row.strip().split('\t')
      if len(cols) < 2:
        continue
      ys.append(cols[0] == '1')
      xs.append(tuple(k in set(cols[1:]) for k in model.keys))
  X = jnp.array(xs) * 2 - 1
  Y = jnp.array(ys)
  return Dataset(X, Y)


def cross_entropy_loss(weights: Array, x: Array, y: Array) -> Array:
  """Calcurates a cross entropy loss with a prediction by a sigmoid function.
  
  Args:
    weights: A weight vector.
    x: An input array.
    y: A target output array.

  Returns:
    A cross entropy loss.
  """
  pred = 1 / (1 + jnp.exp(-x.dot(weights)))
  return -jnp.mean(y * jnp.log(pred) + (1 - y) * jnp.log(1 - pred))


def fit(weights: Array,
        train_dataset: Dataset,
        iter: int = 1000,
        learning_rate: float = 0.1) -> Array:
  """Updates the weights with the given dataset.
  
  Args:
    train_dataset: A train dataset.
    iter: A number of iterations (default: 1000).
    learning_rate: A learning rate (default: 0.1).

  Returns:
    An updated weight vector.
  """
  grad_loss = jit(grad(cross_entropy_loss, argnums=0))
  for _ in range(iter):
    weights = weights - learning_rate * grad_loss(weights, train_dataset.X,
                                                  train_dataset.Y)
  return weights


def parse_args(
    test: typing.Optional[typing.List[str]] = None) -> argparse.Namespace:
  """Parses commandline arguments.

  Args:
    test (typing.Optional[typing.List[str]], optional): Commandline args for
      testing. Defaults to None.

  Returns:
    Parsed arguments (argparse.Namespace).
  """
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
      'train_data', help='File path for the encoded training data.')
  parser.add_argument('base_model', help='File path for the base model file.')
  if test is None:
    return parser.parse_args()
  else:
    return parser.parse_args(test)


def main() -> None:
  args = parse_args()
  train_data_path: str = args.train_data
  base_model_path: str = args.base_model

  model = load_model(base_model_path)
  train_dataset = load_dataset(train_data_path, model)
  fit(model.weights, train_dataset.X, train_dataset.Y)


if __name__ == '__main__':
  main()
