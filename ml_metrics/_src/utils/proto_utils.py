# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Proto utils."""

import collections
from collections.abc import Iterable
from typing import Any
from ml_metrics._src.tools.telemetry import telemetry
import numpy as np
from tensorflow.core.example import example_pb2


_ExampleOrBytes = bytes | example_pb2.Example


def _maybe_deserialize(ex: _ExampleOrBytes) -> example_pb2.Example:
  if isinstance(ex, bytes):
    return example_pb2.Example.FromString(ex)
  if isinstance(ex, example_pb2.Example):
    return ex
  raise TypeError('Unsupported type: %s' % type(ex))


@telemetry.function_monitor(api='ml_metrics', category=telemetry.CATEGORY.UTIL)
def tf_examples_to_dict(
    examples: Iterable[_ExampleOrBytes] | _ExampleOrBytes,
) -> dict[
    str,
    list[int | float | bytes] | list[list[int | float | bytes]],
]:
  """Parses serialized or unserialized tf.train.Examples to a dict.

  The conversion assumes all examples have the same features. If not, a
  ValueError will be raised.

  Args:
    examples: A single tf.train.Example, serialized tf.train.Example, or an
      iterable of tf.train.Examples and/or serialized tf.train.Examples.

  Returns:
    A dict mapping feature names to lists of feature values.

  Raises:
    ValueError: If the features are not all present in all examples.
  """

  single_example = False
  if isinstance(examples, (bytes, example_pb2.Example)):
    single_example = True
    examples = [examples]

  result = collections.defaultdict(list)

  for ex in examples:
    ex = _maybe_deserialize(ex)
    features = dict(ex.features.feature)

    if result and result.keys() != features.keys():
      raise ValueError(
          'All examples must have the same features, got %s and %s'
          % (result.keys(), features.keys())
      )

    for name, values in features.items():
      result[name].append(getattr(values, values.WhichOneof('kind')).value)

  if single_example:
    return {k: v[0] for k, v in result.items()}
  return result


@telemetry.function_monitor(api='ml_metrics', category=telemetry.CATEGORY.UTIL)
def dict_to_tf_example(data: dict[str, Any]) -> example_pb2.Example:
  """Creates a tf.Example from a dictionary."""

  example = example_pb2.Example()
  for key, values in data.items():
    if isinstance(values, (str, bytes, np.floating, float, int, np.integer)):
      values = [values]

    if not values:
      # Skip empty features.
      continue

    if isinstance(values[0], str):
      for v in values:
        assert isinstance(v, str), f'bad str type: {values}'
        example.features.feature[key].bytes_list.value.append(v.encode())
      continue

    if isinstance(values[0], bytes):
      feature_kind = 'bytes_list'
    elif isinstance(values[0], (float, np.floating)):
      feature_kind = 'float_list'
    elif isinstance(values[0], (int, np.integer)):
      feature_kind = 'int64_list'
      for v in values:
        if isinstance(v, (float, np.floating)):
          # If a float is encountered in the list, we consider the whole feature
          # to be a float_list.
          feature_kind = 'float_list'
          break
        elif not isinstance(v, (int, np.integer)):
          break
    else:
      raise TypeError(f'Values for "{key}" is not a supported type.')

    feature_list = getattr(example.features.feature[key], feature_kind).value
    feature_list.extend(values)

  return example
