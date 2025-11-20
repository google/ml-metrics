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
import chainable as cnb
from chainable import test_utils
from ml_metrics._src.utils import proto_utils
import numpy as np
from absl.testing import absltest
from absl.testing import parameterized
from tensorflow.core.example import example_pb2


def _get_tf_example(**kwargs):
  example = example_pb2.Example()
  for k, v in kwargs.items():
    example.features.feature[k].bytes_list.value.append(v)
    return example


class TFExampleTest(parameterized.TestCase):

  def test_single_example_all_scalar(self):
    data = {
        'bytes_key': b'\x80abc',  # not utf-8 decodable
        'str_key': 'str_test1',
        'init_key': 123,
        'np_int': np.int32(123),
        'float_key': 4.56,
        'np_float': np.float32(123),
    }
    e = proto_utils.dict_to_tf_example(data).SerializeToString()
    actual = proto_utils.tf_example_to_dict(e)
    self.assertDictAlmostEqual(data, actual, places=6)

  def test_single_example(self):
    data = {
        'bytes_key': b'\x80abc',  # not utf-8 decodable
        'str_key': 'str_test1',
        'multi_str_key': ['str_test1', 'str_test2'],
        'init_key': 123,
        'multi_init_key': [123, 234],
        'np_int': np.int32(123),
        'float_key': 4.56,
        'np_float': np.float32(123),
    }
    e = proto_utils.dict_to_tf_example(data).SerializeToString()
    actual = proto_utils.tf_example_to_dict(e)
    self.assertDictAlmostEqual(data, actual, places=6)

  def test_batch_example(self):
    data = {
        'bytes_key': [b'\x80abc', b'\x80def'],  # not utf-8 decodable
        'str_key': ['str_test', 'str_test2'],
        'init_key': [123],
        'np_int': [np.int32(123), np.int32(456)],
        'float_key': [4.56, 7.89],
        'np_float': [np.float32(123), np.float32(456)],
    }
    e = proto_utils.dict_to_tf_example(data)
    actual = proto_utils.tf_example_to_dict(e, unwrap_scalar=False)
    test_utils.assert_nested_container_equal(self, data, actual, places=6)

  def test_multiple_examples_with_chainable_batch(self):
    data = [{'a': 1, 'b': [2, 3]}, {'a': 1, 'b': [2, 3]}]
    examples = [proto_utils.dict_to_tf_example(d) for d in data]
    p = (
        cnb.P()
        .ds(examples, parse_fn=proto_utils.tf_example_to_dict)
        .select('a', 'b')
        .batch(3)
    )
    actual = list(p.make())
    expected = [{'a': [1, 1], 'b': [[2, 3], [2, 3]]}]
    self.assertEqual(expected, actual)

  def test_unsupported_type(self):
    with self.assertRaisesRegex(TypeError, 'Unsupported type'):
      proto_utils.tf_example_to_dict('unsupported_type')

  def test_unsupported_value_type(self):
    with self.assertRaisesRegex(
        TypeError, 'Value for "a" is not a supported type'
    ):
      proto_utils.dict_to_tf_example({'a': [example_pb2.Example()]})


if __name__ == '__main__':
  absltest.main()
