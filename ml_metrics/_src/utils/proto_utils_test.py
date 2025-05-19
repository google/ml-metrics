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
from ml_metrics._src.utils import proto_utils
from ml_metrics._src.utils import test_utils
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

  def test_single_example(self):
    data = {
        'bytes_key': b'\x80abc',  # not utf-8 decodable
        'str_key': 'str_test',
        'init_key': 123,
        'np_int': np.int32(123),
        'float_key': 4.56,
        'np_float': np.float32(123),
    }
    e = proto_utils.dict_to_tf_example(data).SerializeToString()
    actual = proto_utils.tf_examples_to_dict(e)
    self.assertDictAlmostEqual(data, actual, places=6)

  def test_batch_example(self):
    data = {
        'bytes_key': [b'\x80abc', b'\x80def'],  # not utf-8 decodable
        'str_key': ['str_test', 'str_test2'],
        'init_key': [123, 456],
        'np_int': [np.int32(123), np.int32(456)],
        'float_key': [4.56, 7.89],
        'np_float': [np.float32(123), np.float32(456)],
    }
    e = proto_utils.dict_to_tf_example(data)
    actual = proto_utils.tf_examples_to_dict(e)
    test_utils.assert_nested_container_equal(self, data, actual, places=6)

  @parameterized.named_parameters(
      dict(
          testcase_name='with_single_example',
          num_elems=1,
      ),
      dict(
          testcase_name='multiple_examples',
          num_elems=3,
      ),
  )
  def test_multiple_examples_as_batch(self, num_elems):
    data = {
        'bytes_key': b'\x80abc',  # not utf-8 decodable
        'str_key': 'str_test',
        'init_key': 123,
        'np_int': np.int32(123),
        'float_key': 4.56,
        'np_float': np.float32(123),
    }
    e = [proto_utils.dict_to_tf_example(data) for _ in range(num_elems)]
    actual = proto_utils.tf_examples_to_dict(e)
    expected = {k: [v] * num_elems for k, v in data.items()}
    test_utils.assert_nested_container_equal(self, expected, actual, places=6)

  def test_empty_example(self):
    self.assertEmpty(proto_utils.tf_examples_to_dict([]))

  def test_unsupported_type(self):
    with self.assertRaisesRegex(TypeError, 'Unsupported type'):
      proto_utils.tf_examples_to_dict('unsupported_type')

  def test_unsupported_value_type(self):
    with self.assertRaisesRegex(
        TypeError, 'Value for "a" is not a supported type'
    ):
      proto_utils.dict_to_tf_example({'a': [example_pb2.Example()]})

  def test_multiple_examples_missing_key(self):
    data = [{'a': 'a', 'b': 1}, {'b': 2}]
    examples = [proto_utils.dict_to_tf_example(d) for d in data]
    with self.assertRaisesRegex(ValueError, 'Missing keys'):
      _ = proto_utils.tf_examples_to_dict(examples)


if __name__ == '__main__':
  absltest.main()
