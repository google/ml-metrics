# Copyright 2025 Google LLC
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
from absl.testing import absltest
from ml_metrics._src import types


class BaseTypesTest(absltest.TestCase):

  def test_is_resolvable(self):

    class Foo:

      @classmethod
      def result_(cls):
        pass

    self.assertIsInstance(Foo(), types.Resolvable)
    self.assertFalse(types.is_resolvable(Foo()))

  def test_is_makeable(self):

    class Foo:

      @classmethod
      def make(cls):
        pass

    self.assertIsInstance(Foo(), types.Makeable)
    self.assertFalse(types.is_makeable(Foo()))

  def test_is_shardable(self):

    class Foo:

      @classmethod
      def shard(cls, shard_index):
        pass

    self.assertFalse(types.is_shardable(Foo()))

  def test_is_serializable(self):

    class Foo:

      def get_config(self):
        pass

      def from_config(self, config):
        pass

    self.assertTrue(types.is_serializable(Foo()))

  def test_is_recoverable(self):

    class Foo:

      @property
      def state(self):
        pass

      def from_state(self, state):
        pass

    self.assertTrue(types.is_recoverable(Foo()))

  def test_is_not_recoverable(self):

    class Foo1:

      @property
      def state(self):
        pass

    self.assertFalse(types.is_recoverable(Foo1()))

    class Foo2:

      def from_state(self, state):
        pass

    self.assertFalse(types.is_recoverable(Foo2()))


if __name__ == "__main__":
  absltest.main()
