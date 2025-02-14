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
"""Library for TreeTransform.

This module contains the main APIs and in-process runner implementations for the
TreeTransform (pipeline).

Following is an example of an evaluation pipeline where:
 1. Data is read processed and mapped as "inputs", "features", and "labels";
 2. Model takes "inputs" for inference generating "outputs".
 3. Some metrics are computed using "outputs" and "labels".
  a. The metrics can be sliced by single or multiple features (slice crossed).
  b. The metrics can be sliced by slice_fn with arbitrary feature transform and
    fanout logic.

  eval_pipeline = (
      TreeTransform.new()
      .data_source(iterator_fn)
      .apply(
          fn=preprocess_fn,
          output_keys=('inputs', 'feature_a', 'feature_b', 'y_true'),
      )
      .assign('outputs', fn=model, input_keys='inputs')
      .aggregate(
          fn=metrics_agg_fn,
          input_keys=('y_true', 'outputs'),
          output_keys=('precision', 'recall'),
      )
      .add_slice('feature_a')
      .add_slice(('feature_a', 'feature_b'))
      .add_slice('feature_b', slice_fn, slice_name='transformed_feature'),
  )
"""

from __future__ import annotations

from collections.abc import Callable, Generator, Iterable, Iterator, Mapping, Sequence
from concurrent import futures
import copy
import dataclasses
import enum
import functools
import itertools
import time
from typing import Any, Generic, Self, TypeVar
import uuid

from absl import logging
from ml_metrics._src import types
from ml_metrics._src.aggregates import base as aggregates
from ml_metrics._src.chainables import io
from ml_metrics._src.chainables import lazy_fns
from ml_metrics._src.chainables import tree
from ml_metrics._src.chainables import tree_fns
from ml_metrics._src.utils import iter_utils
import more_itertools as mit


TreeMapKey = tree.TreeMapKey
TreeMapKeys = tree.TreeMapKeys

Fn = Callable[..., Any]
FnT = TypeVar('FnT', Callable, aggregates.Aggregatable)
TreeFnT = TypeVar('TreeFnT', bound=tree_fns.TreeFn)
TreeMapKeyT = TypeVar('TreeMapKeyT', bound=TreeMapKey)
Map = Mapping[TreeMapKeyT, Any]
TreeTransformT = TypeVar('TreeTransformT', bound='TreeTransform')
TreeFn = tree_fns.TreeFn
_ValueT = TypeVar('_ValueT')

_LOGGING_INTERVAL_SECS = 60
_DEFAULT_NUM_THREADS = 0

_is_dict = lambda x: isinstance(x, dict)


def iterate_with_returned(
    iterator: Iterable[_ValueT],
) -> Generator[_ValueT, None, Any]:
  """Converts a generator to an iterator with returned value as the last."""
  returned = yield from iter(iterator)
  yield returned


class RunnerMode(enum.StrEnum):
  DEFAULT = 'default'
  AGGREGATE = 'aggregate'


@dataclasses.dataclass(frozen=True)
class AggregateResult:
  agg_result: Any = None
  agg_state: Any = dataclasses.field(kw_only=True, default=None)


@dataclasses.dataclass(frozen=True)
class _IteratorState:
  input_states: list[io.ShardConfig]
  agg_state: dict[MetricKey, Any] | None


class _MergedIterator(types.Recoverable, iter_utils.MergedIterator[_ValueT]):
  """An iterator that merges multiple iterables."""

  def from_state(self, states: Sequence[io.ShardConfig]) -> Self:
    data_sources = []
    for data_source, ds_state in zip(self._data_sources, states, strict=True):
      if not types.is_recoverable(data_source):
        raise TypeError(f'Data source is not recoverable, got {data_source=}.')
      data_sources.append(data_source.from_state(ds_state))
    return self.__class__(data_sources, self._parallism)

  @property
  def state(self) -> list[io.ShardConfig]:
    states = []
    for iterator in self._iterators:
      if not types.is_recoverable(iterator):
        raise TypeError(f'Data source is not serializable, got {iterator=}.')
      states.append(iterator.state)
    return states


class _IteratorWithAggResult(types.Recoverable, Iterable[_ValueT]):
  """An iterator that returns the last value."""
  input_iterator: _MergedIterator[_ValueT]
  agg_state: dict[MetricKey, Any] | None

  def __init__(
      self,
      tree_fn: CombinedTreeFn,
      *,
      input_iterator: _MergedIterator[_ValueT],
      with_result: bool = True,
      with_agg_state: bool = True,
      with_agg_result: bool = True,
      state: Any = None,
  ):
    self._tree_fn = tree_fn
    self.input_iterator = input_iterator
    self._with_result = with_result
    self.agg_state = state
    self._with_agg_state = state and (with_agg_state or with_agg_result)
    self._with_agg_result = with_agg_result
    self._thread_pool = None

    def iterate_fn(
        input_iterator: Iterable[tree.MapLikeTree | None] = (),
    ) -> Iterator[tree.MapLikeTree | None]:
      """Call a chain of functions in sequence."""
      result = input_iterator
      for fn in self._tree_fn.input_fns:
        result = fn.iterate(result)
      yield from result

    if not tree_fn.num_threads:
      iterator = iterate_fn(self.input_iterator)
    else:
      iterator = iter_utils.piter(
          iterate_fn,
          input_iterators=[self.input_iterator],
          thread_pool=self._get_thread_pool(),
          max_parallism=tree_fn.num_threads,
      )
    self._iterator = iter(iterator)
    self._prev_ticker = time.time()
    self.batch_index = 0

  def _get_thread_pool(self) -> futures.ThreadPoolExecutor:
    if self._thread_pool is None:
      self._thread_pool = futures.ThreadPoolExecutor()
    return self._thread_pool

  @property
  def name(self) -> str:
    return self._tree_fn.name

  @property
  def agg_result(self) -> tree.MapLikeTree[Any] | None:
    if self.agg_state:
      return self._tree_fn.get_result(self.agg_state)
    return None

  def from_state(self, state: _IteratorState) -> _IteratorWithAggResult:
    input_iterator = self.input_iterator.from_state(state.input_states)
    return _IteratorWithAggResult(
        self._tree_fn,
        input_iterator=input_iterator,
        with_result=self._with_result,
        with_agg_state=self._with_agg_state,
        with_agg_result=self._with_agg_result,
        state=state.agg_state,
    )

  @property
  def state(self) -> _IteratorState:
    input_states = self.input_iterator.state
    return _IteratorState(input_states, agg_state=copy.deepcopy(self.agg_state))

  def __next__(self) -> _ValueT:
    if (ticker := time.time()) - self._prev_ticker > _LOGGING_INTERVAL_SECS:
      logging.info(
          'chainable: "%s" processed %d batch.', self.name, self.batch_index
      )
      self._prev_ticker = ticker
    try:
      batch_output = next(self._iterator)
      self.batch_index += 1
      if self._with_agg_state:
        self.agg_state = self._tree_fn.update_state(
            self.agg_state, batch_output
        )
      logging.debug(
          'chainable: "%s" batch cnt %d.', self.name, self.batch_index
      )
      return batch_output if self._with_result else None
    except StopIteration as e:
      returned = None
      if self._with_agg_state:
        # This can collect the agg_state and the agg_result at the end of
        # the iteration and return them as the generator return value.
        agg_result = self.agg_result if self._with_agg_result else None
        returned = AggregateResult(agg_result, agg_state=self.agg_state)
      logging.info(
          'chainable: %s',
          f'"{self.name}" iterator exhausted after'
          f' {self.batch_index} batches, returned a type of'
          f' {type(returned)}.',
      )
      if self._thread_pool is not None:
        self._thread_pool.shutdown(wait=False)
      raise StopIteration(returned) if returned else e

  def __iter__(self):
    return self


@dataclasses.dataclass(frozen=True)
class MetricKey:
  """Metric key to index aggregation state per slice per metric.

  Attributes:
    metrics: The metric name for the corresponding metric.
    slice: The slice key that includes both the feature name and the sliced
      value.
  """

  metrics: TreeMapKey | TreeMapKeys = ()
  slice: tree_fns.SliceKey = tree_fns.SliceKey()


def clear_cache():
  """Clear the cache for maybe_make."""
  lazy_fns.clear_cache()


@dataclasses.dataclass(frozen=True, kw_only=True)
class CombinedTreeFn:
  """Combining multiple transforms into concrete functions.

  This class encapsulates all in-process logic of a TreeTransform.
  The ordering of the execution is as follows:
    input_iterator -> input_fns (w/ slicers) -> agg_fns -> output_fns.

  From a TreeTransform standpoint, the sequence of TreeTransforms are translted
  and into one function here:
    * Base `TreeTrsnform`s (`apply()`, `assign()`, `select()`) are translated to
      `input_fns` in sequence.
    * The first `AggregateTransform` in the chain is translated to `slicers` and
      `agg_fns`.
    * Any transforms after the first `AggregateTransform` in the chain is
      converted to a callable and translated as `output_fns`.
  """

  name: str
  input_fns: Sequence[tree_fns.TreeFn]
  agg_fns: dict[TreeMapKeys | TreeMapKey, tree_fns.TreeAggregateFn]
  slicers: list[tree_fns.Slicer]
  output_fns: Sequence[tree_fns.TreeFn]
  # TODO: b/395649147 - Rename to data_source.
  input_iterator: Iterable[Any] | None
  num_threads: int

  @classmethod
  def from_transform(
      cls,
      transform: TreeTransform,
      *,
      recursive: bool,
      input_state: io.ShardConfig | None,
      mode: RunnerMode = RunnerMode.DEFAULT,
  ) -> Self:
    """Makes a TreeFn from a transform."""
    # Flatten the transform into nodes and find the first aggregation node as
    # the aggregation node for the concrete function. Any node before it is
    # input nodes, any node after it regardless of its type (aggregation or not)
    # is output_node.
    if recursive:
      transforms = transform.flatten_transform(remove_input_transform=True)
    else:
      transforms = [transform]
    input_nodes, output_nodes = [], []
    data_source, agg_node = None, None
    name = ''
    for i, node in enumerate(transforms):
      name = node.name or name
      if node.input_iterator is not None:
        if i == 0:
          data_source = node.input_iterator
        else:
          raise ValueError(
              f'data_source has to be the first node, it is at {i}th position.'
          )
      # Iterator node can also be other types of nodes.
      if agg_node is None:
        if isinstance(node, AggregateTransform):
          if node.is_noop:
            raise ValueError('Empty aggregation, forgot to add aggregation?')
          agg_node = node
        else:
          input_nodes.append(node)
      else:
        output_nodes.append(node)

    # Reset the iterator_node and input_nodes if this is an aggregator.
    if mode == RunnerMode.AGGREGATE:
      data_source = None
      input_nodes = []
      assert agg_node, 'Aggregation fn is required for "Aggregate" mode.'

    # Collect input_iterator.
    data_source = lazy_fns.maybe_make(data_source)
    if input_state is not None:
      if types.is_recoverable(data_source):
        data_source = data_source.from_state(input_state)
      else:
        raise TypeError(
            f'Data source is not configurable but {input_state=} is provided.'
            f' data source type: {type(data_source)}.'
        )
    # Collect all the input functions from the input nodes.
    input_fns = list(
        itertools.chain.from_iterable(node.fns for node in input_nodes)
    )
    input_fns = [tree_fn.maybe_make() for tree_fn in input_fns]
    # Collect all the aggregation functions from the aggregation node.
    agg_fns, slice_fns = {}, []
    if agg_node:
      slice_fns = [slice_fn.maybe_make() for slice_fn in agg_node.slicers]
      for agg_fn in agg_node.fns:
        actual_agg_fn = agg_fn.maybe_make()
        if not isinstance(actual_agg_fn, aggregates.Aggregatable):
          raise ValueError(f'Not an aggregatable: {agg_fn}: {actual_agg_fn}')
        # The output_keys can be a single key or a dict.
        flattened_output_keys = itertools.chain.from_iterable(
            k if isinstance(k, dict) else (k,) for k in agg_fn.output_keys
        )
        agg_fns[tuple(flattened_output_keys)] = actual_agg_fn
    # Collect all the output functions from the output nodes.
    output_fns = []
    for node in output_nodes:
      # The output nodes can be aggregate, uses it as a callable since they are
      # all ran in-process (after the first aggregation node).
      if isinstance(node, AggregateTransform):
        output_fns.append(
            cls.from_transform(node, recursive=False, input_state=None)
        )
      else:
        output_fns.extend([tree_fn.maybe_make() for tree_fn in node.fns])
    return CombinedTreeFn(
        name=name,
        input_fns=input_fns,
        agg_fns=agg_fns,
        slicers=slice_fns,
        output_fns=output_fns,
        input_iterator=data_source,
        num_threads=transform.num_threads,
    )

  @property
  def has_agg(self):
    return True if self.agg_fns else False

  def create_state(self) -> dict[MetricKey, Any]:
    return {
        MetricKey(key): tree_fn.create_state()
        for key, tree_fn in self.agg_fns.items()
    }

  def update_state(
      self, state: dict[MetricKey, Any], inputs: tree.MapLikeTree[Any]
  ) -> dict[MetricKey, Any]:
    """Updates the state by inputs."""

    # All aggregates operate on the same inputs.
    for output_key, tree_agg_fn in self.agg_fns.items():
      try:
        state[MetricKey(output_key)] = tree_agg_fn.update_state(
            state[MetricKey(output_key)], inputs
        )
        if tree_agg_fn.disable_slicing:
          continue
        # state with slicing:
        for slicer in self.slicers:
          for slice_key, masks in slicer.iterate_and_slice(inputs):
            metric_key = MetricKey(output_key, slice_key)
            if metric_key not in state:
              state[metric_key] = tree_agg_fn.create_state()
            tree_agg_fn = tree_agg_fn.with_masks(
                masks,
                replace_mask_false_with=slicer.replace_mask_false_with,
            )
            state[metric_key] = tree_agg_fn.update_state(
                state[metric_key], inputs
            )
      except KeyError as e:
        raise KeyError(
            f'{output_key=} not found from: {list(state.keys())=}'
        ) from e
      except Exception as e:
        raise ValueError(
            f'Falied to update {tree_agg_fn=} with inputs:'
            f' {tree.tree_shape(inputs)}'
        ) from e
    return state

  def merge_states(
      self,
      states: Iterable[dict[MetricKey, Any]],
      strict_states_cnt: int = 0,
  ) -> dict[MetricKey, Any]:
    """Merges multiple states into one.

    Args:
      states: the states to be merged.
      strict_states_cnt: the expected number of states to be merged.

    Returns:
      The merged state.
    """
    states_by_fn = {}
    states_cnt = 0
    for state in states:
      for key, fn_state in state.items():
        if key in states_by_fn:
          agg_fn = self.agg_fns[key.metrics]
          fn_state = agg_fn.merge_states([states_by_fn[key], fn_state])
        states_by_fn[key] = fn_state
      states_cnt += 1
      logging.info('chainable: merged %d states.', states_cnt)
    if strict_states_cnt and states_cnt != strict_states_cnt:
      raise ValueError(
          'chainable: unexpected number of aggregation states. Workers'
          f' might have partially crashed: got {states_cnt} states, '
          f'needs {strict_states_cnt}.'
      )
    return states_by_fn

  def get_result(
      self, state: dict[MetricKey, Any]
  ) -> tree.MapLikeTree[Any] | None:
    """Gets the result from the aggregation state."""
    result = tree.TreeMapView()
    for key, fn_state in state.items():
      outputs = self.agg_fns[key.metrics].get_result(fn_state)
      flattened_keys = key.metrics
      # Only convert str key to MetricKey format when there is slices.
      if key.slice != tree_fns.SliceKey():
        assert isinstance(key.metrics, tuple), f'{key.metrics}'
        flattened_keys = tuple(
            MetricKey(metric, key.slice) for metric in key.metrics
        )
      outputs = tree.TreeMapView(outputs)[key.metrics]
      result = result.copy_and_set(flattened_keys, outputs)
    return functools.reduce(
        lambda inputs, fn: fn(inputs), self.output_fns, result.data
    )

  def _actual_inputs(self, inputs, data_source):
    """Selects the inputs when user provided one."""
    if inputs is not None and data_source is not None:
      raise ValueError(
          'Inputs or input_iterator cannot be set at the same time, got both'
          f' for {self}'
      )
    # Overrides the internal input_iterator if either inputs or input_iterator
    # is provided.
    if inputs is not None:
      return [inputs]
    if data_source is None:
      data_source = self.input_iterator
    if isinstance(data_source, _MergedIterator):
      return data_source
    if self.num_threads and types.is_shardable(data_source):
      data_sources = [
          data_source.shard(i, self.num_threads)
          for i in range(self.num_threads)
      ]
    else:
      data_sources = [data_source]
    return _MergedIterator(data_sources, self.num_threads)

  def __iter__(self):
    return self.iterate()

  def iterate(
      self,
      # TODO: b/395649147 - Rename input_iterator to data_source.
      input_iterator: Iterable[Any] | None = None,
      *,
      with_result: bool = True,
      with_agg_state: bool = True,
      with_agg_result: bool = True,
      state: Any = None,
  ) -> _IteratorWithAggResult[Any]:
    """An iterator runner that takes an input_iterator runs the transform.

    The iterator by default yields the output of all the input_functions before
    the first aggregation function in the chain. Optionally, it can also yield
    the aggregation state and the aggregation result at the end of the
    iteration.

    Args:
      input_iterator: the input iterator.
      with_result: whether to yield the output of running the input_functions.
      with_agg_state: whether to yield the aggregation state at the end of the
        iteration.
      with_agg_result: whether to yield the aggregation result at the end of the
        iteration.
      state: an optional initial aggregation state.

    Returns:
      An iterator that also optionally keeps the aggregation result and state.
    """
    return _IteratorWithAggResult(
        self,
        input_iterator=self._actual_inputs(None, input_iterator),
        with_result=with_result,
        with_agg_state=with_agg_state,
        with_agg_result=with_agg_result,
        state=state or self.create_state(),
    )

  def __call__(self, inputs=None, *, input_iterator=None):
    if (
        not self.agg_fns
        and inputs is None
        and (input_iterator is not None or self.input_iterator is not None)
    ):
      raise ValueError(
          'Non-aggregate transform is not callable with iterator inputs, uses '
          '`iterate()` instead.'
      )
    iter_result = self.iterate(self._actual_inputs(inputs, input_iterator))
    result = mit.last(iter_result)
    if iter_result.agg_result is not None:
      return iter_result.agg_result
    return result


def _eq(a, b):
  try:
    return a == b
  except Exception:  # pylint: disable=broad-exception-caught
    return False

_is_dict = lambda x: isinstance(x, dict)


@dataclasses.dataclass(frozen=True, kw_only=True, eq=False)
class TreeTransform(Generic[TreeFnT]):
  """A lazy transform interface that works on a map like data.

  Each Transform represents a routine operating on a map-like data and returning
  a map-like result. There are following main operations supported:

    * Apply: it applies a routine on the inputs and directly returns the outputs
      from the routine.
    * Assign: it applies a routine on the inputs and assign the result back to
      the inputs with the provided output_keys.
    * Select: it is a routineless Apply that selects some inputs given some
      input keys and optionally with the output_keys.
    * Aggregate: it applies aggregate function(s) on the inputs and outputs the
      aggregation results.

  note: the transform can be chainable by themselves by initializing with an
  input transform. `aggregate()` automatically separates the transform into two
  transforms by assigning the pre-aggregate transform as the input_transform
  of the new AggregateTransform constructed by calling `aggregate()`.

  The following is an example of running an image classification model and
  calculating corresponding metrics:

    ```
    predictions = (
        core.TreeTransform.new()
        # Reading
        .data_source(iterator_make=get_iterator(path))
        # Example pre_processing
        .apply(
            output_keys=('image_bytes', 'label_id', Key.SKIP, 'label_text'),
            fn=parse_proto,
            input_keys='protos',
        )
        # Inference
        .assign(
            'predictions',
            fn=keras.models.load_model(model_path),
            input_keys='image_bytes',
        )
        # Example post_processing
        .assign(
            'pred_id',
            fn=lambda pred: str(np.argmax(pred)),
            input_keys='predictions',
        )
        .assign(
            'label_id',
            fn=str,
            input_keys='label_id',
        # Metrics.
        ).aggregate(
            input_keys=('label_id', 'pred_id'),
            fn=retrieval.TopKRetrievalAggFn(
                metrics=('accuracy', 'precision', 'recall', 'f1_score'),
                input_type=retrieval.InputType.MULTICLASS,
            ),
            output_keys=('accuracy', 'precision', 'recall', 'f1_score'),
        )
        .add_slice('pred_id')  # single feature slice
        .add_slice(('label_id', 'pred_id')) # slice crosses
    )
    ```

  Attributes:
    name: a readable name of the transform.
    input_iterator: the input iterator, cannot coexist with the input_transform.
    input_transform: the transform that outputs the input of this transform.
    fns: the underlying routines of the transforms.
    num_threads: maximum number of threads to run the transform.
    id: the id of the transform, unique for each transform.
    is_noop: whether the transform is a no-op, e.g., no function to execute.
      This is useful at the beginning of a chain by calling `Transform.new()`.
  """

  name: str = ''
  # TODO: b/395649147 - Rename input_iterator to data_source.
  input_iterator: types.MaybeResolvable[Iterable[Any]] | None = None
  input_transform: TreeTransform | None = None
  fns: tuple[TreeFnT, ...] = dataclasses.field(default_factory=tuple)
  num_threads: int = _DEFAULT_NUM_THREADS
  _id: uuid.UUID = dataclasses.field(
      default_factory=uuid.uuid4, init=False, repr=False
  )

  def __post_init__(self):
    if self.input_iterator is not None and self.input_transform is not None:
      raise ValueError(
          'Ambiguous inputs: input_iteartor and input_transform are both set.'
          f'got {self.input_iterator=} and {self.input_transform=}.'
      )
    input_transform = self.input_transform
    if input_transform and input_transform.is_noop:
      # Skip the input_transform if there is a no-op logically.
      input_transform = input_transform.input_transform
    # Uses __setattr__ for frozen dataclasses.
    object.__setattr__(self, 'input_transform', input_transform)
    object.__setattr__(self, '_id', uuid.uuid3(uuid.uuid4(), self.name))

  @classmethod
  def new(
      cls,
      *,
      name: str = '',
      input_iterator: types.MaybeResolvable[Iterable[Any]] | None = None,
      input_transform: TreeTransformT | None = None,
      num_threads: int = _DEFAULT_NUM_THREADS,
  ) -> Self:
    return cls(
        input_transform=input_transform,
        input_iterator=input_iterator,
        name=name,
        num_threads=num_threads,
    )

  @property
  def id(self):
    return self._id

  @property
  def is_noop(self):
    return not self.fns and self.input_iterator is None

  def __hash__(self):
    return hash(self._id)

  def __eq__(self, other: TreeTransform) -> bool:
    return self.id == other.id

  def maybe_replace(self, **kwargs) -> Self:
    filtered = {k: v for k, v in kwargs.items() if not _eq(getattr(self, k), v)}
    return dataclasses.replace(self, **filtered) if filtered else self

  def chain(self, transform: TreeTransform):
    """Chains self to the input of the first node in the other transform."""
    transforms = transform.flatten_transform()
    input_transform = self
    for transform in transforms:
      input_transform = transform.maybe_replace(input_transform=input_transform)
    return input_transform

  def named_transforms(self) -> dict[str, TreeTransform]:
    """Returns a dict of transforms with their names as the keys."""
    result = {}
    # Consecutive transforms with the same name will be considered as the same
    # transform.
    tracked = None
    for transform in self.flatten_transform() + [TreeTransform.new()]:
      if not tracked:
        tracked = transform
        continue
      if transform.name != tracked.name:
        if tracked.name in result:
          raise ValueError(f'Duplicate transform {tracked.name}.')
        result[tracked.name] = tracked
        tracked = transform.maybe_replace(input_transform=None)
      else:
        tracked = transform.maybe_replace(input_transform=tracked)
    return result

  def make(
      self,
      *,
      recursive=True,
      # TODO: b/318463291 - deprecates runner mode in favor named transform.
      mode: RunnerMode = RunnerMode.DEFAULT,
      shard: io.ShardConfig | None = None,
  ) -> CombinedTreeFn:
    """Makes the concrete function instance from the transform."""
    return CombinedTreeFn.from_transform(
        self,
        recursive=recursive,
        mode=mode,
        input_state=shard,
    )

  def data_source(self, iterator: Any = None) -> TreeTransform:
    return TreeTransform.new(
        input_iterator=iterator,
        name=self.name,
        num_threads=self.num_threads,
    )

  @property
  def output_keys(self) -> set[TreeMapKey]:
    """Returns the output_keys (assign_keys for assign) of this transform."""
    result = set()
    for fn in self.fns:
      non_dict_keys, dict_keys = mit.partition(_is_dict, fn.output_keys)
      result.update(itertools.chain(non_dict_keys, *dict_keys))
    return result

  def _check_assign_keys(self, assign_keys: TreeMapKeys):
    """Checks the assign keys are valid."""
    non_dict_keys, dict_keys = mit.partition(_is_dict, assign_keys)
    new_keys = set(itertools.chain(non_dict_keys, *dict_keys))
    output_keys = self.output_keys
    if conflicting_keys := new_keys.intersection(output_keys):
      raise KeyError(
          f'Duplicate output_keys: {conflicting_keys} from assignment of'
          f' {assign_keys}'
      )
    all_keys = output_keys | new_keys
    if tree.Key.SELF in all_keys and len(all_keys) > 1:
      raise KeyError(
          'Cannot mix SELF with other keys as output keys, got'
          f' {assign_keys=} all output keys so far: {output_keys}.'
      )

  def assign(
      self,
      assign_keys: TreeMapKey | TreeMapKeys = (),
      # TODO: b/376296013 - output_keys is deprecated, use assign_keys instead.
      output_keys: TreeMapKey | TreeMapKeys = (),
      *,
      fn: types.MaybeResolvable[Callable[..., Any]] | None = None,
      input_keys: TreeMapKey | TreeMapKeys = tree.Key.SELF,
      fn_batch_size: int = 0,
      batch_size: int = 0,
  ) -> TreeTransform:
    """Assign some key value pairs back to the input mapping."""
    if output_keys:
      raise ValueError(
          '`output_keys` is deprecated, use positional arguments or'
          ' `assign_keys` instead.'
      )
    assign_keys = assign_keys or output_keys
    fn = tree_fns.Assign.new(
        output_keys=assign_keys,
        fn=fn,
        input_keys=input_keys,
        fn_batch_size=fn_batch_size,
        batch_size=batch_size,
    )
    self._check_assign_keys(fn.output_keys)
    return self._maybe_new_transform(fn)

  def select(
      self,
      input_keys: TreeMapKeys,
      output_keys: TreeMapKeys | None = None,
      batch_size: int = 0,
  ) -> TreeTransform:
    output_keys = output_keys or input_keys
    fn = tree_fns.Select.new(
        input_keys=input_keys, output_keys=output_keys, batch_size=batch_size
    )
    return self._maybe_new_transform(fn)

  # TODO: b/356633410 - support rebatching.
  def aggregate(
      self,
      output_keys: TreeMapKey | TreeMapKeys = tree.Key.SELF,
      *,
      fn: types.MaybeResolvable[aggregates.Aggregatable] | None = None,
      input_keys: TreeMapKey | TreeMapKeys = tree.Key.SELF,
      disable_slicing: bool = False,
  ) -> AggregateTransform:
    """Create an aggregate transform on the previous transform."""
    if not fn:
      input_keys, output_keys = (), ()
    return self._maybe_new_agg_transform().add_aggregate(
        output_keys=output_keys,
        fn=fn,
        input_keys=input_keys,
        disable_slicing=disable_slicing,
    )

  def agg(self, **kwargs) -> AggregateTransform:
    """Alias for aggregate."""
    return self.aggregate(**kwargs)

  def apply(
      self,
      *,
      output_keys: TreeMapKey | TreeMapKeys = tree.Key.SELF,
      fn: types.MaybeResolvable[Callable[..., Any]] | None = None,
      input_keys: TreeMapKey | TreeMapKeys = tree.Key.SELF,
      fn_batch_size: int = 0,
      batch_size: int = 0,
  ) -> TreeTransform:
    """Applies a TreeFn on the selected inputs and directly outputs the result."""
    fn = tree_fns.TreeFn.new(
        output_keys=output_keys,
        fn=fn,
        input_keys=input_keys,
        fn_batch_size=fn_batch_size,
        batch_size=batch_size,
    )
    return self._maybe_new_transform(fn)

  def flatten_transform(
      self, remove_input_transform: bool = False
  ) -> list[TreeTransform]:
    """Flatten all the chain of transforms into a list."""
    if self.input_transform is None:
      return [self]
    current_transform = self
    if remove_input_transform:
      current_transform = current_transform.maybe_replace(input_transform=None)
    return self.input_transform.flatten_transform(remove_input_transform) + [
        current_transform
    ]

  def _maybe_new_agg_transform(
      self,
      fn: tree_fns.TreeAggregateFn | None = None,
  ) -> AggregateTransform:
    """Appends a new aggregate while optionally creates a new transform."""
    fns = (fn,) if fn else ()
    if isinstance(self, AggregateTransform):
      if fn:
        self._check_assign_keys(fn.output_keys)
      return dataclasses.replace(self, fns=self.fns + fns)
    elif self.is_noop:
      # assert self.input_transform is None, f'got {self.input_transform=}'
      return AggregateTransform(
          fns=fns,
          name=self.name,
          num_threads=self.num_threads,
          input_transform=self.input_transform,
      )
    else:
      return AggregateTransform(
          input_transform=self,
          name=self.name,
          fns=fns,
          num_threads=self.num_threads,
      )

  def _maybe_new_transform(self, fn) -> TreeTransform:
    """Breaks apart the transform when this is an aggregate or source."""
    if isinstance(self, AggregateTransform):
      if self.name:
        raise ValueError(
            f'Cannot add assign/apply to a named transform {self.name}.'
            'Separate the transforms and connect theme with `chain()`.'
        )
      return TreeTransform(
          input_transform=self,
          fns=(fn,),
          num_threads=self.num_threads,
      )
    return dataclasses.replace(self, fns=self.fns + (fn,))


@dataclasses.dataclass(frozen=True, kw_only=True, eq=False)
class AggregateTransform(TreeTransform[tree_fns.TreeAggregateFn]):
  """An AggregateTransform reduce rows.

  Attributes:
    input_transform: the transform that outputs the input of this transform.
    fns: the underlying routines of the transforms.
  """
  slicers: tuple[tree_fns.Slicer, ...] = ()

  def add_aggregate(
      self,
      *,
      output_keys: TreeMapKey | TreeMapKeys = tree.Key.SELF,
      fn: types.MaybeResolvable[aggregates.Aggregatable],
      input_keys: TreeMapKey | TreeMapKeys = tree.Key.SELF,
      disable_slicing: bool = False,
  ) -> Self:
    """Adds a aggregate and stack it on the existing aggregates."""
    if fn:
      fn = tree_fns.TreeAggregateFn.new(
          output_keys=output_keys,
          fn=fn,
          input_keys=input_keys,
          disable_slicing=disable_slicing,
      )
    return self._maybe_new_agg_transform(fn)

  def add_agg(self, **kwargs) -> Self:
    return self.add_aggregate(**kwargs)

  def add_slice(
      self,
      keys: TreeMapKey | TreeMapKeys,
      slice_name: str | tuple[str, ...] = (),
      slice_fn: tree_fns.SliceIteratorFn | None = None,
      slice_mask_fn: tree_fns.SliceMaskIteratorFn | None = None,
      replace_mask_false_with: Any = tree.DEFAULT_FILTER,
  ) -> 'AggregateTransform':
    """Adds a slice and stack it on the existing slicers.

    This can be used in the following ways:

      * Slice on a single feature: `add_slice('feature')`.
      * Slice crosses with multiple features: `add_slice(('a', 'b')).
      * Slice with arbitrary slicing function that returns an iterable of
        slices:
        `add_slice(('a', 'b'), 'slice_name', slice_fn)`.
      * Multiple slices: `add_slice('a').add_slice('b')`, note that this is not
        the same as `add_slice(('a', 'b'))` that is slice crosses of feature
        'a' and 'b'.
      * Intra-example slicing: `add_slice('a', 'slice_name', slice_mask_fn)`,
        the slice_mask_fn will yield a tuple of slice value and the
        corresponding masks for the aggregation function inputs. The mask is an
        array-like object with the same shape of the to be masked inputs. If
        only one mask is provided, it will be applied to all the inputs. If
        multiple masks are provided, the order of the masks have to match the
        order of the inputs of the aggregations that is configured with this
        slicing. The masking behavior is controlled by `mask_behavior` and
        `replace_mask_false_with`. By default, it only filter out the entries
        with False values in the mask.

    Args:
      keys: input keys for the slicer.
      slice_name: the slice name, default to same as keys, but is required when
        slice_fn or slice_mask_fn is provided.
      slice_fn: optional callable that returns an iterable of slices.
      slice_mask_fn: optional callable that returens an iterable of slice and
        masks pair. The order of the masks have to match the order of the inputs
        of the aggregations that is configured with this slicing.
      replace_mask_false_with: the value to replace the false values in the
        mask. When not set, the maksing behavior is to filter out the entries
        with False values in the mask.

    Returns:
      The AggregateTransform with slices.
    """
    slicer = tree_fns.Slicer.new(
        input_keys=keys,
        slice_fn=slice_fn,
        slice_name=slice_name,
        slice_mask_fn=slice_mask_fn,
        replace_mask_false_with=replace_mask_false_with,
    )
    if slicer.slice_name in set(slicer.slice_name for slicer in self.slicers):
      raise ValueError(f'Duplicate slice name {slicer.slice_name}.')
    return dataclasses.replace(self, slicers=self.slicers + (slicer,))
