# Copyright 2024 Google LLC
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
import dataclasses
import inspect
import itertools
import queue
import time
from typing import Any, Generic, TypeVar

from absl import logging
from ml_metrics._src import base_types
from ml_metrics._src.aggregates import base as aggregates
from ml_metrics._src.chainables import lazy_fns
from ml_metrics._src.chainables import tree
from ml_metrics._src.chainables import tree_fns

TreeMapKey = tree.TreeMapKey
TreeMapKeys = tree.TreeMapKeys

Fn = Callable[..., Any]
FnT = TypeVar('FnT', Callable, aggregates.Aggregatable)
TreeFnT = TypeVar('TreeFnT', bound=tree_fns.TreeFn)
TreeMapKeyT = TypeVar('TreeMapKeyT', bound=TreeMapKey)
Map = Mapping[TreeMapKeyT, Any]
TreeTransformT = TypeVar('TreeTransformT', bound='TreeTransform')
TreeFn = tree_fns.TreeFn

_LOGGING_INTERVAL_SECS = 60


def is_stop_iteration(inputs) -> bool:
  """Returns whether the inputs is a StopIteration."""
  return isinstance(inputs, StopIteration)


class PrefetchableIterator:
  """An iterator that can also prefetch before iterated."""

  def __init__(self, generator, prefetch_size: int = 2):
    self._data = queue.SimpleQueue()
    self._returned = None
    if not inspect.isgenerator(generator):
      generator = iter(generator)
    self._generator = generator
    self._exceptions = []
    self._prefetch_size = prefetch_size
    self._exhausted = False
    self._error_cnt = 0
    self._cnt = 0
    self._data_size = 0

  @property
  def cnt(self) -> int:
    return self._cnt

  @property
  def returned(self) -> Any:
    assert (
        self._exhausted
    ), 'Generator is not exhausted, returned is not available.'
    return self._returned

  @property
  def data_size(self) -> int:
    return self._data_size

  @property
  def exhausted(self) -> bool:
    return self._exhausted

  @property
  def exceptions(self) -> list[Exception]:
    return self._exceptions

  def flush_prefetched(self, batch_size: int = 0) -> list[Any]:
    """Flushes the prefetched data.

    Args:
      batch_size: the batch size of the data to be flushed. If batch_size = 0,
        it takes all prefetche immediately.

    Returns:
      The flushed data.
    """
    result = []
    while (
        self.data_size < batch_size
        and not self._exhausted
        and not self._exceptions
    ):
      time.sleep(0)
    batch_size = batch_size or self.data_size
    while self.data_size and len(result) < batch_size:
      result.append(self._data.get())
      self._data_size -= 1
    logging.info('Chainables: flush_prefetched: %s', len(result))
    return result

  def __next__(self):
    self.prefetch(1)
    if not self._data.empty():
      return self._data.get()
    else:
      logging.info('chainables: Generator exhausted from %s.', self._generator)
      raise StopIteration(self._returned)

  def __iter__(self):
    return self

  def prefetch(self, num_items: int = 0):
    """Prefeches items from the undelrying generator."""
    while not self._exhausted and self._data.qsize() < (
        num_items or self._prefetch_size
    ):
      try:
        self._data.put(next(self._generator))
        self._cnt += 1
        self._data_size += 1
      except StopIteration as e:
        self._exhausted = True
        self._returned = e.value
        logging.info(
            'chainables: prefetch exhausted after %d items.', self._cnt
        )
      except Exception as e:  # pylint: disable=broad-exception-caught
        logging.exception('chainables: Got error during prefetch.')
        if 'generator already executing' != str(e):
          self._exceptions.append(e)
          if len(self._exceptions) > 3:
            logging.exception('chainables: Too many errors, stop prefetching.')
            break

        time.sleep(0)


def _call_fns(
    fns: Sequence[tree.MapLikeTreeCallable | tree_fns.TreeAggregateFn],
    inputs: tree.MapLikeTree | None = None,
) -> tree.MapLikeTree | None:
  """Call a chain of functions in sequence."""
  result = inputs
  for fn in fns:
    try:
      result = fn(result)
    except Exception as e:  # pylint: disable=too-broad-exception
      raise ValueError(
          f'Falied to execute {fn} with inputs: {tree.tree_shape(result)}'
      ) from e

  return result


class RunnerMode(base_types.StrEnum):
  DEFAULT = 'default'
  AGGREGATE = 'aggregate'
  SAMPLE = 'sample'
  SEPARATE = 'separate'


@dataclasses.dataclass(kw_only=True, frozen=True)
class AggregateResult:
  agg_state: Any = None
  agg_result: Any = None


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


def iterate_with_returned(
    generator: Generator[Any, None, AggregateResult],
) -> Iterator[AggregateResult]:
  """Converts a generator to an iterator with returned value as the last."""
  returned = yield from generator
  yield returned


def get_generator_returned(
    generator: Generator[Any, None, AggregateResult],
) -> AggregateResult | None:
  """Returns the aggregate result by from a TreeTransform based generator."""
  result = AggregateResult()
  for result in iterate_with_returned(generator):
    pass
  return result


@dataclasses.dataclass(frozen=True, kw_only=True)
class CombinedTreeFn:
  """Combining multiple transforms into concrete functions.

  This class encapsulates all in-process logic of a TreeTransform.
  The ordering of the execution is as follows:
    input_iterator -> input_fns (w/ slicers) -> agg_fns -> output_fns.

  From a TreeTransform standpoint, the sequence of TreeTransforms are translted
  and into one function here:
    * `IteratorSource` is translated to `input_iterator`
    * Base `TreeTrsnform`s (`apply()`, `assign()`, `select()`) are translated to
      `input_fns` in sequence.
    * The first `AggregateTransform` in the chain is translated to `slicers` and
      `agg_fns`.
    * Any transforms after the first `AggregateTransform` in the chain is
      converted to a callable and translated as `output_fns`.
  """

  input_fns: Sequence[tree.MapLikeTreeCallable] = ()
  agg_fns: dict[TreeMapKeys | TreeMapKey, tree_fns.TreeAggregateFn] = (
      dataclasses.field(default_factory=dict)
  )
  slicers: list[tree_fns.Slicer] = dataclasses.field(default_factory=list)
  output_fns: Sequence[tree.MapLikeTreeCallable] = ()
  input_iterator: Iterable[Any] | None = None

  @classmethod
  def from_transform(
      cls,
      transform: TreeTransform,
      recursive: bool = True,
      mode: RunnerMode = RunnerMode.DEFAULT,
  ):
    """Builds a TreeFn from a transform and optionally its input transforms."""
    # Flatten the transform into nodes and find the first aggregation node as
    # the aggregation node for the concrete function. Any node before it is
    # input nodes, any node after it regardless of its type (aggregation or not)
    # is output_node.
    transforms = transform.flatten_transform() if recursive else [transform]
    input_nodes, output_nodes = [], []
    iterator_node, agg_node = None, None
    for node in transforms:
      if isinstance(node, IteratorSource):
        iterator_node = node
      elif agg_node is None and isinstance(node, AggregateTransform):
        agg_node = node
      elif agg_node is None:
        input_nodes.append(node)
      else:
        output_nodes.append(node)

    # Reset the iterator_node and input_nodes if this is an aggregator.
    if mode == RunnerMode.AGGREGATE:
      iterator_node = None
      input_nodes = []
      assert agg_node, 'No aggregation is required for "Aggregate" mode.'
    elif mode == RunnerMode.SAMPLE:
      agg_node = None
      output_nodes = []

    # Collect input_iterator.
    input_iterator = lazy_fns.maybe_make(
        iterator_node and iterator_node.iterator
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
        agg_fns[agg_fn.output_keys] = actual_agg_fn
        if not isinstance(actual_agg_fn, aggregates.Aggregatable):
          raise ValueError(f'Not an aggregatable: {agg_fn}: {actual_agg_fn}')
    # Collect all the output functions from the output nodes.
    output_fns = []
    for node in output_nodes:
      # the output nodes can be aggregate, uses it as a callable since they are
      # all ran in-process (after the first aggregation node).
      if isinstance(node, AggregateTransform):
        output_fns.append(cls.from_transform(node, recursive=False))
      else:
        output_fns.extend([tree_fn.maybe_make() for tree_fn in node.fns])
    return cls(
        input_fns=input_fns,
        agg_fns=agg_fns,
        slicers=slice_fns,
        output_fns=output_fns,
        input_iterator=input_iterator,
    )

  @property
  def has_agg(self):
    return True if self.agg_fns else False

  def create_state(self) -> dict[MetricKey, Any]:
    return {
        MetricKey(key): tree_fn.create_state()
        for key, tree_fn in self.agg_fns.items()
    }

  def _update_state(
      self, state: dict[MetricKey, Any], inputs: tree.MapLikeTree[Any]
  ) -> dict[MetricKey, Any]:
    """Updates the state by inputs."""

    # All aggregates operate on the same inputs.
    for metric_name, tree_agg_fn in self.agg_fns.items():
      try:
        state[MetricKey(metric_name)] = tree_agg_fn.update_state(
            state[MetricKey(metric_name)], inputs
        )
        # state with slicing:
        for slicer in self.slicers:
          for slice_key, masks in slicer.iterate_and_slice(inputs):
            metric_key = MetricKey(metric_name, slice_key)
            if metric_key not in state:
              state[metric_key] = tree_agg_fn.create_state()
            tree_agg_fn = dataclasses.replace(tree_agg_fn, masks=masks)
            state[metric_key] = tree_agg_fn.update_state(
                state[metric_key], inputs
            )
      except KeyError as e:
        raise KeyError(
            f'{metric_name=} not found from: {list(state.keys())=}'
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
      logging.info('chainables: merged %d states.', states_cnt)
    if strict_states_cnt and states_cnt != strict_states_cnt:
      raise ValueError(
          'chainables: unexpected number of aggregation states. Workers'
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
      outputs = self.agg_fns[key.metrics].actual_fn.get_result(fn_state)
      flattened_keys = key.metrics
      if key.slice != tree_fns.SliceKey():
        assert isinstance(key.metrics, tuple), f'{key.metrics}'
        flattened_keys = tuple(
            MetricKey(metric, key.slice) for metric in key.metrics
        )
      result = result.copy_and_set(flattened_keys, outputs)
    result = result.data
    return _call_fns(self.output_fns, result)

  def _actual_inputs(self, inputs, input_iterator):
    if inputs is not None and input_iterator is not None:
      raise ValueError(
          'Inputs or input_iterator cannot be set at the same time, got both'
          f' for {self}'
      )
    # Overrides the internal input_iterator if either inputs or input_iterator
    # is provided.
    if input_iterator is None and self.input_iterator is None:
      return [inputs]
    else:
      return input_iterator or self.input_iterator

  def iterate(
      self,
      input_iterator: Iterable[Any] | None = None,
      *,
      with_result: bool = True,
      with_agg_state: bool = False,
      with_agg_result: bool = False,
      state: Any = None,
  ) -> Generator[Any, None, AggregateResult | None]:
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

    Yields:
      The output of the input_fns, optionally the aggregation state and the
      aggregation result.
    """
    input_iterator = self._actual_inputs(None, input_iterator)
    state = state or self.create_state()
    with_agg_state = with_agg_state or with_agg_result
    prev_ticker = time.time()
    batch_index = -1
    for batch_index, batch in enumerate(input_iterator):
      if (ticker := time.time()) - prev_ticker > _LOGGING_INTERVAL_SECS:
        logging.info('chainables: calculating for batch %d.', batch_index)
        prev_ticker = ticker
      batch_output = _call_fns(self.input_fns, batch)
      yield batch_output if with_result else None
      if with_agg_state:
        state = self._update_state(state, batch_output)
    # This can collect the agg_state and the agg_result at the end of
    # the iteration and return them as the generator return value.
    agg_result = self.get_result(state) if with_agg_result else None
    if with_agg_state:
      logging.info(
          'chainables: returns aggregation after %d batches.',
          batch_index + 1,
      )
      return AggregateResult(agg_state=state, agg_result=agg_result)

  def update_state(
      self,
      state: Any = None,
      inputs=None,
      *,
      input_iterator: Iterable[Any] | None = None,
  ):
    """Updates the state by either the inputs or an iterator of the inputs."""
    input_iterator = self._actual_inputs(inputs, input_iterator)
    agg_result = get_generator_returned(
        self.iterate(input_iterator, with_agg_state=True, state=state)
    )
    assert agg_result is not None
    return agg_result.agg_state

  def __call__(self, inputs=None, *, input_iterator=None):
    iter_input = self._actual_inputs(inputs, input_iterator)
    if self.agg_fns:
      return self.get_result(self.update_state(input_iterator=iter_input))
    else:
      result = list(self.iterate(iter_input))
      # Directly returns the result when inputs (vs. iterator) is fed.
      if not input_iterator and not self.input_iterator:
        return result[0]
      return result


@dataclasses.dataclass(frozen=True, kw_only=True)
class TreeTransform(Generic[TreeFnT]):
  """A lazy transform interface that works on a map like data.

  Each Transform represents a routine operating on a map-like data and returning
  a map-like result. There are following main operations suppoted:
    * Apply: it applies a routine on the inputs and directly returns the ouputs
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

  Attributes:
    input_transform: the transform that outputs the input of this transform.
    fns: the underlying routines of the transforms.
    is_noop: whether the transform is a no-op, e.g., no function to execute.
      This is useful at the beginning of a chain by calling `Transform.new()`.
  """

  input_transform: TreeTransform | None = None
  fns: list[TreeFnT] = dataclasses.field(default_factory=list)
  _default_constructor: bool = dataclasses.field(default=True, repr=False)

  def __post_init__(self):
    if self._default_constructor:
      raise ValueError(
          f'Do not use default constructor {self.__class__.__name__}, uses'
          ' "new()" instead.'
      )

  @classmethod
  def new(cls, *, input_transform: TreeTransformT | None = None):
    if input_transform and input_transform.is_noop:
      # Skip the input_transform if there is a no-op logically.
      input_transform = input_transform.input_transform
    return cls(input_transform=input_transform, _default_constructor=False)

  @property
  def is_noop(self):
    return not self.fns

  def chain(self, transform: TreeTransform):
    """Chains self to the input of the first node in the other transform."""
    transforms = transform.flatten_transform()
    input_transform = self
    for transform in transforms:
      input_transform = dataclasses.replace(
          transform, input_transform=input_transform
      )
    return input_transform

  def make(
      self, *, recursive=True, mode: RunnerMode = RunnerMode.DEFAULT
  ) -> CombinedTreeFn:
    """Makes the concrete function instance from the transform."""
    return CombinedTreeFn.from_transform(self, recursive=recursive, mode=mode)

  def data_source(self, iterator: Any = None) -> IteratorSource:
    if self.input_transform:
      raise ValueError(
          f'Cannot add data_source to {self} when there is already an'
          ' input_transform.'
      )
    return IteratorSource.new(iterator)

  def assign(
      self,
      output_keys: TreeMapKey | TreeMapKeys = (),
      *,
      fn: lazy_fns.LazyFn | Callable[..., Any] | None = None,
      input_keys: TreeMapKey | TreeMapKeys = tree.Key.SELF,
  ) -> TreeTransform:
    """Assign some key value pairs back to the input mapping."""
    fn = tree_fns.Assign.new(
        output_keys=output_keys,
        fn=fn,
        input_keys=input_keys,
    )
    fns = {}
    for fn_ in self.fns:
      fns.update({key: fn_ for key in fn_.output_keys})
    if conflicting_keys := set(fn.output_keys).intersection(fns):
      raise ValueError(
          f'Duplicate output_keys: {conflicting_keys} from assignment of'
          f' {fn.output_keys}'
      )
    if tree.Key.SELF in fns:
      raise ValueError(
          f'Cannot add new key {fn.output_keys} when other aggregate output is'
          f' SELF: {fns[tree.Key.SELF]}'
      )
    return self._maybe_new_transform(fn)

  def select(
      self, input_keys: TreeMapKeys, output_keys: TreeMapKeys | None = None
  ) -> TreeTransform:
    output_keys = output_keys or input_keys
    fn = tree_fns.Select.new(input_keys=input_keys, output_keys=output_keys)
    return self._maybe_new_transform(fn)

  def aggregate(
      self,
      output_keys: TreeMapKey | TreeMapKeys = tree.Key.SELF,
      *,
      fn: lazy_fns.LazyFn | aggregates.Aggregatable | None = None,
      input_keys: TreeMapKey | TreeMapKeys = tree.Key.SELF,
  ) -> AggregateTransform:
    return AggregateTransform.new(input_transform=self).add_aggregate(
        input_keys=input_keys, output_keys=output_keys, fn=fn
    )

  def agg(self, **kwargs) -> AggregateTransform:
    """Alias for aggregate."""
    return self.aggregate(**kwargs)

  def apply(
      self,
      *,
      output_keys: TreeMapKey | TreeMapKeys = tree.Key.SELF,
      fn: lazy_fns.LazyFn | Callable[..., Any] | None = None,
      input_keys: TreeMapKey | TreeMapKeys = tree.Key.SELF,
  ) -> TreeTransformT:
    """Applys a TreeFn on the selected inputs and directly outputs the result."""
    fn = tree_fns.TreeFn.new(
        output_keys=output_keys,
        fn=fn,
        input_keys=input_keys,
    )
    return self._maybe_new_transform(fn)

  def flatten_transform(self):
    """Flatten all the chain of transforms into a list."""
    if not self.input_transform:
      return [self]
    return self.input_transform.flatten_transform() + [self]

  def _maybe_new_transform(self, fn):
    # Break apart the transform when this is an aggregate transform.
    if isinstance(self, AggregateTransform):
      return TreeTransform(
          input_transform=self, fns=[fn], _default_constructor=False
      )
    elif isinstance(self, IteratorSource):
      return TreeTransform(
          input_transform=self, fns=[fn], _default_constructor=False
      )
    else:
      return dataclasses.replace(self, fns=self.fns + [fn])


@dataclasses.dataclass(frozen=True, kw_only=True)
class IteratorSource(TreeTransform):
  """A source that wraps around an iterator."""

  # Any iterable or an instance that can make an iterable either from the
  # registered lazy_fns.makeables or from an implemented LazyFn.
  iterator: Iterable[Any] | lazy_fns.LazyFn | Any

  def __post_init__(self):
    if self.input_transform:
      raise ValueError(
          f'Source cannot have an input_transform, got {self.input_transform}'
      )
    if self.fns:
      raise ValueError(f'Source must not have any fn, got {self.fns}')

  @classmethod
  def new(cls, iterator: Any):
    return cls(iterator=iterator, _default_constructor=False)

  @property
  def is_noop(self):
    return not self.iterator


@dataclasses.dataclass(frozen=True, kw_only=True)
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
      fn: lazy_fns.LazyFn | aggregates.Aggregatable | None = None,
      input_keys: TreeMapKey | TreeMapKeys = tree.Key.SELF,
  ) -> 'AggregateTransform':
    """Adds a aggregate and stack it on the existing aggregates."""
    fn = tree_fns.TreeAggregateFn.new(
        output_keys=output_keys,
        fn=fn,
        input_keys=input_keys,
    )
    agg_fns = {}
    for fn_ in self.fns:
      agg_fns.update({key: fn_ for key in fn_.output_keys})
    if conficting_keys := set(fn.output_keys).intersection(agg_fns):
      raise ValueError(
          f'Duplicate output_keys {conficting_keys} from aggregate output keys'
          f' {fn.output_keys}.'
      )
    agg_fns.update({key: fn for key in fn.output_keys})
    if tree.Key.SELF in agg_fns and len(agg_fns) > 1:
      raise ValueError(
          f'Cannot add new key {fn.output_keys} when other aggregate output is'
          f' SELF: {agg_fns[tree.Key.SELF]}'
      )
    return dataclasses.replace(self, fns=self.fns + [fn])

  def add_slice(
      self,
      keys: TreeMapKey | TreeMapKeys,
      slice_name: str | tuple[str, ...] | None = None,
      slice_fn: tree_fns.SliceIteratorFn | None = None,
      slice_mask_fn: tree_fns.SliceMaskIteratorFn | None = None,
      mask_behavior: tree.MaskBehavior = tree.MaskBehavior.FILTER,
      mask_replace_false_with: Any = None,
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
        `mask_replace_false_with`. By default, it only filter out the entries
        with False values in the mask.

    Args:
      keys: input keys for the slicer.
      slice_name: the slice name, default to same as keys, but is required when
        slice_fn or slice_mask_fn is provided.
      slice_fn: optional callable that returns an iterable of slices.
      slice_mask_fn: optional callable that returens an iterable of slice and
        masks pair. The order of the masks have to match the order of the inputs
        of the aggregations that is configured with this slicing.
      mask_behavior: give a boolean mask, whether to filter out or replace the
        items corresponding to the false values in the mask.
      mask_replace_false_with: the value to replace the false values in the
        mask.

    Returns:
      The AggregateTransform with slices.
    """
    slice_name = slice_name or keys
    slicer = tree_fns.Slicer.new(
        input_keys=keys,
        slice_fn=slice_fn,
        slice_name=slice_name,
        slice_mask_fn=slice_mask_fn,
        mask_behavior=mask_behavior,
        mask_replace_false_with=mask_replace_false_with,
    )
    if slicer.slice_name in set(slicer.slice_name for slicer in self.slicers):
      raise ValueError(f'Duplicate slice name {slicer.slice_name}.')
    return dataclasses.replace(self, slicers=self.slicers + (slicer,))
