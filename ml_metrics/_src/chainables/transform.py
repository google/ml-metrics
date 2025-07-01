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
      TreeTransform()
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

from collections.abc import Callable, Generator, Iterable, Iterator, Mapping, Sequence, Sized
import copy
import dataclasses
import enum
import functools
import itertools
import time
from typing import Any, Generic, Self, TypeVar
import uuid

from absl import logging
import deprecated
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
TreeFnT = TypeVar('TreeFnT', bound=tree_fns.TreeFn)
TreeFn = tree_fns.TreeFn
_ValueT = TypeVar('_ValueT')
_Aggregatable = aggregates.Aggregatable | aggregates.HasAsAggFn

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
  agg_state: _AggState | None


AUTO_SIZE = -1


class _RunnerIterator(iter_utils.MultiplexIterator[_ValueT]):
  """An iterator that returns the last value."""
  agg_state: _AggState

  def __init__(
      self,
      runner: TransformRunner,
      *,
      data_sources: Sequence[Iterable[_ValueT]],
      ignore_error: bool,
      with_result: bool = True,
      with_agg_state: bool = True,
      state: Any = None,
      data_source_size: int = 0,
  ):
    self._runner = runner
    self._ignore_error = ignore_error
    self._with_result = with_result
    # Only keep the states relevant to the runner.
    self.agg_state = {
        k: v for k, v in state.items() if k.metrics in self._runner.agg_fns
    }
    self._with_agg = state and with_agg_state
    self._total = data_source_size
    if data_source_size == AUTO_SIZE:
      total = sum(len(ds) for ds in data_sources if isinstance(ds, Sized))
      if batch_size := runner.transform.batch_size:
        total, residual = divmod(total, batch_size)
        total = total + 1 if residual else total
      self._total = total

    def iter_fn(
        input_iterator: Iterable[tree.TreeLike] = (),
    ) -> Iterator[tree.TreeLike]:
      """Call a chain of functions in sequence."""
      result = input_iterator
      for fn in self._runner.fns:
        fn = dataclasses.replace(fn, ignore_error=self._ignore_error)
        result = fn.iterate(result)
      yield from result

    self.batch_index = 0
    super().__init__(
        data_sources=data_sources,
        iter_fn=iter_fn,
        parallism=self._runner.num_threads,
        name=self._runner.name,
    )

  @property
  def name(self) -> str:
    return self._runner.name

  @property
  def has_agg(self) -> bool:
    return self._runner.has_agg

  @property
  def agg_result(self) -> tree.TreeLike:
    return self._runner.get_result(self.agg_state or {})

  def from_state(self, state: _IteratorState) -> _RunnerIterator:
    return super().from_state(
        state.input_states,
        runner=self._runner,
        ignore_error=self._ignore_error,
        with_result=self._with_result,
        with_agg_state=self._with_agg,
        state=state.agg_state,
    )

  @property
  def state(self) -> _IteratorState:
    return _IteratorState(
        copy.deepcopy(super().state),
        agg_state=copy.deepcopy(self.agg_state),
    )

  def __len__(self) -> int:
    return self._total or self.batch_index

  def __next__(self) -> _ValueT:
    logging.log_every_n_seconds(
        logging.INFO,
        f'"{self.name}" processed {self.batch_index} batch.',
        _LOGGING_INTERVAL_SECS,
    )
    try:
      batch_output = super().__next__()
      self.batch_index += 1
      if self._with_agg:
        self.agg_state = self._runner.update_state(self.agg_state, batch_output)
      logging.debug(
          'chainable: %s', f'"{self.name}" batch cnt {self.batch_index}.'
      )
      return batch_output if self._with_result else None
    except StopIteration as e:
      logging.info(
          'chainable: %s',
          f'"{self.name}" iterator exhausted after {self.batch_index} batches.',
      )
      raise e


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

_AggState = dict[MetricKey, Any]


def clear_cache():
  """Clear the cache for maybe_make."""
  lazy_fns.clear_cache()


@dataclasses.dataclass(frozen=True, kw_only=True)
class TransformRunner(aggregates.Aggregatable, Iterable[_ValueT]):
  """Combining multiple transforms into concrete functions.

  This class encapsulates all in-process logic of a TreeTransform.
  The ordering of the execution is as follows:
    input_iterator -> input_fns (w/ slicers) -> agg_fns -> output_fns.

  From a TreeTransform standpoint, the sequence of TreeTransforms are translated
  and into one function here:
    * fns in `apply()`, `assign()`, `select()`) are translated to `fns`.
    * fns in `aggregate()`, add_slice()` are translated to `agg_fns` and
      `slicers`.
  """

  name: str
  fns: Sequence[tree_fns.TreeFn]
  agg_fns: dict[TreeMapKeys | TreeMapKey, tree_fns.TreeAggregateFn]
  slicers: list[tree_fns.Slicer]
  data_source: Iterable[Any] | None
  num_threads: int
  transform: TreeTransform

  @classmethod
  def from_transform(
      cls,
      transform: TreeTransform,
      *,
      input_state: io.ShardConfig | None,
      agg_only: bool = False,
  ) -> Self:
    """Makes a TreeFn from a transform."""
    # Reset the iterator_node and input_nodes if this is an aggregator.
    if agg_only:
      assert transform.agg_fns, 'Aggregation is required for "Aggregate" mode.'
      transform = transform.maybe_replace(fns=(), data_source_=None)
    name = transform.name or ''
    data_source = transform.data_source_
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
    input_fns = [tree_fn.maybe_make() for tree_fn in transform.fns]
    # Collect all the aggregation functions from the aggregation node.
    agg_fns, slice_fns = {}, []
    if transform.agg_fns:
      slice_fns = [slice_fn.maybe_make() for slice_fn in transform.slicers]
      for agg_fn in transform.agg_fns:
        actual_agg_fn = agg_fn.maybe_make()
        if not isinstance(actual_agg_fn, aggregates.Aggregatable):
          raise ValueError(f'Not an aggregatable: {agg_fn}: {actual_agg_fn}')
        # The output_keys can be a single key or a dict.
        flattened_output_keys = itertools.chain.from_iterable(
            k if isinstance(k, dict) else (k,) for k in agg_fn.output_keys
        )
        agg_fns[tuple(flattened_output_keys)] = actual_agg_fn
    return TransformRunner(
        name=name,
        fns=input_fns,
        agg_fns=agg_fns,
        slicers=slice_fns,
        data_source=data_source,
        num_threads=transform.num_threads,
        transform=transform,
    )

  @property
  def has_agg(self):
    return True if self.agg_fns else False

  def create_state(self) -> _AggState:
    return {
        MetricKey(key): tree_fn.create_state()
        for key, tree_fn in self.agg_fns.items()
    }

  def update_state(
      self, state: _AggState, inputs: tree.TreeLike[Any]
  ) -> _AggState:
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
    return state

  def merge_states(
      self,
      states: Iterable[_AggState],
      strict_states_cnt: int = 0,
  ) -> _AggState:
    """Merges multiple states into one.

    Args:
      states: the states to be merged.
      strict_states_cnt: the expected number of states to be merged.

    Returns:
      The merged state.
    """
    # Iterate over the states and only merge states in the runner.
    states_by_fn = {}
    states_cnt = 0
    for state in states:
      for key, fn_state in state.items():
        if agg_fn := self.agg_fns.get(key.metrics):
          if key in states_by_fn:
            fn_state = agg_fn.merge_states([states_by_fn[key], fn_state])
          states_by_fn[key] = fn_state
      states_cnt += 1
      logging.info('chainable: %s', f'merged {states_cnt} states.')
    if strict_states_cnt and states_cnt != strict_states_cnt:
      raise ValueError(
          'unexpected number of aggregation states. Workers'
          f' might have partially crashed: got {states_cnt} states, '
          f'needs {strict_states_cnt}.'
      )
    return states_by_fn

  def get_result(self, state: _AggState) -> tree.TreeMapView:
    """Gets the result from the aggregation state."""
    result = {}
    for key, fn_state in state.items():
      assert isinstance(key.metrics, tuple), f'{key.metrics}'
      outputs = self.agg_fns[key.metrics].get_result(fn_state)
      flattened_keys = tuple(
          MetricKey(metric, key.slice) for metric in key.metrics
      )
      outputs = tree.TreeMapView(outputs)[key.metrics]
      for k, output in zip(flattened_keys, outputs, strict=True):
        result[k] = output
    # Only removes the slices when there is no slicing at all.
    unwrapped = {(k if k.slice else k.metrics): v for k, v in result.items()}
    result = tree.TreeMapView(key_paths=tuple(unwrapped.keys()))
    return result.copy_and_update(unwrapped)

  def _actual_inputs(self, inputs, data_source) -> list[Iterable[Any]]:
    """Selects the inputs when user provided one."""
    if inputs is not None and data_source is not None:
      raise ValueError(
          'Inputs or input_iterator cannot be set at the same time, got both'
          f' for {self}'
      )
    # Overrides the internal input_iterator if either inputs or input_iterator
    # is provided.
    if inputs is not None:
      # An iterable of a single element.
      return [io.SequenceDataSource([inputs])]
    if data_source is None:
      data_source = self.data_source
    if self.num_threads and types.is_shardable(data_source):
      data_sources = [
          data_source.shard(i, self.num_threads)
          for i in range(self.num_threads)
      ]
    else:
      data_sources = [data_source]
    return data_sources  # pytype: disable=bad-return-type

  def __iter__(self):
    return self.iterate()

  def iterate(
      self,
      data_source: Iterable[Any] | None = None,
      *,
      with_result: bool = True,
      with_agg_state: bool = True,
      state: Any = None,
      ignore_error: bool = False,
      data_source_size: int = 0,
  ) -> _RunnerIterator[_ValueT]:
    """An iterator runner that takes an data_source runs the transform.

    The iterator by default yields the output of all the input_functions before
    the first aggregation function in the chain. Optionally, it can also yield
    the aggregation state and the aggregation result at the end of the
    iteration.

    Args:
      data_source: the input iterator.
      with_result: whether to yield the output of running the input_functions.
      with_agg_state: whether to yield the aggregation state at the end of the
        iteration.
      state: an optional initial aggregation state.
      ignore_error: whether to ignore the error when running the transform.
      data_source_size: the total number of elements in the data_source. 
        If AUTO_LEN, the length of the data_source will be used.

    Returns:
      An iterator that also optionally keeps the aggregation result and state.
    """
    return _RunnerIterator(
        self,
        data_sources=self._actual_inputs(None, data_source),
        ignore_error=ignore_error,
        with_result=with_result,
        with_agg_state=with_agg_state,
        state=state or self.create_state(),
        data_source_size=data_source_size,
    )


class _ChainedRunnerIterator(Iterable[_ValueT]):
  """An iterator that runs a chain of transforms."""

  def __init__(
      self,
      iterators: Iterable[_RunnerIterator[Any]],
      with_result: bool,
      with_agg_state: bool,
      with_agg_result: bool,
      state: Any = None,
  ):
    self._iterators = list(iterators)
    assert all(isinstance(it, _RunnerIterator) for it in self._iterators)
    self._with_result = with_result
    self._prev_ticker = time.time()
    self._with_agg = state and (with_agg_state or with_agg_result)
    self._with_agg_result = with_agg_result

  def maybe_stop(self):
    for it in self._iterators:
      if isinstance(it, types.Stoppable):
        it.maybe_stop()

  def named_iterators(
      self, agg_only: bool = False
  ) -> dict[str, _RunnerIterator[Any]]:
    if agg_only:
      return {r.name: r for r in self._iterators if r.has_agg}
    return {r.name: r for r in self._iterators}

  def __next__(self) -> _ValueT:
    try:
      batch_output = next(self._iterators[-1])
      return batch_output if self._with_result else None
    except StopIteration as e:
      returned = None
      if self._with_agg:
        # This can collect the agg_state and the agg_result at the end of
        # the iteration and return them as the generator return value.
        agg_result = self.agg_result if self._with_agg_result else None
        returned = AggregateResult(agg_result, agg_state=self.agg_state)
      name = ' '.join(it.name for it in self._iterators if it.name)
      logging.info(
          'chainable: %s', f'"{name}" iterator returned a {type(returned)}'
      )
      raise StopIteration(returned) if returned else e

  def __iter__(self) -> Iterator[_ValueT]:
    return self

  def __len__(self) -> int:
    return len(self._iterators[-1])

  @property
  def agg_result(self) -> tree.TreeLike[Any] | None:
    its = self.named_iterators(agg_only=True)
    if not its:
      return None
    it_agg_results = (it.agg_result.items() for it in its.values())
    result = tree.TreeMapView()
    return result.copy_and_update(
        itertools.chain.from_iterable(it_agg_results)
    ).data

  @property
  def agg_state(self) -> _AggState | None:
    agg_iterators = self.named_iterators(agg_only=True)
    if not agg_iterators:
      return None

    it_agg_states = (it.agg_state.items() for it in agg_iterators.values())
    return dict(itertools.chain.from_iterable(it_agg_states))

  @property
  def state(self) -> dict[str, _IteratorState] | _IteratorState:
    result = {it.name: it.state for it in self._iterators}
    if len(result) == 1:
      return next(iter(result.values()))
    return result

  def from_state(self, state: dict[str, _IteratorState] | _IteratorState):
    if isinstance(state, _IteratorState):
      assert len(self._iterators) == 1, f'{len(self._iterators)=}'
      state = {it.name: state for it in self._iterators}
    iterators = [it.from_state(state[it.name]) for it in self._iterators]
    return _ChainedRunnerIterator(
        iterators,
        with_result=self._with_result,
        with_agg_state=self._with_agg,
        with_agg_result=self._with_agg_result,
    )


class ChainedRunner(Iterable[_ValueT]):
  """A runner that runs a chain of transforms."""

  def __init__(self, runners: list[TransformRunner]):
    self._runners = runners
    aggs_cnt = len([r for r in runners if r.has_agg])
    assert len(self.named_aggs) == aggs_cnt, f'{self.named_aggs=}'

  @functools.cached_property
  def named_aggs(self) -> dict[str, TransformRunner]:
    return {r.name: r for r in self._runners if r.has_agg}

  @property
  def data_source(self) -> Iterable[Any] | None:
    return self._runners[0].data_source

  @property
  def has_agg(self) -> bool:
    return True if self.named_aggs else False

  @property
  def agg_fns(self) -> dict[TreeMapKeys | TreeMapKey, tree_fns.TreeAggregateFn]:
    it_agg_fns = (r.agg_fns.items() for r in self.named_aggs.values())
    return dict(itertools.chain.from_iterable(it_agg_fns))

  def create_state(self) -> _AggState:
    it_state = (r.create_state().items() for r in self.named_aggs.values())
    return dict(itertools.chain.from_iterable(it_state))

  def update_state(self, state: _AggState, inputs: Any) -> _AggState | None:
    """Updates the state by inputs."""
    next(it := self.iterate([inputs], state=state))
    return it.agg_state

  def merge_states(
      self,
      states: Iterable[_AggState],
      strict_states_cnt: int = 0,
  ) -> _AggState:
    """Merges multiple states into one."""
    states = list(states)
    if strict_states_cnt and len(states) != strict_states_cnt:
      raise ValueError(
          'unexpected number of aggregation states. Workers'
          f' might have partially crashed: got {len(states)} states, '
          f'needs {strict_states_cnt}.'
      )
    it_state = (
        r.merge_states(states).items() for r in self.named_aggs.values()
    )
    return dict(itertools.chain.from_iterable(it_state))

  def get_result(self, state: _AggState) -> tree.TreeLike[Any] | None:
    it_result = itertools.chain.from_iterable(
        agg_result.items()
        for r in self.named_aggs.values()
        if (agg_result := r.get_result(state))
    )
    return tree.TreeMapView().copy_and_update(it_result).data

  def _actual_inputs(self, inputs, data_source) -> Iterable[Any] | None:
    """Selects the inputs when user provided one."""
    if inputs is not None and data_source is not None:
      raise ValueError(
          'Inputs or input_iterator cannot be set at the same time, got both'
          f' for {self}'
      )
    if inputs is not None:
      return io.SequenceDataSource([inputs])
    if data_source is None:
      data_source = self.data_source
    return data_source

  def iterate(
      self,
      data_source: Iterable[Any] | None = None,
      *,
      with_result: bool = True,
      with_agg_state: bool = True,
      with_agg_result: bool = True,
      state: Any = None,
      ignore_error: bool = False,
      data_source_size: int = 0,
  ) -> _ChainedRunnerIterator[Any]:
    """An iterator runner that takes an data_source runs the transform."""
    iterators = []
    iterator = self._actual_inputs(None, data_source)
    if data_source_size == AUTO_SIZE:
      data_source_size = len(iterator) if isinstance(iterator, Sized) else 0
    for r in self._runners:
      iterator = r.iterate(
          iterator,
          with_agg_state=with_agg_state,
          state=state if r.has_agg else None,
          ignore_error=ignore_error,
          data_source_size=data_source_size,
      )
      data_source_size = len(iterator)
      iterators.append(iterator)
    return _ChainedRunnerIterator(
        iterators,
        with_result=with_result,
        with_agg_state=with_agg_state,
        with_agg_result=with_agg_result,
        state=state or self.create_state(),
    )

  def __iter__(self):
    return self.iterate()

  def __call__(
      self, inputs=None, *, input_iterator=None, ignore_error: bool = False
  ):
    if (
        not self.agg_fns
        and inputs is None
        and (input_iterator is not None or self.data_source is not None)
    ):
      raise ValueError(
          'Non-aggregate transform is not callable with iterator inputs, uses '
          '`iterate()` instead.'
      )
    iter_result = self.iterate(
        data_source=self._actual_inputs(inputs, input_iterator),
        ignore_error=ignore_error,
    )
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

  The following is an example of running an image classification model and
  calculating corresponding metrics:

    ```
    predictions = (
        core.TreeTransform()
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
        ).batch(64)  # normally aggregates takes a batch of inputs.
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
    data_source_: the input iterator, cannot coexist with the input_transform.
    input_transform: the transform that outputs the input of this transform.
    fns: the underlying routines of the transforms.
    agg_fns: the underlying aggregate functions of the transforms.
    slicers: the underlying slicers of the transforms.
    num_threads: maximum number of threads to run the transform.
    id: the id of the transform, unique for each transform.
    is_noop: whether the transform is a no-op, e.g., no function to execute.
      This is useful at the beginning of a chain by calling `Transform()`.
    batch_size: the batch size of the output of the transform, being 0 if not
      explicitly batched (the inputs can still be batched).
  """

  name: str = ''
  data_source_: types.MaybeResolvable[Iterable[Any]] | None = None
  input_transform: TreeTransform | None = None
  fns: tuple[TreeFnT, ...] = ()
  agg_fns: tuple[TreeFnT, ...] = ()
  slicers: tuple[tree_fns.Slicer, ...] = ()
  num_threads: int = _DEFAULT_NUM_THREADS
  _id: uuid.UUID = dataclasses.field(
      default_factory=uuid.uuid4, init=False, repr=False
  )

  def __post_init__(self):
    if self.data_source_ is not None and self.input_transform is not None:
      raise ValueError(
          'Ambiguous inputs: input_iteartor and input_transform are both set.'
          f'got {self.data_source_=} and {self.input_transform=}.'
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
      cls, *, name: str = '', num_threads: int = _DEFAULT_NUM_THREADS
  ) -> Self:
    return cls(name=name, num_threads=num_threads)

  @property
  def id(self) -> uuid.UUID:
    return self._id

  @property
  def is_noop(self) -> bool:
    return not self.agg_fns and not self.fns and self.data_source_ is None

  @property
  def batch_size(self) -> int:
    return self.fns[-1].output_batch_size if self.fns else 0

  def __hash__(self) -> int:
    return hash(self._id)

  def __eq__(self, other: Self) -> bool:
    return self.id == other.id

  def maybe_replace(self, **kwargs) -> Self:
    filtered = {k: v for k, v in kwargs.items() if not _eq(getattr(self, k), v)}
    return dataclasses.replace(self, **filtered) if filtered else self

  # TODO: b/424269199 - deprecates chain in favor of fuse or interleave.
  @deprecated.deprecated('Use "fuse(child)" or "interleave(child)" instead.')
  def chain(self, child: Self) -> Self:
    """Chains self with a child transform, fuses it when name is the same."""
    if child.input_transform is not None or child.data_source_ is not None:
      raise ValueError(
          'Cannot fuse a transform with input_transform or data_source, got'
          f' {child.input_transform=} and {child.data_source=}.'
      )
    if self.name == child.name:
      return self.fuse(child)

    return self.interleave(child)

  def interleave(self, child: Self) -> Self:
    """Behave like Chain, but also interleave the transforms."""
    prev_chains = self.flatten_transform()
    prev_names = set((t.name for t in prev_chains))
    if child.name in prev_names:
      raise ValueError(
          f'Chaining duplicate transform name "{child.name}", {prev_names=}.'
      )
    prev_agg_keys = set(
        itertools.chain.from_iterable(t.agg_output_keys for t in prev_chains)
    )
    if dups := child.agg_output_keys.intersection(prev_agg_keys):
      raise ValueError(
          f'Chaining transforms with duplicate aggregation output keys: {dups}'
      )
    transforms = child.flatten_transform()
    input_transform = self
    for child in transforms:
      input_transform = child.maybe_replace(input_transform=input_transform)
    return input_transform

  def fuse(self, child: Self) -> Self:
    """Behave like Chain, but also fuse all the fns into one transform."""
    if child.is_noop:
      return self
    if self.agg_output_keys.intersection(child.agg_output_keys):
      raise ValueError(
          'Cannot chain a transform with conflicting agg_output_keys'
          f' got {self.agg_output_keys=} and {child.agg_output_keys=}.'
      )
    return self.maybe_replace(
        fns=self.fns + child.fns,
        agg_fns=self.agg_fns + child.agg_fns,
        slicers=self.slicers + child.slicers,
    )

  def named_transforms(self) -> dict[str, Self]:
    """Returns a dict of transforms with their names as the keys."""
    return {
        t.name: t.maybe_replace(input_transform=None)
        for t in self.flatten_transform()
    }

  def make(
      self,
      *,
      recursive: bool = True,
      # TODO: b/318463291 - deprecates runner mode in favor named transform.
      mode: RunnerMode = RunnerMode.DEFAULT,
      shard: io.ShardConfig | None = None,
  ) -> ChainedRunner:
    """Makes the concrete function instance from the transform."""
    transforms = self.flatten_transform()
    if not recursive:
      transforms = [transforms[-1]]
    agg_only = mode == RunnerMode.AGGREGATE
    if agg_only:
      transforms = [t for t in transforms if t.agg_fns]
    runners = []
    for transform in transforms:
      runner = TransformRunner.from_transform(
          transform, agg_only=agg_only, input_state=shard
      )
      runners.append(runner)
    return ChainedRunner(runners)

  def data_source(self, data_source: Any = None, /) -> Self:
    if not self.is_noop:
      raise ValueError(
          f'Cannot add a data source to a non-empty transform, got {self}.'
      )
    return TreeTransform(
        data_source_=data_source,
        name=self.name,
        num_threads=self.num_threads,
    )

  @property
  def output_keys(self) -> set[TreeMapKey]:
    """Returns the output_keys (assign_keys for assign) of this transform."""
    result = set()
    for fn in self.fns:
      non_dict_keys, dict_keys = mit.partition(_is_dict, fn.output_keys)
      new_keys = set(itertools.chain(non_dict_keys, *dict_keys))
      # Assign's output_keys (assign_keys) need to be merged.
      if isinstance(fn, tree_fns.Assign):
        result.update(new_keys)
        continue
      result = new_keys
    return result

  @property
  def agg_output_keys(self) -> set[TreeMapKey]:
    """Returns the output_keys (assign_keys for assign) of this transform."""
    result = set()
    for fn in self.agg_fns:
      non_dict_keys, dict_keys = mit.partition(_is_dict, fn.output_keys)
      result.update(itertools.chain(non_dict_keys, *dict_keys))
    return result

  def _check_assign_keys(
      self,
      assign_keys: TreeMapKeys,
      exisiting_keys: set[TreeMapKeys] | None = None,
      is_aggregate: bool = False,
  ) -> None:
    """Checks the assign keys are valid."""
    non_dict_keys, dict_keys = mit.partition(_is_dict, assign_keys)
    new_keys = set(itertools.chain(non_dict_keys, *dict_keys))
    if exisiting_keys is None:
      exisiting_keys = self.output_keys
    # Allow SELF as aggregate output_keys, but not for assign.
    if not is_aggregate and tree.Key.SELF in new_keys:
      raise KeyError(
          f'Cannot assign to SELF, got {new_keys=}, uses apply instead.'
      )
    if conflicting_keys := new_keys.intersection(exisiting_keys):
      raise KeyError(
          f'Duplicate output_keys: {conflicting_keys} from assignment of'
          f' {assign_keys}'
      )
    all_keys = exisiting_keys | new_keys
    if tree.Key.SELF in all_keys and len(all_keys) > 1:
      raise KeyError(
          'Cannot mix SELF with other keys as output keys, got'
          f' {assign_keys=} all output keys so far: {exisiting_keys}.'
      )

  def filter(
      self,
      fn: Callable[..., bool],
      *,
      input_keys: TreeMapKey | TreeMapKeys = tree.Key.SELF,
  ) -> Self:
    """Filters the input of this transform."""
    assert fn is not None, 'fn must be provided, got None.'
    # The output_keys here is mostly for correct `self.output_keys`, which is
    # not used in FilterFn when filtering.
    fn = tree_fns.FilterFn(
        fn=fn,
        input_keys=input_keys,
        output_keys=tuple(self.output_keys),
        input_batch_size=self.batch_size,
    )
    return self._maybe_new_transform(fn)

  def assign(
      self,
      assign_keys: TreeMapKey | TreeMapKeys = (),
      *,
      fn: types.MaybeResolvable[Callable[..., Any]] | None = None,
      input_keys: TreeMapKey | TreeMapKeys = tree.Key.SELF,
      fn_batch_size: int = 0,
      batch_size: int = 0,
  ) -> Self:
    """Assign some key value pairs back to the input mapping."""
    # TODO: b/413743757 - remove batch_size and fn_batch_size.
    if batch_size or fn_batch_size:
      raise ValueError('batch_size is deprecated, use batch() instead.')
    if not assign_keys:
      raise ValueError(f'Assign should have assign_keys, got {assign_keys=}')

    fn = tree_fns.Assign(
        output_keys=assign_keys,
        fn=fn,
        input_keys=input_keys,
        input_batch_size=self.batch_size,
    )
    self._check_assign_keys(fn.output_keys)
    return self._maybe_new_transform(fn)

  def select(
      self,
      input_keys: tuple[TreeMapKey, ...] | TreeMapKey,
      *other_input_keys: TreeMapKey,
      output_keys: TreeMapKeys | None = None,
      batch_size: int = 0,
  ) -> Self:
    """Selects the input of this transform."""
    # TODO: b/413743757 - remove batch_size.
    if batch_size:
      raise ValueError('batch_size is deprecated, use batch() instead.')
    if isinstance(input_keys, Mapping):
      raise TypeError(f'illegal mapping for select op: {input_keys=}')

    input_keys = tree.normalize_keys(input_keys) + other_input_keys
    output_keys = output_keys or input_keys
    fn = tree_fns.Select(
        input_keys=input_keys,
        output_keys=output_keys,
        input_batch_size=self.batch_size,
    )
    return self._maybe_new_transform(fn)

  def batch(
      self,
      batch_size: int,
      *,
      batch_fn: Callable[..., Any] | None = None,
  ) -> Self:
    """Batches the input of this transform.

    This batches single element to a list of elements. E.g., for inputs of
    [1, 2, 3], given `p = TreeTransform().batch(2)`, `p.make().iterate(inputs)`
    yields [[1, 2], [3]]. This also works for multiple output operations where
    multiple output_keys are provided. E.g., with a dict inputs of [{'a': 1,
    'b': 2}, {'a': 3, 'b': 4}], given `p = TreeTransform().select(('a',
    'b')).batch(2)`, `p.make().iterate(inputs)` yields [{'a': [1, 3], 'b': [2,
    4]}].

    Args:
      batch_size: the batch size.
      batch_fn: the batch function that takes a tuple of inputs and returns a
        tuple of batched outputs.

    Returns:
      A TreeTransform that batches the input of this transform.
    """
    if batch_fn is None:
      batch_fn = lambda x: [x]
    fn = lambda *args: tuple(batch_fn(arg) for arg in args)
    keys = tuple(self.output_keys) or tree.Key.SELF
    fn = tree_fns.TreeFn(
        input_batch_size=self.batch_size,
        batch_size=batch_size,
        fn=fn,
        input_keys=keys,
        output_keys=keys,
    )
    return self._maybe_new_transform(fn)

  def sink(
      self,
      sink: types.MaybeResolvable[types.SinkT],
      *,
      input_keys: TreeMapKeys | TreeMapKey = tree.Key.SELF,
  ) -> Self:
    """Sinks the input of this transform.

    The sink function need to implement the `write` and `close` methods. When
    input_keys are provided, the selected inputs will be fed to the `sink.write`
    method as positional or keyword arguements depending on whether the
    input_keys are list of keys or dict keys. The original inputs then are
    then forwarded after each write without any changes. When the iteration is
    done, the `sink.close` method will be called.

    Args:
      sink: the sink function.
      input_keys: the input keys to be fed to the sink.

    Returns:
      A TreeTransform that sinks the input of this transform.
    """
    # The output_keys here is for `self.output_keys`, which is not used in Sink.
    fn = tree_fns.Sink(
        fn=sink,
        input_keys=input_keys,
        output_keys=input_keys,
        input_batch_size=self.batch_size,
    )
    return self._maybe_new_transform(fn)

  # TODO: b/356633410 - support rebatching.
  def aggregate(
      self,
      fn: types.MaybeResolvable[_Aggregatable] | None = None,
      *,
      input_keys: TreeMapKey | TreeMapKeys = tree.Key.SELF,
      output_keys: TreeMapKey | TreeMapKeys = '',
      disable_slicing: bool = False,
  ) -> Self:
    """Create an aggregate transform on the previous transform."""
    if self.agg_fns:
      raise ValueError('Cannot have more than one aggregations.')
    if not fn:
      input_keys, output_keys = (), ()
    return self.add_aggregate(
        output_keys=output_keys,
        fn=fn,
        input_keys=input_keys,
        disable_slicing=disable_slicing,
    )

  def agg(
      self,
      fn: types.MaybeResolvable[_Aggregatable] | None = None,
      *,
      input_keys: TreeMapKey | TreeMapKeys = tree.Key.SELF,
      output_keys: TreeMapKey | TreeMapKeys = '',
      disable_slicing: bool = False,
  ) -> Self:
    """Alias for aggregate."""
    return self.aggregate(
        fn,
        input_keys=input_keys,
        output_keys=output_keys,
        disable_slicing=disable_slicing,
    )

  def apply(
      self,
      fn: types.MaybeResolvable[Callable[..., Any]] | None = None,
      *,
      output_keys: TreeMapKey | TreeMapKeys = tree.Key.SELF,
      input_keys: TreeMapKey | TreeMapKeys = tree.Key.SELF,
      fn_batch_size: int = 0,
      batch_size: int = 0,
  ) -> Self:
    """Applies a TreeFn on the selected inputs and directly outputs the result."""
    # TODO: b/413743757 - remove batch_size and fn_batch_size.
    if batch_size or fn_batch_size:
      raise ValueError('(fn_)batch_size is deprecated, use batch() instead.')

    if fn or not (input_keys == output_keys and input_keys is tree.Key.SELF):
      fn = tree_fns.TreeFn(
          output_keys=output_keys,
          fn=fn,
          input_keys=input_keys,
          input_batch_size=self.batch_size,
      )
    return self._maybe_new_transform(fn)

  def flatten_transform(self, split_agg: bool = False) -> list[Self]:
    """Flatten all the chain of transforms into a list."""
    if self.input_transform is not None:
      ancestors = self.input_transform.flatten_transform()
      current = self.maybe_replace(input_transform=None)
      return ancestors + current.flatten_transform(split_agg=split_agg)

    if self.is_noop:
      return []

    if not split_agg:
      return [self]

    t = self.maybe_replace(agg_fns=(), slicers=()).flatten_transform()
    t_agg = self.maybe_replace(fns=(), data_source_=None).flatten_transform()
    return t + t_agg

  def _maybe_new_agg_transform(
      self,
      fn: tree_fns.TreeAggregateFn | None = None,
  ) -> Self:
    """Appends a new aggregate while optionally creates a new transform."""
    if not fn:
      return self
    self._check_assign_keys(fn.output_keys, self.agg_output_keys, True)
    agg_fns = self.agg_fns + (fn,)
    return dataclasses.replace(self, agg_fns=agg_fns)

  def _maybe_new_transform(self, fn: tree_fns.TreeFn) -> Self:
    """Breaks apart the transform when this is an aggregate or source."""
    if not fn:
      return self
    if self.agg_fns:
      raise ValueError(
          f'Aggregation has to be the last node, attempting to add {fn}'
      )
    return self.maybe_replace(fns=self.fns + (fn,))

  def add_aggregate(
      self,
      fn: types.MaybeResolvable[_Aggregatable],
      *,
      output_keys: TreeMapKey | TreeMapKeys = '',
      input_keys: TreeMapKey | TreeMapKeys = tree.Key.SELF,
      disable_slicing: bool = False,
  ) -> Self:
    """Adds a aggregate and stack it on the existing aggregates."""
    if fn:
      fn = tree_fns.TreeAggregateFn(
          output_keys=output_keys,
          fn=fn,
          input_keys=input_keys,
          disable_slicing=disable_slicing,
          input_batch_size=self.batch_size,
      )
    return self._maybe_new_agg_transform(fn)

  def add_agg(
      self,
      fn: types.MaybeResolvable[_Aggregatable],
      *,
      output_keys: TreeMapKey | TreeMapKeys = '',
      input_keys: TreeMapKey | TreeMapKeys = tree.Key.SELF,
      disable_slicing: bool = False,
  ) -> Self:
    """Alias for add_aggregate."""
    return self.add_aggregate(
        fn,
        output_keys=output_keys,
        input_keys=input_keys,
        disable_slicing=disable_slicing,
    )

  def add_slice(
      self,
      keys: TreeMapKey | TreeMapKeys,
      slice_name: str | tuple[str, ...] = (),
      slice_fn: tree_fns.SliceIteratorFn | None = None,
      replace_mask_false_with: Any = tree.DEFAULT_FILTER,
  ) -> Self:
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
      * Intra-example slicing: deprecated, do not use.

    Args:
      keys: input keys for the slicer.
      slice_name: the slice name, default to same as keys, but is required when
        slice_fn is provided.
      slice_fn: optional callable that returns an iterable of slices.
      replace_mask_false_with: the value to replace the false values in the
        mask. When not set, the maksing behavior is to filter out the entries
        with False values in the mask.

    Returns:
      The TreeTransform with slices.
    """
    if not self.agg_fns:
      raise ValueError(f'Cannot add slice without aggregate, {self.agg_fns=}')
    slicer = tree_fns.Slicer.new(
        input_keys=keys,
        slice_fn=slice_fn,
        slice_name=slice_name,
        replace_mask_false_with=replace_mask_false_with,
    )
    if slicer.slice_name in set(slicer.slice_name for slicer in self.slicers):
      raise ValueError(f'Duplicate slice name {slicer.slice_name}.')
    return dataclasses.replace(self, slicers=self.slicers + (slicer,))


# TODO: b/404264788 - remove this alias.
AggregateTransform = TreeTransform
