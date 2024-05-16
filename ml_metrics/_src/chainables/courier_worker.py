# Copyright 2024 Google LLC
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
"""Courier worker that can take and run a registered makeable instance."""

from collections.abc import Callable, Iterator
from concurrent import futures
import dataclasses
import time
from typing import Any

from absl import logging
import courier
from ml_metrics._src.chainables import lazy_fns
from ml_metrics._src.chainables import transform


_LOGGING_INTERVAL_SEC = 60
_NUM_TOTAL_FAILURES_THRESHOLD = 60
picklers = lazy_fns.picklers


@dataclasses.dataclass(kw_only=True, frozen=True)
class Task:
  """Lazy function that runs on courier methods.

  Attributes:
    courier_method: The courier method name to be called.
    args: The positional arguments later used to pass in the fn.
    kwargs: The named arguments later used to pass in the fn.
    blocking: Whether the wait for the call to complete.
    keep_result: Whether to keep the result.
    server_name: The server address this task is sent to.
    parent_task: The parent task that has to be run first.
    state: the result of the task.
    exception: the exception of the running this task if there is any.
    result: get the result of the task if there is any.
  """

  courier_method: str = ''
  args: tuple[Any, ...] = ()
  kwargs: dict[str, Any] = dataclasses.field(default_factory=dict)
  blocking: bool = False
  server_name: str = ''
  parent_task: 'Task | None' = None
  state: futures.Future[Any] | None = None

  @classmethod
  def new(
      cls,
      *args,
      courier_method: str = '',
      blocking: bool = False,
      **kwargs,
  ):
    """Convenient function to make a Task."""
    return cls(
        courier_method=courier_method,
        args=args,
        kwargs=kwargs,
        blocking=blocking,
    )

  def iterate(self, worker_pool):
    state = worker_pool.get_worker_by_name(self.server_name).call(
        courier_method='next_batch_from_generator'
    )
    return dataclasses.replace(self, state=state)

  @classmethod
  def from_list_of_tasks(cls, tasks: list['Task']) -> 'Task':
    iter_tasks = iter(tasks)
    task = next(iter_tasks)
    assert isinstance(task, Task)
    for next_task in iter_tasks:
      task = dataclasses.replace(next_task, parent_task=task)
    return task

  @property
  def done(self) -> bool | None:
    """Checks whether the task is done."""
    if self.state is not None:
      return self.state.done()

  @property
  def exception(self):
    return self.state and self.state.exception()

  @property
  def result(self):
    if self.state is None:
      return None
    return picklers.default.loads(self.state.result())

  def set(self, **kwargs):
    return dataclasses.replace(self, **kwargs)

  def add_task(
      self,
      task: 'Task | Any',
      *,
      courier_method: str = '',
      blocking: bool = False,
  ):
    """Append a task behind this task."""
    if not isinstance(task, Task):
      task = Task.new(
          task,
          blocking=blocking,
          courier_method=courier_method,
      )
    result = self
    for each_task in task.flatten():
      result = dataclasses.replace(each_task, parent_task=result)
    return result

  def flatten(self) -> list['Task']:
    if self.parent_task is None:
      return [self]
    return self.parent_task.flatten() + [self]


def get_results(
    states: list[futures.Future[Any]],
    with_states: bool = False,
    blocking: bool = True,
):
  """Gets the result of the futures, optionally outputs the non-done states."""
  states_left, result = [], []
  if blocking:
    wait_until_done(states)
  for state in states:
    assert isinstance(
        state, futures.Future
    ), f'state has to be a futures.Future, got {type(state)}'
    if state.done():
      result.append(state.result())
    else:
      states_left.append(state)
  result = [
      picklers.default.loads(state) if isinstance(state, bytes) else state
      for state in result
  ]
  return (result, states_left) if with_states else result


def _get_exception(state: futures.Future[Any]):
  """Gets the exception, non blocking."""
  if state.done():
    return state.exception()


def get_exceptions(
    states: list[futures.Future[Any]],
) -> list[Exception]:
  """Gets the exceptions from all states, non blocking."""
  result = [_get_exception(state) for state in states]
  return [state for state in result if state is not None]


def wait_until_done(states: list[futures.Future[Any]]):
  while not all(state.done() for state in states):
    time.sleep(0.01)


def _normalize_args(args, kwargs):
  """Normalizes the args and kwargs to be picklable."""
  result_args = []
  for arg in args:
    try:
      result_args.append(picklers.default.dumps(arg))
    except Exception as e:
      raise ValueError(f'Having issue pickling arg: {arg}') from e
  result_kwargs = {}
  for k, v in kwargs.items():
    try:
      result_kwargs[k] = picklers.default.dumps(v)
    except Exception as e:
      raise ValueError(f'Having issue pickling {k}: {v}') from e
  return result_args, result_kwargs


# TODO(b/311207032): Adds unit test to cover logic for disconneted worker.
@dataclasses.dataclass
class Worker:
  """Courier client wrapper that works as a chainable worker."""

  server_name: str
  call_timeout: int = 60
  max_parallelism: int = 1
  heartbeat_threshold_secs: int = 180
  _client: courier.Client | None = dataclasses.field(default=None, init=False)
  _pendings: list[futures.Future[Any]] = dataclasses.field(
      default_factory=list, init=False
  )
  _heartbeat: futures.Future[Any] | None = dataclasses.field(
      default=None, init=False
  )
  _last_heartbeat: float = 0.0

  def __post_init__(self):
    self._client = courier.Client(
        self.server_name, call_timeout=self.call_timeout
    )

  @property
  def has_capacity(self) -> bool:
    return len(self.pendings) < self.max_parallelism

  def _check_heartbeat(self) -> bool:
    """Ping the worker to check the heartbeat once."""
    if not self._heartbeat:
      self._heartbeat = Worker(
          self.server_name, call_timeout=self.call_timeout
      ).call(None)
    try:
      if self._heartbeat.done() and self._heartbeat.result():
        self._heartbeat = None
        self._last_heartbeat = time.time()
        return True
    except Exception:  # pylint: disable=broad-exception-caught
      logging.warning(
          'chainables: Worker %s missed a heartbeat.', self.server_name
      )
      self._heartbeat = None
    return False

  def wait_until_alive(
      self,
      num_attempts: int = 30,
      sleep_interval: float = 6.0,
  ):
    """Waits for the workers to be alive with retries."""
    for _ in range(num_attempts):
      try:
        if self.is_alive:
          break
      except Exception as e:  # pylint: disable=broad-exception-caught
        logging.warning('chainables: exception when connecting: %s', e)
      time.sleep(sleep_interval)
    else:
      raise ValueError(
          f'Failed to connect to worker {self.server_name} after'
          f' {num_attempts} tries.'
      )

  @property
  def is_alive(self) -> bool:
    """Checks whether the worker is alive."""
    if self._check_heartbeat():
      return True
    # No last heartbeat recorded, consider it dead.
    if self._last_heartbeat:
      time_passed = time.time() - self._last_heartbeat
      return time_passed < self.heartbeat_threshold_secs
    else:
      return False

  @property
  def pendings(self) -> list[futures.Future[Any]]:
    """Returns the states that are not done."""
    self._pendings = [state for state in self._pendings if not state.done()]
    return self._pendings

  def call(
      self, *args, courier_method: str = '', **kwargs
  ) -> futures.Future[Any]:
    args, kwargs = _normalize_args(args, kwargs)
    courier_method = courier_method or 'maybe_make'
    assert self._client is not None
    state = getattr(self._client.futures, courier_method)(*args, **kwargs)
    self._pendings.append(state)
    return state

  def run_task(
      self,
      tasks: Task,
      sleep_interval: float = 0.01,
  ) -> Task:
    """Runs tasks sequentially and returns the futures."""
    result = []
    for task in tasks.flatten():
      while not self.has_capacity:
        time.sleep(sleep_interval)
      state = self.call(
          *task.args, courier_method=task.courier_method, **task.kwargs
      )
      if task.blocking:
        wait_until_done([state])
      result.append(
          dataclasses.replace(task, state=state, server_name=self.server_name)
      )
    return Task.from_list_of_tasks(result)

  def clear_cache(self):
    return self.call(courier_method='clear_cache')

  def set_timeout(self, call_timeout: int):
    self.call_timeout = call_timeout
    self._client = courier.Client(self.server_name, call_timeout=call_timeout)

  def shutdown(self):
    self.state = self._client.futures.shutdown()
    return self.state


def _raise_if_return_error(task: Task):
  if not (result := task.state) or (result := task.exception):
    raise TypeError(
        f'Expected iterator, got {result} from'
        f' task: {dataclasses.replace(task, parent_task=None)}'
    )


@dataclasses.dataclass
class WorkerPool:
  """Worker group that constructs a group of courier workers."""

  server_names: list[str]
  call_timeout: int = 30
  max_parallelism: int = 1
  heartbeat_threshold_secs: int = 180
  _workers: dict[str, Worker] = dataclasses.field(default_factory=dict)

  def __post_init__(self):
    self._workers = {
        name: Worker(
            name,
            call_timeout=self.call_timeout,
            max_parallelism=self.max_parallelism,
            heartbeat_threshold_secs=self.heartbeat_threshold_secs,
        )
        for name in self.server_names
    }

  def wait_until_alive(
      self,
      deadline_secs: int = 180,
      *,
      minimum_num_workers: int = 1,
      sleep_interval_secs: int = 6,
  ):
    """Waits for the workers to be alive with retries."""
    num_attempts = max(deadline_secs // sleep_interval_secs, 1)
    for _ in range(num_attempts):
      try:
        workers = self.idle_workers()
        logging.info('chainables: Available workers: %d', len(workers))
        assert get_results(
            [worker.clear_cache() for worker in workers], blocking=True
        ) == [None] * len(workers)
        logging.info('chainables: %d workers connected.', len(workers))
        # Proceed if reached minimum number of workers.
        if len(workers) >= minimum_num_workers:
          break
      except Exception as e:  # pylint: disable=broad-exception-caught
        logging.warning('chainables: exception when connecting: %s', e)
      time.sleep(sleep_interval_secs)
    else:
      raise ValueError(
          f'Failed to connect to workers after {num_attempts} tries: workers:'
          f' {self.server_names}'
      )

  def call_and_wait(self, *args, courier_method='', **kwargs):
    states = [
        c.call(*args, courier_method=courier_method, **kwargs)
        for c in self._workers.values()
    ]
    states = [state for state in states if state is not None]
    return get_results(states)

  def set_timeout(self, timeout: int):
    self.call_timeout = timeout
    for c in self._workers.values():
      c.set_timeout(timeout)

  def idle_workers(self):
    return [c for c in self._workers.values() if c.has_capacity and c.is_alive]

  @property
  def workers(self):
    return [c for c in self._workers.values() if c.is_alive]

  @property
  def num_workers(self):
    return len(self.server_names)

  def get_worker_by_name(self, name: str) -> Worker:
    return self._workers[name]

  def shutdown(self):
    states = [c.shutdown() for c in self._workers.values()]
    wait_until_done(states)

  def run_tasks(
      self, tasks: list[Task], sleep_interval: float = 0.01
  ) -> Iterator[Task]:
    """Run tasks within the worker pool."""
    running_tasks = []
    while tasks or running_tasks:
      if not self.workers:
        raise ValueError(
            'No workers are alive, remaining'
            f' {len(tasks)+len(running_tasks)} tasks.'
        )
      for worker in self.idle_workers():
        if tasks:
          running_tasks.append(worker.run_task(tasks.pop()))
      still_running = []
      for task in running_tasks:
        if task.done:
          yield task
        elif not self._workers[task.server_name].is_alive:
          logging.warning(
              'chainables: Worker %s is not alive, re-appending task %s',
              task.server_name,
              dataclasses.replace(task, parent_task=None),
          )
          tasks.append(task)
        else:
          still_running.append(task)
      running_tasks = still_running
      time.sleep(sleep_interval)

  def run_and_iterate(
      self,
      tasks: list[Task],
      sleep_interval: float = 0.0,
      num_total_failures_threshold: int = _NUM_TOTAL_FAILURES_THRESHOLD,
  ) -> Iterator[Any]:
    """Iterates through the result of a generator if the iterator task."""
    tasks = [task for task in tasks]
    pending_tasks = []
    running_tasks = []
    total_tasks = len(tasks)
    total_failures_cnt = 0
    finished_tasks_cnt = 0
    ticker = time.time()
    while tasks or pending_tasks or running_tasks:
      if not self.workers:
        raise ValueError(
            'No workers are alive, remaining'
            f' {len(tasks)+len(running_tasks)} tasks.'
        )
      # Assign to the iterator tasks
      for worker in self.idle_workers():
        # Only assign non-running tasks to the workers that are not running.
        if worker.server_name not in {
            task.server_name for task in running_tasks
        }:
          if tasks:
            task = worker.run_task(tasks.pop())
            pending_tasks.append(task)
      # Check the instantiated iterator, then assign iteratate if successful.
      new_pending_tasks: list[Task] = []
      for task in pending_tasks:
        if task.done:
          _raise_if_return_error(task)
          running_tasks.append(task.iterate(self))
        else:
          new_pending_tasks.append(task)
      pending_tasks = new_pending_tasks
      # Fetching finsihed outputs. Re-collect task if failed during iteration.
      still_running: list[Task] = []
      for task in running_tasks:
        if task.done:
          try:
            states = task.result
            if states and lazy_fns.is_stop_iteration(states[-1]):
              logging.info(
                  'chainables: worker %s generator exhausted.',
                  task.server_name,
              )
              yield from states[:-1]
              finished_tasks_cnt += 1
            else:
              yield from states
              still_running.append(task.iterate(self))
          except Exception as e:  # pylint: disable=broad-exception-caught
            logging.exception(
                'chainables: exception when iterating, re-appending task %s, \n'
                ' exception: %s',
                dataclasses.replace(task, parent_task=None),
                e,
            )
            total_failures_cnt += 1
            if total_failures_cnt > num_total_failures_threshold:
              raise ValueError(
                  f'chainables: too many failures: {total_failures_cnt} >'
                  f' {num_total_failures_threshold}, stopping the iteration.'
              ) from e
            tasks.append(task)
        elif not self._workers[task.server_name].is_alive:
          logging.warning(
              'chainables: Worker %s is not alive, re-appending task %s',
              task.server_name,
              dataclasses.replace(task, parent_task=None),
          )
          tasks.append(task)
        else:
          still_running.append(task)
      running_tasks = still_running
      assert (
          finished_tasks_cnt
          + len(running_tasks)
          + len(pending_tasks)
          + len(tasks)
      ) == total_tasks, 'Total tasks mismatch.'
      if time.time() - ticker > _LOGGING_INTERVAL_SEC:
        logging.info(
            'chainables: iterate progress: %d/%d/%d/%d/%d in %.2f secs.'
            ' (running/pending/remaining/finished/total).',
            len(running_tasks),
            len(pending_tasks),
            len(tasks),
            finished_tasks_cnt,
            total_tasks,
            time.time() - ticker,
        )
        ticker = time.time()
      time.sleep(sleep_interval)

  def iterate_from_pipeline(
      self,
      define_pipeline: Callable[..., transform.TreeTransform],
      /,
      *pipeline_args,
      shard_per_worker: int = 1,
      with_result: bool = True,
      with_agg_state: bool = True,
      with_agg_result: bool = False,
      **pipeline_kwargs,
  ) -> Iterator[Any]:
    """The orchestration for the remote distributed chainable workers."""
    # logging.info('chainables: resolved address: %s', resolved_addrs)
    # pool = WorkerPool(
    #     list(resolved_addrs), call_timeout=3000, heartbeat_threshold_secs=300
    # )
    num_shards = shard_per_worker * self.num_workers
    self.wait_until_alive(deadline_secs=600)
    sharded_tasks = [
        Task.new(
            lazy_fns.trace(define_pipeline)(
                *pipeline_args,
                shard_index=i,
                num_shards=num_shards,
                **pipeline_kwargs,
            )
            .make()
            .iterate(
                with_agg_state=with_agg_state,
                with_result=with_result,
                with_agg_result=with_agg_result,
            ),
            blocking=True,
        )
        for i in range(num_shards)
    ]
    # Does not allow 10% of total number of tasks failed.
    batch_outputs_iterator = self.run_and_iterate(
        sharded_tasks,
        num_total_failures_threshold=int(len(sharded_tasks) * 0.5) + 1,
    )
    logging.info('chainables: running %d remote tasks', len(sharded_tasks))
    yield from batch_outputs_iterator

  def aggregate_by_pipeline(
      self,
      define_pipeline: Callable[..., transform.TreeTransform],
      /,
      *pipeline_args,
      shard_per_worker: int = 1,
      **pipeline_kwargs,
  ):
    """The orchestration for the remote distributed chainable workers."""
    batch_outputs_iterator = self.iterate_from_pipeline(
        define_pipeline,
        *pipeline_args,
        with_result=False,
        shard_per_worker=shard_per_worker,
        **pipeline_kwargs,
    )
    try:
      assert next(batch_outputs_iterator) is None
    except StopIteration as e:
      raise StopIteration(
          f'chainables: empty dataset from {define_pipeline}.'
      ) from e

    # Inference and aggregations.
    num_shards = self.num_workers * shard_per_worker
    agg_fn = define_pipeline(
        *pipeline_args, shard_index=0, num_shards=num_shards, **pipeline_kwargs
    ).make(mode=transform.RunnerMode.AGGREGATE)
    merged_state = None
    assert hasattr(agg_fn, 'merge_states')
    try:
      merged_state = agg_fn.merge_states(
          batch_outputs_iterator,
          mixed_input_types=True,
          strict_states_cnt=num_shards,
      )
    except ValueError as e:
      logging.exception('chainables: error during merging, exception: %s', e)
    results = None
    if merged_state is not None:
      results = agg_fn.get_result(merged_state)
      logging.info('chainables: results: %s', results)

    # Chainable pipeline user code end
    logging.info('chainables: shutting down pool')
    return results
