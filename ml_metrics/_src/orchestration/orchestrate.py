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
"""Orchestration for the remote distributed chainable workers."""

from __future__ import annotations

from collections.abc import Callable, Iterator
import queue
import threading
import time
from typing import Any

from absl import logging
from ml_metrics._src.chainables import courier_worker
from ml_metrics._src.chainables import lazy_fns
from ml_metrics._src.chainables import transform


def iterate_from_generators(
    worker_pool: courier_worker.WorkerPool,
    define_pipeline: Callable[..., transform.TreeTransform],
    /,
    *pipeline_args,
    shards_multiplier: int = 1,
    with_result: bool = True,
    result_queue: queue.SimpleQueue[transform.AggregateResult] | None = None,
    **pipeline_kwargs,
) -> Iterator[Any]:
  """The orchestration for the remote distributed chainable workers.

  Args:
    worker_pool: The worker pool to run the pipeline.
    define_pipeline: The pipeline definition.
    *pipeline_args: The pipeline args.
    shards_multiplier: The multiplier for the number of shards per worker. Set
      to N means there will be N x num_workers number of subtasks to be
      computed. This is set to slightly increase the parallelism.
    with_result: Whether to return the result.
    result_queue: The queue to output the result.
    **pipeline_kwargs: The pipeline kwargs.

  Yields:
    The batch outputs of of the pipeline.
  """
  calculate_agg_result = result_queue is not None
  num_shards = shards_multiplier * worker_pool.num_workers
  worker_pool.wait_until_alive(deadline_secs=600)
  sharded_tasks = [
      courier_worker.GeneratorTask.new(
          lazy_fns.trace(define_pipeline)(
              *pipeline_args,
              shard_index=i,
              num_shards=num_shards,
              **pipeline_kwargs,
          )
          .make()
          .iterate(
              with_result=with_result,
              with_agg_state=calculate_agg_result,
              # In the distributed fashion, the result for each shard is not
              # meaningful, thus is always disabled.
              with_agg_result=False,
          ),
      )
      for i in range(num_shards)
  ]
  logging.info('chainables: distributed on %d shards', len(sharded_tasks))
  # Inference and aggregations.
  states_queue = queue.SimpleQueue()
  if calculate_agg_result:

    def compute_result(
        states_queue: queue.SimpleQueue[transform.AggregateResult],
        result_queue: queue.SimpleQueue[transform.AggregateResult],
    ):
      agg_fn = define_pipeline(
          *pipeline_args,
          shard_index=0,
          num_shards=num_shards,
          **pipeline_kwargs,
      ).make(mode=transform.RunnerMode.AGGREGATE)
      if not agg_fn.has_agg:
        raise ValueError('chainables: no aggregations found in the pipeline.')

      def iterate_agg_state():
        while True:
          try:
            state = states_queue.get()
          except queue.Empty:
            time.sleep(0)
            continue
          if isinstance(state, StopIteration):
            return
          yield state.agg_state

      merged_state = agg_fn.merge_states(iterate_agg_state())
      # At most only one item in the output_q.
      result_queue.put(
          transform.AggregateResult(
              agg_state=merged_state,
              agg_result=agg_fn.get_result(merged_state),
          ),
      )

    thread = threading.Thread(
        target=compute_result, args=(states_queue, result_queue)
    )
    thread.start()

  # Does not allow 50% of total number of tasks failed.
  iterator = worker_pool.iterate(
      sharded_tasks,
      agg_result_queue=states_queue,
      num_total_failures_threshold=int(len(sharded_tasks) * 0.5) + 1,
  )
  logging.info('chainables: iterator: %s', iterator)
  yield from iterator

  if states_queue:
    states_queue.put(StopIteration())
