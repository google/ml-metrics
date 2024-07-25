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
"""launchpad utils."""

# TODO: b/311207032 - Add unit test.

from collections.abc import Iterable
import queue
from typing import Callable

from absl import logging
import launchpad as lp
from ml_metrics._src.chainables import courier_server
from ml_metrics._src.chainables import courier_worker
from ml_metrics._src.chainables import transform
from ml_metrics._src.orchestration import orchestrate
import tensorflow as tf

MASTER = 'master'
WORKERS = 'workers'


def _init_tpu():
  resolver = tf.distribute.cluster_resolver.TPUClusterResolver('')
  tf.config.experimental_connect_to_cluster(resolver)
  tf.tpu.experimental.initialize_tpu_system(resolver)
  tf.config.experimental.enable_mlir_bridge()


def run_courier_server(worker_addr: str | lp.Address):
  """Runs the courier server."""
  try:
    _init_tpu()
  except Exception as e:  # pylint: disable=broad-exception-caught
    logging.warning('chainables: failed to init tpu: %s', e)
  if isinstance(worker_addr, lp.Address):
    worker_addr = worker_addr.resolve()
    server = courier_server.CourierServerWrapper(
        port=lp.get_port_from_address(worker_addr), prefetch_size=32
    )
  else:
    server = courier_server.CourierServerWrapper(worker_addr, prefetch_size=32)
  logging.info('chainables: courier resolved address: %s', worker_addr)
  server.run_until_shutdown()


def lp_run_master(
    define_pipeline,
    /,
    *pipeline_args,
    worker_addrs: Iterable[lp.Address | str],
    **pipeline_kwargs,
):
  """The orchestration code for the remote distributed chainable workers."""
  resolved_addrs = [
      addr.resolve() if isinstance(addr, lp.Address) else addr
      for addr in worker_addrs
  ]
  logging.info('chainables: resolved address: %s', resolved_addrs)
  server_thread = None
  # The server prefetch size has to be larger than the worker iterate_batch_size
  # to avoid the server being blocked.
  iterate_batch_size = 32
  if not resolved_addrs:
    server = courier_server.CourierServerWrapper(
        prefetch_size=iterate_batch_size
    )
    resolved_addrs = [server.build_server().address]
    server_thread = server.start()
  worker_pool = courier_worker.WorkerPool(
      list(resolved_addrs),
      call_timeout=3000,
      heartbeat_threshold_secs=300,
      iterate_batch_size=iterate_batch_size,
  )
  result_queue = queue.SimpleQueue()
  for _ in orchestrate.run_sharded_pipelines_as_iterator(
      worker_pool,
      define_pipeline,
      *pipeline_args,
      with_batch_output=False,
      result_queue=result_queue,
      **pipeline_kwargs,
  ):
    pass
  logging.info('eval: shutdown workers.')
  logging.info('eval: %s', result_queue.get())
  worker_pool.shutdown()
  if server_thread:
    server_thread.join()


def build_program_from_pipeline(
    name,
    define_pipeline: Callable[..., transform.TreeTransform],
    *args,
    num_workers: int,
    worker_prefix: str = '',
    workers_only: bool = False,
    **kwargs,
) -> lp.Program:
  """Builds a launchpad program from a chainables definition.

  Args:
    name: name of the program.
    define_pipeline: a function that returns a pipeline.
    *args: args to pass to define_pipeline.
    num_workers: number of workers.
    worker_prefix: prefix for the worker addresses.
    workers_only: if true, only workers are started.
    **kwargs: kwargs to pass to define_pipeline.

  Returns:
    a launchpad program.
  """
  program = lp.Program(name)

  # worker node owns this address
  if worker_prefix:
    worker_addrs = [f'{worker_prefix}_{i}' for i in range(num_workers)]
  else:
    worker_addrs = [lp.Address() for _ in range(num_workers)]

  with program.group(WORKERS):
    for addr in worker_addrs:
      worker_node = lp.PyNode(run_courier_server, addr)
      if isinstance(addr, lp.Address):
        worker_node.allocate_address(addr)
      program.add_node(worker_node)

  if not workers_only:
    program.add_node(
        lp.PyNode(
            lp_run_master,
            define_pipeline,
            *args,
            worker_addrs=worker_addrs,
            **kwargs,
        ),
        label=MASTER,
    )
  return program
