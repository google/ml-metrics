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
"""Chainable specific logging utilities."""

from absl import logging

_CHAINABLE_MSG = 'chainable: %s'


def info(fstr: str, msg: str = _CHAINABLE_MSG):
  logging.info(msg, fstr)


def debug(fstr: str, msg: str = _CHAINABLE_MSG):
  logging.debug(msg, fstr)


def warning(fstr: str, msg: str = _CHAINABLE_MSG):
  logging.warning(msg, fstr)


def error(fstr: str, msg: str = _CHAINABLE_MSG):
  logging.error(msg, fstr)


def exception(fstr: str, msg: str = _CHAINABLE_MSG):
  logging.exception(msg, fstr)


def fatal(fstr: str, msg: str = _CHAINABLE_MSG):
  logging.fatal(msg, fstr)


def log_every_n_seconds(
    level: int, fstr: str, n_seconds: int, msg: str = _CHAINABLE_MSG
) -> None:
  logging.log_every_n_seconds(level, msg, n_seconds, fstr)
