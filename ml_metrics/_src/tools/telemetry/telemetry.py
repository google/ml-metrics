"""Telemetry for MLAT."""

from typing import Any, Callable

from absl import logging


def increment_counter(**kwargs) -> None:
  """Log message."""
  message = ', '.join([f'{k}={v}' for k, v in kwargs.items()])
  logging.info('Logging: %s', message)


def _monitor(**kwargs) -> Callable[[Any], Any]:
  """Decorator to log callable usage."""

  def decorator(
      callable_to_decorate: Callable[[Any], Any],
  ) -> Callable[[Any], Any]:
    increment_counter(**kwargs)
    return callable_to_decorate

  return decorator


function_monitor = _monitor
class_monitor = _monitor
