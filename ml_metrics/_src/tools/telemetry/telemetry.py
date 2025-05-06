"""Telemetry for MLAT."""

from absl import logging


def increment_counter(
    api: str, category: str, reference: str, execution_succeed: bool
):
  logging.info(
      'Logging counter: api=%s, category=%s, reference=%s,'
      ' execution_succeed=%s',
      api,
      category,
      reference,
      execution_succeed,
  )


class WithTelemetry:
  """Decorator to log usage."""

  def __init__(
      self,
      api: str,
      category: str,
      reference: str,
      *,
      target_methods: list[str] | str | None = None,
  ):
    self.api = api
    self.category = category
    self.reference = reference
    self.target_methods = target_methods

  def __call__(self, class_or_func_ref):
    increment_counter(self.api, self.category, self.reference, True)
    return class_or_func_ref
