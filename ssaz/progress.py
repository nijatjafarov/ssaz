"""Built-in progress reporting for long pipeline phases"""

from __future__ import annotations

import sys
import time
from typing import Iterable, Iterator, Optional, TextIO


class ProgressBar:
    """Minimal terminal progress bar with count, percent, and ETA.

    Args:
        label: Short phase name shown before the bar.
        total: Number of steps the bar completes at.
        width: Bar width in characters.
        stream: Output stream; defaults to ``sys.stderr``.
    """

    # Minimum seconds between in-place redraws
    min_interval = 0.1

    def __init__(self, label: str, total: int, width: int = 30,
                 stream: Optional[TextIO] = None) -> None:
        self.label = label
        self.total = max(total, 1)
        self.width = width
        self.stream = stream if stream is not None else sys.stderr
        self.started = time.perf_counter()
        self.done = 0
        self._extra = ""
        self._interactive = bool(getattr(self.stream, "isatty",
                                         lambda: False)())
        self._last_draw = 0.0
        self._last_length = 0
        self._last_milestone = -1

    def _line(self) -> str:
        fraction = min(self.done / self.total, 1.0)
        filled = int(self.width * fraction)
        elapsed = time.perf_counter() - self.started
        eta = elapsed * (1 - fraction) / fraction if fraction > 0 else 0.0
        return (f"{self.label:<11} [{'█' * filled}"
                f"{'░' * (self.width - filled)}] "
                f"{self.done}/{self.total}  {fraction:>4.0%}  "
                f"ETA {eta:5.1f}s  {self._extra}").rstrip()

    def update(self, done: Optional[int] = None, extra: str = "") -> None:
        self.done = self.done + 1 if done is None else done
        if extra:
            self._extra = extra
        if self._interactive:
            now = time.perf_counter()
            if (now - self._last_draw < self.min_interval
                    and self.done < self.total):
                return
            self._last_draw = now
            line = self._line()
            padding = " " * max(self._last_length - len(line), 0)
            self.stream.write("\r" + line + padding)
            self._last_length = len(line)
            self.stream.flush()
        else:
            milestone = min(int(self.done * 10 / self.total), 10)
            if milestone > self._last_milestone:
                self._last_milestone = milestone
                self.stream.write(self._line() + "\n")
                self.stream.flush()

    def finish(self) -> None:
        """Draw the final state and end the bar line"""
        if self._interactive:
            line = self._line()
            padding = " " * max(self._last_length - len(line), 0)
            self.stream.write("\r" + line + padding + "\n")
            self.stream.flush()
        elif self._last_milestone < 10:
            self.stream.write(self._line() + "\n")
            self.stream.flush()


def track(items: Iterable, label: str,
          stream: Optional[TextIO] = None) -> Iterator:
    items = list(items) if not hasattr(items, "__len__") else items
    bar = ProgressBar(label, len(items), stream=stream)
    for item in items:
        yield item
        bar.update()
    bar.finish()
