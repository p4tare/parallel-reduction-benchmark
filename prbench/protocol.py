from __future__ import annotations

import json
import selectors
from dataclasses import dataclass
from typing import Any, TextIO


class ProtocolError(RuntimeError):
    pass


@dataclass(frozen=True)
class WorkerEvent:
    event: str
    payload: dict[str, Any]


def read_event(stream: TextIO, timeout_s: float | None = None) -> WorkerEvent:
    if timeout_s is not None:
        try:
            selector = selectors.DefaultSelector()
            selector.register(stream, selectors.EVENT_READ)
            try:
                if not selector.select(timeout_s):
                    raise TimeoutError(f"worker event timeout after {timeout_s:.1f}s")
            finally:
                selector.close()
        except (AttributeError, OSError, ValueError):
            # Some non-POSIX text streams do not expose a selectable file descriptor.
            # The research reference platform is Linux; keep a graceful fallback elsewhere.
            pass
    line = stream.readline()
    if line == "":
        raise ProtocolError("worker closed stdout unexpectedly")
    try:
        payload = json.loads(line)
    except json.JSONDecodeError as exc:
        raise ProtocolError(f"worker emitted non-JSON stdout: {line!r}") from exc
    event = payload.get("event")
    if not isinstance(event, str):
        raise ProtocolError(f"worker message has no string event: {payload!r}")
    return WorkerEvent(event=event, payload=payload)


def send_command(stream: TextIO, command: str) -> None:
    stream.write(command.rstrip("\n") + "\n")
    stream.flush()
