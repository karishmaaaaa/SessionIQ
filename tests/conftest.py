"""Shared test fixtures and offline stand-ins.

Everything here keeps the suite offline: a ``FakeClient`` replays queued tool
outputs (or raises queued errors) in place of ``anthropic.Anthropic``, so no test
touches the network or needs an API key.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import anthropic
import httpx
import pytest

from src.client import TOOL_NAME
from src.config import TRANSCRIPTS_DIR
from src.transcript import load_transcript

FIXTURES = Path(__file__).parent / "fixtures"


class FakeClient:
    """Stand-in for ``anthropic.Anthropic``.

    ``messages.create`` consumes one queued action per call: a dict becomes a
    tool-use response carrying it; an exception instance is raised. ``calls``
    counts how many times the API was hit.
    """

    def __init__(self, actions):
        self._actions = iter(actions)
        self.calls = 0
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **_kwargs):
        self.calls += 1
        action = next(self._actions)
        if isinstance(action, BaseException):
            raise action
        return SimpleNamespace(
            content=[SimpleNamespace(type="tool_use", name=TOOL_NAME, input=action)],
            usage=SimpleNamespace(input_tokens=1000, output_tokens=200),
            model="claude-sonnet-5",
            stop_reason="tool_use",
        )


def _status_error(cls, status_code):
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    return cls("simulated", response=httpx.Response(status_code, request=request), body=None)


@pytest.fixture
def fake_client():
    """Factory: ``fake_client([action, ...]) -> FakeClient``."""
    return FakeClient


@pytest.fixture
def rate_limit_error():
    """Factory for a fresh 429 RateLimitError (transient)."""
    return lambda: _status_error(anthropic.RateLimitError, 429)


@pytest.fixture
def bad_request_error():
    """Factory for a fresh 400 BadRequestError (non-retryable)."""
    return lambda: _status_error(anthropic.BadRequestError, 400)


@pytest.fixture
def load_fixture():
    """Loader: ``load_fixture("valid") -> dict``."""
    return lambda name: json.loads((FIXTURES / f"{name}.json").read_text())


@pytest.fixture
def session1():
    return load_transcript(TRANSCRIPTS_DIR / "session_01_strong.txt")


@pytest.fixture
def session5():
    return load_transcript(TRANSCRIPTS_DIR / "session_05_adversarial.txt")
