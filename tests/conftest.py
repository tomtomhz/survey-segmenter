"""Test isolation from the machine the tests happen to run on.

Two of these tests assert on whether an Anthropic API key is configured — one expects the
"no key yet" path and checks the app says so politely rather than crashing. Read from the real
home directory, that assertion passes on a fresh machine and fails on the developer's, which is
exactly the kind of test that gets marked flaky and then ignored.

The stronger reason is that it should not be reading a real credential at all. A test run has no
business anywhere near the user's key, and a stray print or a failure dump would put it in a log.
"""
import os

import pytest


@pytest.fixture(autouse=True)
def _isolated_config(tmp_path, monkeypatch):
    """Point the key store and the project store at a throwaway directory, for every test."""
    monkeypatch.setenv("SURVEY_SEGMENTER_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("SURVEY_SEGMENTER_PROJECTS", str(tmp_path / "projects"))
    # An environment key would override the file and defeat the isolation.
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    # ai_interpret reads the location once, at import, so the module-level paths are re-pointed
    # here rather than relying on the environment variable being consulted lazily.
    import importlib
    from pathlib import Path

    ai = importlib.import_module("ai_interpret")
    monkeypatch.setattr(ai, "_CONFIG_DIR", Path(os.environ["SURVEY_SEGMENTER_HOME"]))
    monkeypatch.setattr(ai, "_CONFIG_FILE",
                        Path(os.environ["SURVEY_SEGMENTER_HOME"]) / "config.json")
    yield
