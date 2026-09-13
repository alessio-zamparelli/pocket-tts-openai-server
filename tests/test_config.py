"""Config parsing tests for the load-shedding knobs."""

from __future__ import annotations

import pytest

from pocket_tts_openai.config import Config


def test_load_shedding_envs():
    cfg = Config.from_env(
        {
            "POCKET_TTS_MAX_WAITING": "8",
            "POCKET_TTS_QUEUE_TIMEOUT_S": "0.25",
        }
    )
    assert cfg.max_waiting == 8
    assert cfg.queue_timeout_s == 0.25


def test_load_shedding_defaults_off():
    cfg = Config.from_env({})
    assert cfg.max_waiting == 0
    assert cfg.queue_timeout_s == 0.0


def test_queue_timeout_accepts_fractional():
    cfg = Config.from_env({"POCKET_TTS_QUEUE_TIMEOUT_S": "0.5"})
    assert cfg.queue_timeout_s == 0.5


def test_load_shedding_non_numeric_rejected():
    with pytest.raises(ValueError, match="POCKET_TTS_MAX_WAITING"):
        Config.from_env({"POCKET_TTS_MAX_WAITING": "lots"})
    with pytest.raises(ValueError, match="POCKET_TTS_QUEUE_TIMEOUT_S"):
        Config.from_env({"POCKET_TTS_QUEUE_TIMEOUT_S": "soon"})
