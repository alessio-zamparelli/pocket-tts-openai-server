"""Unit tests for the voice registry (persistence, name validation, atomic IO)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pocket_tts_openai.voice_registry import DEFAULT_CACHE_DIR, CustomVoice, VoiceRegistry, validate_voice_name


def _registry(tmp_path: Path) -> VoiceRegistry:
    return VoiceRegistry(directory=tmp_path / "voices")


def _reload(tmp_path: Path) -> VoiceRegistry:
    reg = VoiceRegistry(directory=tmp_path / "voices")
    reg.load()
    return reg


def test_empty_registry_lists_nothing(tmp_path: Path):
    reg = _registry(tmp_path)
    assert reg.names() == set()
    assert reg.all() == []
    assert reg.get("alba") is None


def test_add_and_get_roundtrip(tmp_path: Path):
    reg = _registry(tmp_path)
    voice = reg.add("mario", language="it")
    assert isinstance(voice, CustomVoice)
    assert voice.name == "mario"
    assert voice.language == "it"
    assert reg.get("mario") is voice
    assert reg.names() == {"mario"}


def test_add_default_language(tmp_path: Path):
    reg = _registry(tmp_path)
    assert reg.add("leo").language is None


def test_add_duplicate_is_idempotent(tmp_path: Path):
    reg = _registry(tmp_path)
    reg.add("mario")
    reg.add("mario", language="it")
    voices = reg.all()
    assert len(voices) == 1
    assert voices[0].language == "it"


def test_remove_and_absent(tmp_path: Path):
    reg = _registry(tmp_path)
    reg.add("mario")
    reg.remove("mario")
    assert reg.get("mario") is None
    assert reg.names() == set()
    # Removing an unknown name is the caller's bug; the registry is strict.
    with pytest.raises(KeyError):
        reg.remove("mario")


def test_path_for_uses_cache_dir(tmp_path: Path):
    reg = _registry(tmp_path)
    assert reg.path_for("mario") == tmp_path / "voices" / "mario.safetensors"


def test_persist_roundtrip_across_instances(tmp_path: Path):
    a = _registry(tmp_path)
    a.add("mario", language="it")
    a.add("leo")
    a.remove("leo")

    # A fresh instance over the same directory must reload exactly.
    b = _reload(tmp_path)
    assert b.names() == {"mario"}
    voice = b.get("mario")
    assert voice is not None
    assert voice.language == "it"


def test_default_stores_none_language(tmp_path: Path):
    reg = _registry(tmp_path)
    voice = reg.add("leo")
    # add() persists exactly what it was given -- no implicit "en".
    assert voice.language is None
    reloaded = _reload(tmp_path)
    zoe = reloaded.get("leo")
    assert zoe is not None
    assert zoe.language is None


def test_flush_is_atomic_and_creates_dir(tmp_path: Path):
    reg = _registry(tmp_path)
    reg.add("mario")
    reg_json = tmp_path / "voices" / "registry.json"
    assert reg_json.exists()
    payload = json.loads(reg_json.read_text())
    mounted = {v["name"]: v for v in payload["voices"]}
    assert mounted["mario"]["language"] is None


def test_from_config_dir_default():
    reg = VoiceRegistry.from_config_dir(None)
    assert reg.directory == DEFAULT_CACHE_DIR / "voices"


def test_from_config_dir_explicit(tmp_path: Path):
    reg = VoiceRegistry.from_config_dir(tmp_path / "cache")
    assert reg.directory == tmp_path / "cache" / "voices"


@pytest.mark.parametrize(
    "name, valid",
    [
        ("mario", True),
        ("alba", True),
        ("voice-1", True),
        ("voice_1", True),
        ("a", True),
        ("trailing-", True),  # _ and - are legal anywhere after the first char
        ("", False),
        ("UPPER", False),
        ("has space", False),
        ("has.slash/", False),
        ("-leading", False),
        ("_leading", False),
        ("x" * 65, False),
    ],
)
def test_validate_voice_name(name: str, valid: bool):
    error = validate_voice_name(name)
    if valid:
        assert error is None
    else:
        assert error is not None
        assert "1-64" in error


def test_registry_add_does_not_validate(tmp_path: Path):
    # Validation is the route layer's responsibility; the registry is a dumb,
    # path-safe store and trusts the caller. (Enforced by tests/test_routes_voices.)
    reg = _registry(tmp_path)
    reg.add("Bad Name")  # must not raise
    assert "Bad Name" in reg.names()


def test_load_ignores_corrupt_registry(tmp_path: Path):
    voices_dir = tmp_path / "voices"
    voices_dir.mkdir(parents=True)
    (voices_dir / "registry.json").write_text("{not json")
    reg = VoiceRegistry(directory=voices_dir)
    reg.load()
    assert reg.names() == set()
