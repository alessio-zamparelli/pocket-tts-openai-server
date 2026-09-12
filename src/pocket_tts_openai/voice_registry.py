"""Persistent registry for custom (cloned) voices.

Custom voices live in ``<cache_dir>/voices/``: one ``<name>.safetensors`` file
per voice (a pocket-tts ``ModelState`` exported with
``pocket_tts.export_model_state``) plus an ``registry.json`` index.

The registry file is written atomically (temp file + ``os.replace``) so a
crash mid-write can never corrupt it; readers always see either the previous
or the new state.
"""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

# Custom voice names must be filesystem- and URL-safe.
NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")

# Default cache dir mirrors pocket-tts' own model/voice cache.
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "pocket_tts"

REGISTRY_FILENAME = "registry.json"


@dataclass(frozen=True)
class CustomVoice:
    name: str
    language: str | None = None
    created: str = ""  # iso8601 UTC


@dataclass
class VoiceRegistry:
    """Index of cloned voices: name -> ``<dir>/<name>.safetensors``."""

    directory: Path
    _voices: dict[str, CustomVoice] = field(default_factory=dict)

    # -- construction -------------------------------------------------------

    @classmethod
    def from_config_dir(cls, cache_dir: Path | None) -> "VoiceRegistry":
        directory = (cache_dir or DEFAULT_CACHE_DIR) / "voices"
        registry = cls(directory=directory)
        registry.load()
        return registry

    def __post_init__(self) -> None:
        # Ensure the storage dir exists so temp uploads / flushes always work.
        self.directory.mkdir(parents=True, exist_ok=True)

    # -- paths ---------------------------------------------------------------

    @property
    def registry_path(self) -> Path:
        return self.directory / REGISTRY_FILENAME

    def path_for(self, name: str) -> Path:
        return self.directory / f"{name}.safetensors"

    # -- CRUD ----------------------------------------------------------------

    def load(self) -> None:
        """Read registry.json (tolerates a missing file)."""
        self._voices.clear()
        if not self.registry_path.exists():
            return
        try:
            raw = json.loads(self.registry_path.read_text(encoding="utf-8"))
            for item in raw.get("voices", []):
                voice = CustomVoice(
                    name=item["name"],
                    language=item.get("language"),
                    created=item.get("created", ""),
                )
                self._voices[voice.name] = voice
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            logger.warning("ignoring unreadable voice registry %s: %s", self.registry_path, exc)

    def names(self) -> set[str]:
        return set(self._voices)

    def get(self, name: str) -> CustomVoice | None:
        return self._voices.get(name)

    def all(self) -> list[CustomVoice]:
        return [self._voices[name] for name in sorted(self._voices)]

    def add(self, name: str, language: str | None = None) -> CustomVoice:
        voice = CustomVoice(
            name=name,
            language=language,
            created=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )
        self._voices[name] = voice
        self._flush()
        return voice

    def remove(self, name: str) -> CustomVoice:
        voice = self._voices.pop(name)
        self._flush()
        return voice

    # -- internals -----------------------------------------------------------

    def _flush(self) -> None:
        """Atomically rewrite registry.json."""
        self.directory.mkdir(parents=True, exist_ok=True)
        payload = {"voices": [vars(v) for v in self.all()]}
        fd, tmp = tempfile.mkstemp(dir=self.directory, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            os.replace(tmp, self.registry_path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise


def validate_voice_name(name: str) -> str | None:
    """Return an error message for invalid custom voice names, else ``None``."""
    if not NAME_RE.match(name):
        return (
            "Voice name must be 1-64 chars, starting with a lowercase letter or "
            "digit; only [a-z0-9_-] allowed."
        )
    return None
