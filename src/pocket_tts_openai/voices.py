"""Voice resolution: OpenAI aliases -> Kyutai catalog names, plus passthrough.

Pocket TTS accepts as voice:
  - a catalog name (e.g. ``alba``, ``giovanni``)
  - a local audio/safetensors path
  - an ``https://`` URL or ``hf://<repo>/<path>[@rev]``

We add OpenAI-compatible aliases on top and let anything else pass through
(forward-compat with future catalog entries and custom cloned voices).
"""

from __future__ import annotations

# OpenAI voice alias -> Kyutai catalog voice.
# `coral` is a non-OpenAI convenience alias pointing at the Italian voice.
DEFAULT_VOICE_ALIASES: dict[str, str] = {
    "alloy": "alba",        # en
    "echo": "charles",      # en
    "fable": "eponine",     # en (British)
    "onyx": "bill_boerst",  # en
    "nova": "eve",          # en
    "shimmer": "fantine",   # en
    "coral": "giovanni",    # it — default Italian voice
}

# Known Kyutai catalog voices (pass-through, no alias needed).
KYUTAI_CATALOG: set[str] = {
    # en
    "alba", "anna", "azelma", "bill_boerst", "caro_davy", "charles", "cosette",
    "eponine", "eve", "fantine", "george", "jane", "javert", "jean", "marius",
    "mary", "michael", "paul", "peter_yearsley", "stuart_bell", "vera",
    # other languages
    "giovanni",   # it
    "lola",       # es
    "juergen",    # de
    "rafael",     # pt
    "estelle",    # fr
}

_PASSTHROUGH_PREFIXES = ("hf://", "https://", "http://")


def is_passthrough_voice(voice: str) -> bool:
    """True for URLs, hf:// paths and filesystem paths (incl. .safetensors)."""
    v = voice.strip()
    return (
        v.startswith(_PASSTHROUGH_PREFIXES)
        or v.startswith(("/", "./", "../", "~/"))
        or v.lower().endswith((".wav", ".mp3", ".flac", ".safetensors"))
    )


def resolve_voice(requested: str, voice_map: dict[str, str]) -> str:
    """Map an alias to a Kyutai voice; catalog names and paths/URLs pass through.

    Unknown bare names are rejected so client typos fail fast with a helpful
    message instead of a cryptic model-side error.
    """
    v = requested.strip()
    if v in voice_map:
        return voice_map[v]
    if v in KYUTAI_CATALOG or is_passthrough_voice(v):
        return v
    catalog = ", ".join(sorted(KYUTAI_CATALOG))
    raise ValueError(
        f"Unknown voice {requested!r}. Use an OpenAI alias, a catalog voice "
        f"({catalog}), a local audio/safetensors path, an https:// URL or an "
        "hf:// reference."
    )
