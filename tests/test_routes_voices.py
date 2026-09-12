"""Route tests for the voice catalog + cloning endpoints (/v1/voices)."""

from __future__ import annotations


def _make_voice(client, name: str = "mario") -> None:
    client.post(
        "/v1/voices",
        files=[
            ("name", (None, name)),
            ("file", ("s.wav", b"\x00" * 512, "audio/wav")),
        ],
    )

def test_list_voices_includes_aliases_catalog_and_customs(client):
    r = client.get("/v1/voices")
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "list"
    data = {d["id"]: d for d in body["data"]}
    # every OpenAI alias present with its target's language
    assert "alloy" in data
    assert data["alloy"]["source"] == "builtin"
    # a non-alias catalog voice (rafeal is not in DEFAULT_VOICE_ALIASES as 'rafael')
    assert "rafael" in data
    assert data["rafael"]["language"] == "pt"
    # no custom voices yet
    customs = [d for d in body["data"] if d["source"] == "custom"]
    assert customs == []


def test_create_voice_201_and_visible_in_list(client, tmp_path):
    files = [
        ("name", (None, "mario")),
        ("file", ("sample.wav", b"\x00" * 2048, "audio/wav")),
    ]
    r = client.post("/v1/voices", files=files)
    assert r.status_code == 201
    created = r.json()
    assert created["id"] == "mario"
    assert created["source"] == "custom"
    assert created["language"] is None

    r2 = client.get("/v1/voices")
    customs = [d for d in r2.json()["data"] if d["source"] == "custom"]
    assert [c["id"] for c in customs] == ["mario"]


def test_create_voice_optional_language(client):
    files = [
        ("name", (None, "luigi")),
        ("file", ("prompt.wav", b"\x00" * 512, "audio/wav")),
        ("language", (None, "it")),
    ]
    r = client.post("/v1/voices", files=files)
    assert r.status_code == 201
    assert r.json()["language"] == "it"


def test_create_voice_collides_with_alias(client, tmp_path):
    files = [
        ("name", (None, "alloy")),  # already an OpenAI alias
        ("file", ("sample.wav", b"\x00" * 512, "audio/wav")),
    ]
    r = client.post("/v1/voices", files=files)
    assert r.status_code == 409
    assert r.json()["error"]["code"] == "conflict"


def test_create_voice_missing_name_is_422(client):
    files = [("file", ("sample.wav", b"\x00" * 512, "audio/wav"))]
    r = client.post("/v1/voices", files=files)
    assert r.status_code == 422


def test_create_voice_invalid_name(client):
    files = [
        ("name", (None, "Bad Name")),
        ("file", ("sample.wav", b"\x00" * 512, "audio/wav")),
    ]
    r = client.post("/v1/voices", files=files)
    assert r.status_code == 400


def test_create_voice_unsupported_extension(client):
    files = [
        ("name", (None, "mario")),
        ("file", ("sample.exe", b"\x00" * 512, "application/binary")),
    ]
    r = client.post("/v1/voices", files=files)
    assert r.status_code == 400
    assert "Unsupported file type" in r.json()["error"]["message"]


def test_create_voice_oversize_413(client):
    files = [
        ("name", (None, "mario")),
        # > 25 MB default limit
        ("file", ("sample.wav", b"\x00" * (26 * 1024 * 1024), "audio/wav")),
    ]
    r = client.post("/v1/voices", files=files)
    assert r.status_code == 413
    assert r.json()["error"]["code"] == "entity_too_large"
    # nothing was persisted (no custom voices after the failed upload)
    customs = [d for d in client.get("/v1/voices").json()["data"] if d["source"] == "custom"]
    assert customs == []


def test_delete_custom_voice_204(client, tmp_path):
    # create first
    client.post(
        "/v1/voices",
        files=[("name", (None, "mario")), ("file", ("s.wav", b"\x00" * 512, "audio/wav"))],
    )
    r = client.delete("/v1/voices/mario")
    assert r.status_code == 204
    assert r.content == b"" or len(r.content) == 0
    # gone from the list, and the underlying file is gone
    customs = [d for d in client.get("/v1/voices").json()["data"] if d["source"] == "custom"]
    assert customs == []


def test_delete_builtin_405(client):
    r = client.delete("/v1/voices/alloy")
    assert r.status_code == 405
    assert r.json()["error"]["code"] == "method_not_allowed"
    # an alias that maps to a catalog voice is also builtin
    r2 = client.delete("/v1/voices/rafael")
    assert r2.status_code == 405


def test_create_voice_clone_failure_is_400_not_500(client, fake_model):
    """A clone failure (e.g. unsupported audio / no voice-cloning model) must
    surface as a clean 400, never an unhandled 500.

    Regression: the route used ``except payload_too_large`` where
    ``payload_too_large`` is a *factory function*, not an exception class,
    which raised ``TypeError: catching classes that do not inherit from
    BaseException`` (the ``except`` expression itself was evaluated and
    raised) and turned every clone failure into a 500.
    """
    def boom(path: str):  # noqa: ARG001 - duck-typed model signature
        raise ValueError("Voice cloning is unsupported for this model")

    fake_model.get_state_for_audio_prompt = boom  # type: ignore[method-assign]
    r = client.post(
        "/v1/voices",
        files=[
            ("name", (None, "mario")),
            ("file", ("s.wav", b"\x00" * 512, "audio/wav")),
        ],
    )
    assert r.status_code == 400
    assert "Failed to encode voice prompt" in r.json()["error"]["message"]
    # nothing persisted
    customs = [d for d in client.get("/v1/voices").json()["data"] if d["source"] == "custom"]
    assert customs == []


def test_delete_unknown_custom_404(client):
    r = client.delete("/v1/voices/ghost")
    assert r.status_code == 404
    assert "ghost" in r.json()["error"]["message"]
    assert r.json()["error"]["code"] == "not_found"


def test_speech_uses_custom_voice(client, tmp_path):
    """A cloned voice name must be usable in /v1/audio/speech."""
    client.post(
        "/v1/voices",
        files=[("name", (None, "mario")), ("file", ("s.wav", b"\x00" * 512, "audio/wav"))],
    )
    r = client.post(
        "/v1/audio/speech",
        json={"input": "hello", "voice": "mario", "response_format": "pcm"},
    )
    assert r.status_code == 200
    assert len(r.content) > 0