"""Backend selection in track1_qwen, without loading any real model."""
from src.track1_query_understanding import track1_qwen as t


def _reset():
    t._state.update(model=None, tokenizer=None, device=None, name=None, alias=None)


def test_missing_gguf_falls_back_to_fp32_once(monkeypatch):
    _reset()
    calls = []

    def fake_gguf(name):
        calls.append(("gguf", name))
        raise FileNotFoundError("no file")

    real_load = t.load_model

    def fake_load(name=t.DEFAULT_MODEL):
        if name == t.DEFAULT_MODEL:          # stand-in for the transformers load
            calls.append(("hf", name))
            t._state.update(model=object(), name=name, device="cpu")
            return name, "cpu"
        return real_load(name)

    monkeypatch.setattr(t, "_load_gguf", fake_gguf)
    monkeypatch.setattr(t, "load_model", fake_load)
    assert t.load_model("data/models/missing.gguf") == (t.DEFAULT_MODEL, "cpu")
    assert t.load_model("data/models/missing.gguf") == (t.DEFAULT_MODEL, "cpu")
    assert calls == [("gguf", "data/models/missing.gguf"), ("hf", t.DEFAULT_MODEL)]
    _reset()


def test_gguf_detection():
    assert t._is_gguf("data/models/Qwen3-1.7B-Q4_K_M.gguf")
    assert not t._is_gguf("Qwen/Qwen3-1.7B")
