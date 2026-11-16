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


class _FakeLlama:
    calls = []
    fail_on_gpu = False

    def __init__(self, model_path, n_gpu_layers=0, **kw):
        _FakeLlama.calls.append(n_gpu_layers)
        if n_gpu_layers != 0 and _FakeLlama.fail_on_gpu:
            raise RuntimeError("out of GPU memory")

    def close(self):
        pass


def _fake_backends(monkeypatch, gpu_build):
    import sys
    import types
    fake_llama = types.ModuleType("llama_cpp")
    fake_llama.Llama = _FakeLlama
    fake_llama.llama_supports_gpu_offload = lambda: gpu_build
    fake_tf = types.ModuleType("transformers")
    fake_tf.AutoTokenizer = types.SimpleNamespace(from_pretrained=lambda name: object())
    monkeypatch.setitem(sys.modules, "llama_cpp", fake_llama)
    monkeypatch.setitem(sys.modules, "transformers", fake_tf)
    monkeypatch.setattr(t, "_gpu_layers", lambda: -1)
    _FakeLlama.calls = []


def test_gguf_uses_gpu_when_build_supports_it(monkeypatch, tmp_path):
    _reset()
    _fake_backends(monkeypatch, gpu_build=True)
    f = tmp_path / "q.gguf"
    f.write_bytes(b"")
    _, device = t._load_gguf(str(f))
    assert _FakeLlama.calls == [-1] and device.startswith("gpu")
    _reset()


def test_gguf_stays_on_cpu_with_cpu_build(monkeypatch, tmp_path):
    _reset()
    _fake_backends(monkeypatch, gpu_build=False)
    f = tmp_path / "q.gguf"
    f.write_bytes(b"")
    _, device = t._load_gguf(str(f))
    assert _FakeLlama.calls == [0] and device.startswith("cpu")
    _reset()


def test_gguf_falls_back_to_cpu_when_gpu_load_fails(monkeypatch, tmp_path):
    _reset()
    _fake_backends(monkeypatch, gpu_build=True)
    _FakeLlama.fail_on_gpu = True
    try:
        f = tmp_path / "q.gguf"
        f.write_bytes(b"")
        _, device = t._load_gguf(str(f))
        assert _FakeLlama.calls == [-1, 0] and device.startswith("cpu")
    finally:
        _FakeLlama.fail_on_gpu = False
        _reset()
