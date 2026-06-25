"""Smoke + functional tests for the Tesseract Python frontend (B-041).

Run with: PYTHONPATH=<build>/python pytest python/tests
"""

import numpy as np
import pytest

import tesseract as ts


def test_import_and_version():
    assert ts.__version__
    assert hasattr(ts, "Tensor")
    assert hasattr(ts.nn, "Linear")
    assert hasattr(ts.models, "LlamaModel")


def test_tensor_numpy_roundtrip():
    a = np.arange(12, dtype=np.float32).reshape(3, 4)
    t = ts.Tensor(a)
    assert t.shape == [3, 4]
    assert t.dtype == ts.float32
    np.testing.assert_array_equal(t.numpy(), a)


def test_tensor_factories():
    z = ts.Tensor.zeros([2, 3])
    assert z.numpy().sum() == 0.0
    o = ts.Tensor.ones([2, 3])
    assert o.numpy().sum() == 6.0


def test_autograd_matmul_sum():
    x = ts.Tensor(np.ones((2, 3), dtype=np.float32))
    x.requires_grad = True
    w = ts.Tensor(np.ones((3, 4), dtype=np.float32))
    w.requires_grad = True
    y = x.matmul(w)              # [2, 4], all == 3
    s = ts.ops.relu(y)          # keep graph; values positive so identity
    loss = ts.ops.cross_entropy(
        y, ts.Tensor(np.array([0, 1], dtype=np.int64)))
    loss.backward()
    gx = x.grad()
    assert gx is not None
    assert gx.shape == [2, 3]


def test_mnist_style_mlp_overfits():
    rng = np.random.default_rng(0)
    N, D, C = 16, 8, 4
    x = ts.Tensor(rng.standard_normal((N, D)).astype(np.float32))
    y_np = rng.integers(0, C, size=N).astype(np.int64)
    y = ts.Tensor(y_np)

    model = ts.nn.Sequential([
        ts.nn.Linear(D, 32),
        ts.nn.ReLU(),
        ts.nn.Linear(32, C),
    ])
    opt = ts.optim.SGD(model.parameters(), lr=0.2)

    losses = []
    for _ in range(60):
        logits = model(x)
        loss = ts.ops.cross_entropy(logits, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())

    # The fixed batch must be memorized: loss falls substantially.
    assert losses[-1] < losses[0]
    assert losses[-1] < 0.5


def test_llama_inference_smoke():
    cfg = ts.models.LlamaConfig()
    cfg.vocab_size = 32
    cfg.hidden_size = 16
    cfg.num_hidden_layers = 2
    cfg.num_attention_heads = 4
    cfg.num_key_value_heads = 4
    cfg.intermediate_size = 32
    cfg.max_position_embeddings = 64

    model = ts.models.LlamaModel(cfg)
    gc = ts.models.LlamaGenerateConfig()
    gc.max_new_tokens = 5
    out = model.generate([1, 5, 9, 3], gc)
    assert len(out) == 4 + 5
    # Determinism (greedy).
    assert model.generate([1, 5, 9, 3], gc) == out


def test_moe_llama_smoke():
    cfg = ts.models.LlamaConfig()
    cfg.vocab_size = 24
    cfg.hidden_size = 16
    cfg.num_hidden_layers = 2
    cfg.num_attention_heads = 4
    cfg.num_key_value_heads = 4
    cfg.intermediate_size = 32
    cfg.num_experts = 4
    cfg.num_experts_per_tok = 2
    model = ts.models.LlamaModel(cfg)
    gc = ts.models.LlamaGenerateConfig()
    gc.max_new_tokens = 4
    out = model.generate([1, 2, 3], gc)
    assert len(out) == 3 + 4


def test_mamba_smoke():
    cfg = ts.models.MambaConfig()
    cfg.vocab_size = 24
    cfg.hidden_size = 16
    cfg.num_hidden_layers = 2
    cfg.d_state = 8
    cfg.d_conv = 4
    model = ts.models.MambaModel(cfg)
    gc = ts.models.MambaGenerateConfig()
    gc.max_new_tokens = 5
    out = model.generate([1, 4, 2], gc)
    assert len(out) == 3 + 5
    assert model.generate([1, 4, 2], gc) == out


def test_whitespace_tokenizer_roundtrip():
    cfg = ts.io.WhitespaceTokenizerConfig()
    cfg.vocab = ["<unk>", "hello", "world", "tesseract"]
    cfg.unk = "<unk>"
    tk = ts.io.WhitespaceTokenizer(cfg)
    ids = tk.encode("hello world", add_special_tokens=False)
    assert ids == [1, 2]
    assert tk.decode(ids) == "hello world"
