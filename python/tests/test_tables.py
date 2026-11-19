import json
from pathlib import Path

import pytest

from cistern_tables.cli import (DTYPE_BYTES, TableError, build_table, derive_llama,
                                verify_against_header)

ROOT = Path(__file__).resolve().parents[2]
TINY = json.loads((ROOT / "tests/fixtures/tiny_config.json").read_text())


def test_tiny_untied_has_lm_head():
    t, dtype = derive_llama(TINY)
    names = [x["name"] for x in t]
    assert dtype == "bf16" and "lm_head.weight" in names
    assert len(t) == 1 + 2 * 9 + 1 + 1
    assert all(x["bytes"] == x["shape"][0] * (x["shape"][1] if len(x["shape"]) > 1 else 1) * 2 for x in t)


def test_tied_drops_lm_head():
    t, _ = derive_llama({**TINY, "tie_word_embeddings": True})
    assert "lm_head.weight" not in [x["name"] for x in t]
    u, _ = derive_llama(TINY)
    assert sum(x["bytes"] for x in u) - sum(x["bytes"] for x in t) == 1003 * 100 * 2


def test_head_dim_and_gqa():
    t, _ = derive_llama({**TINY, "head_dim": 32})
    shapes = {x["name"]: x["shape"] for x in t}
    assert shapes["model.layers.0.self_attn.q_proj.weight"] == [4 * 32, 100]
    assert shapes["model.layers.0.self_attn.k_proj.weight"] == [2 * 32, 100]
    assert shapes["model.layers.0.self_attn.o_proj.weight"] == [100, 4 * 32]


def test_biases_add_tensors():
    base, _ = derive_llama(TINY)
    biased, _ = derive_llama({**TINY, "attention_bias": True, "mlp_bias": True})
    assert len(biased) - len(base) == 2 * (4 + 3)


def test_unknown_model_type_errors():
    with pytest.raises(TableError):
        derive_llama({**TINY, "model_type": "gpt2"})


def test_committed_real_table():
    d = json.loads((ROOT / "data/llama-3.2-1b.tensors.json").read_text())
    assert d["verified"] is True
    assert d["total_params"] == 1235814400 and d["total_bytes"] == 2471628800 and d["tensor_count"] == 146


def test_verify_detects_mismatch():
    t, _ = derive_llama(TINY)
    hdr = {x["name"]: {"dtype": "BF16", "shape": x["shape"], "data_offsets": [0, x["bytes"]]} for x in t}
    hdr["__metadata__"] = {"format": "pt"}
    verify_against_header(t, hdr)
    bad = dict(hdr)
    bad["model.norm.weight"] = {"dtype": "F32", "shape": [100], "data_offsets": [0, 200]}
    with pytest.raises(TableError):
        verify_against_header(t, bad)
    del hdr["lm_head.weight"]
    with pytest.raises(TableError):
        verify_against_header(t, hdr)
