"""Tensor-table tooling for cistern. Standard library only.

Run with isolated mode so nothing from the environment or cwd is imported:

    python3 -I tools/tensor_table.py llama --config src/table/llama-3.2-1b.config.json --out ...

All downloaded data (config.json, safetensors header) is untrusted: it is parsed
as JSON only, with explicit type and size checks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import sys
import urllib.request
from pathlib import Path

SCHEMA = "cistern.tensors/1"
DEFAULT_REPO = "unsloth/Llama-3.2-1B"
# Resolved through https://huggingface.co/api/models/unsloth/Llama-3.2-1B
PINNED_REVISION = "9535bd9b1d1dea6acafbdc4813b728796aeb28da"
HF = "https://huggingface.co"
MAX_HEADER_BYTES = 64 * 1024 * 1024
MAX_CONFIG_BYTES = 1024 * 1024
UA = "cistern-tables/1"

DTYPE_BYTES = {"bf16": 2, "fp16": 2, "fp32": 4}
CONFIG_DTYPE = {"bfloat16": "bf16", "float16": "fp16", "float32": "fp32"}
SAFETENSORS_DTYPE = {"BF16": "bf16", "F16": "fp16", "F32": "fp32"}


class TableError(Exception):
    pass


def _http(url: str, headers: dict[str, str] | None = None, limit: int = MAX_CONFIG_BYTES) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": UA, **(headers or {})})
    with urllib.request.urlopen(req, timeout=60) as resp:  # noqa: S310 (https only, fixed host)
        data = resp.read(limit + 1)
    if len(data) > limit:
        raise TableError(f"response from {url} exceeds {limit} bytes")
    return data


def resolve_revision(repo: str) -> str:
    meta = json.loads(_http(f"{HF}/api/models/{repo}"))
    sha = meta.get("sha") if isinstance(meta, dict) else None
    if not isinstance(sha, str) or len(sha) != 40 or any(c not in "0123456789abcdef" for c in sha):
        raise TableError("API did not return a 40-hex revision sha")
    return sha


def _int(cfg: dict, key: str, default: int | None = None) -> int:
    v = cfg.get(key, default)
    if isinstance(v, bool) or not isinstance(v, int) or v <= 0:
        raise TableError(f"config.json: {key!r} must be a positive integer, got {v!r}")
    return v


def derive_llama(cfg: dict) -> tuple[list[dict], str]:
    """Return (tensors, dtype) in HF module order from a Llama config."""
    if not isinstance(cfg, dict):
        raise TableError("config.json must be a JSON object")
    mt = cfg.get("model_type")
    if mt != "llama":
        raise TableError(f"unsupported model_type {mt!r} (only 'llama')")
    h = _int(cfg, "hidden_size")
    layers = _int(cfg, "num_hidden_layers")
    heads = _int(cfg, "num_attention_heads")
    kv = _int(cfg, "num_key_value_heads", heads)
    inter = _int(cfg, "intermediate_size")
    vocab = _int(cfg, "vocab_size")
    hd = cfg.get("head_dim")
    if hd is None:
        if h % heads:
            raise TableError("hidden_size not divisible by num_attention_heads and no head_dim")
        hd = h // heads
    hd = _int({"head_dim": hd}, "head_dim")
    dt_name = cfg.get("torch_dtype", cfg.get("dtype", "bfloat16"))
    if dt_name not in CONFIG_DTYPE:
        raise TableError(f"unsupported dtype {dt_name!r}")
    dtype = CONFIG_DTYPE[dt_name]
    attn_bias = bool(cfg.get("attention_bias", False))
    mlp_bias = bool(cfg.get("mlp_bias", False))
    tied = bool(cfg.get("tie_word_embeddings", False))
    qd, kvd = heads * hd, kv * hd

    out: list[tuple[str, list[int]]] = [("model.embed_tokens.weight", [vocab, h])]
    for i in range(layers):
        p = f"model.layers.{i}"
        out += [
            (f"{p}.input_layernorm.weight", [h]),
            (f"{p}.self_attn.q_proj.weight", [qd, h]),
            (f"{p}.self_attn.k_proj.weight", [kvd, h]),
            (f"{p}.self_attn.v_proj.weight", [kvd, h]),
            (f"{p}.self_attn.o_proj.weight", [h, qd]),
        ]
        if attn_bias:
            out += [(f"{p}.self_attn.q_proj.bias", [qd]), (f"{p}.self_attn.k_proj.bias", [kvd]),
                    (f"{p}.self_attn.v_proj.bias", [kvd]), (f"{p}.self_attn.o_proj.bias", [h])]
        out += [
            (f"{p}.post_attention_layernorm.weight", [h]),
            (f"{p}.mlp.gate_proj.weight", [inter, h]),
            (f"{p}.mlp.up_proj.weight", [inter, h]),
            (f"{p}.mlp.down_proj.weight", [h, inter]),
        ]
        if mlp_bias:
            out += [(f"{p}.mlp.gate_proj.bias", [inter]), (f"{p}.mlp.up_proj.bias", [inter]),
                    (f"{p}.mlp.down_proj.bias", [h])]
    out.append(("model.norm.weight", [h]))
    if not tied:
        out.append(("lm_head.weight", [vocab, h]))

    esz = DTYPE_BYTES[dtype]
    tensors = [{"name": n, "shape": s, "dtype": dtype, "bytes": math.prod(s) * esz} for n, s in out]

    # Independent closed-form consistency check.
    expect = vocab * h * (1 if tied else 2) + h
    per = 2 * h + qd * h + 2 * kvd * h + h * qd + 3 * inter * h
    if attn_bias:
        per += qd + 2 * kvd + h
    if mlp_bias:
        per += 2 * inter + h
    expect += layers * per
    got = sum(math.prod(t["shape"]) for t in tensors)
    assert got == expect, f"param count mismatch: tensors={got} closed-form={expect}"
    return tensors, dtype


def build_table(name: str, tensors: list[dict], source: dict, verified: bool) -> dict:
    total_bytes = sum(t["bytes"] for t in tensors)
    total_params = sum(math.prod(t["shape"]) for t in tensors)
    assert total_bytes == sum(math.prod(t["shape"]) * DTYPE_BYTES[t["dtype"]] for t in tensors)
    return {
        "schema": SCHEMA,
        "name": name,
        "tensor_count": len(tensors),
        "total_params": total_params,
        "total_bytes": total_bytes,
        "source": source,
        "verified": verified,
        "tensors": tensors,
    }


def fetch_safetensors_header(repo: str, revision: str) -> dict:
    """Fetch ONLY the safetensors header via HTTP Range (8 bytes, then the header)."""
    url = f"{HF}/{repo}/resolve/{revision}/model.safetensors"
    first = _http(url, {"Range": "bytes=0-7"}, limit=8)
    if len(first) != 8:
        raise TableError("did not receive exactly 8 bytes for the header length")
    (n,) = struct.unpack("<Q", first)
    if not 2 <= n <= MAX_HEADER_BYTES:
        raise TableError(f"implausible safetensors header length {n}")
    raw = _http(url, {"Range": f"bytes=8-{8 + n - 1}"}, limit=n)
    if len(raw) != n:
        raise TableError("short read of safetensors header")
    hdr = json.loads(raw)
    if not isinstance(hdr, dict):
        raise TableError("safetensors header is not a JSON object")
    return hdr


def verify_against_header(tensors: list[dict], hdr: dict) -> None:
    hdr = {k: v for k, v in hdr.items() if k != "__metadata__"}
    mine = {t["name"]: t for t in tensors}
    problems = []
    for n in sorted(set(mine) - set(hdr)):
        problems.append(f"missing in safetensors: {n}")
    for n in sorted(set(hdr) - set(mine)):
        problems.append(f"extra in safetensors: {n}")
    for n in sorted(set(mine) & set(hdr)):
        e = hdr[n]
        if not isinstance(e, dict) or not isinstance(e.get("shape"), list):
            problems.append(f"{n}: malformed header entry")
            continue
        dt = SAFETENSORS_DTYPE.get(e.get("dtype"))
        if e["shape"] != mine[n]["shape"]:
            problems.append(f"{n}: shape {e['shape']} != {mine[n]['shape']}")
        if dt != mine[n]["dtype"]:
            problems.append(f"{n}: dtype {e.get('dtype')} != {mine[n]['dtype']}")
        off = e.get("data_offsets")
        if (not isinstance(off, list) or len(off) != 2 or not all(isinstance(x, int) for x in off)
                or off[1] - off[0] != mine[n]["bytes"]):
            problems.append(f"{n}: data_offsets {off} do not span {mine[n]['bytes']} bytes")
    if problems:
        raise TableError("verification failed:\n  " + "\n  ".join(problems[:50]))


def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1) + "\n")


def cmd_fetch_config(a: argparse.Namespace) -> None:
    rev = a.revision or resolve_revision(a.repo)
    raw = _http(f"{HF}/{a.repo}/resolve/{rev}/config.json")
    json.loads(raw)  # must be valid JSON; never executed
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_bytes(raw)
    print(json.dumps({"repo": a.repo, "revision": rev, "config_sha256": hashlib.sha256(raw).hexdigest()}))


def cmd_llama(a: argparse.Namespace) -> None:
    raw = Path(a.config).read_bytes()
    tensors, dtype = derive_llama(json.loads(raw))
    source = {"repo": a.repo, "revision": a.revision, "config": Path(a.config).name,
              "config_sha256": hashlib.sha256(raw).hexdigest()}
    verified = False
    if a.verify:
        verify_against_header(tensors, fetch_safetensors_header(a.repo, a.revision))
        verified = True
        source["verified_against"] = f"{HF}/{a.repo}/resolve/{a.revision}/model.safetensors (header only)"
    table = build_table(a.name, tensors, source, verified)
    write_json(Path(a.out), table)
    print(f"{a.name}: tensors={table['tensor_count']} params={table['total_params']:,} "
          f"bytes={table['total_bytes']:,} dtype={dtype} verified={verified}")
    if a.expect_params is not None:
        assert table["total_params"] == a.expect_params, "total params differ from --expect-params"
    if a.expect_bytes is not None:
        assert table["total_bytes"] == a.expect_bytes, "total bytes differ from --expect-bytes"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="tensor_table.py")
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch-config", help="download config.json at a revision")
    f.add_argument("--repo", default=DEFAULT_REPO)
    f.add_argument("--revision", help="default: resolve through the HF API")
    f.add_argument("--out", required=True)
    f.set_defaults(fn=cmd_fetch_config)
    l = sub.add_parser("llama", help="derive a Llama tensor table from config.json")
    l.add_argument("--config", required=True)
    l.add_argument("--out", required=True)
    l.add_argument("--repo", default=DEFAULT_REPO)
    l.add_argument("--revision", default=PINNED_REVISION)
    l.add_argument("--name", default="llama-3.2-1b")
    l.add_argument("--verify", action="store_true", help="compare with safetensors header (Range reads)")
    l.add_argument("--expect-params", type=int)
    l.add_argument("--expect-bytes", type=int)
    l.set_defaults(fn=cmd_llama)
    a = ap.parse_args(argv)
    try:
        a.fn(a)
    except (TableError, AssertionError, OSError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
