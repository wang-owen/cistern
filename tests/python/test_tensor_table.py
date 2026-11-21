"""Tests for tools/tensor_table.py (stdlib unittest; run with `python3 -I`)."""

import importlib.util
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# -I drops the script directory from sys.path, so load the tool by explicit path.
_spec = importlib.util.spec_from_file_location("tensor_table", ROOT / "tools" / "tensor_table.py")
tt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tt)

TINY = json.loads((ROOT / "tests/fixtures/tiny_config.json").read_text())


class DeriveLlama(unittest.TestCase):
    def test_tiny_untied_has_lm_head(self):
        t, dtype = tt.derive_llama(TINY)
        names = [x["name"] for x in t]
        self.assertEqual(dtype, "bf16")
        self.assertIn("lm_head.weight", names)
        self.assertEqual(len(t), 1 + 2 * 9 + 1 + 1)
        for x in t:
            cols = x["shape"][1] if len(x["shape"]) > 1 else 1
            self.assertEqual(x["bytes"], x["shape"][0] * cols * 2)

    def test_tied_drops_lm_head(self):
        t, _ = tt.derive_llama({**TINY, "tie_word_embeddings": True})
        self.assertNotIn("lm_head.weight", [x["name"] for x in t])
        u, _ = tt.derive_llama(TINY)
        self.assertEqual(sum(x["bytes"] for x in u) - sum(x["bytes"] for x in t), 1003 * 100 * 2)

    def test_head_dim_and_gqa(self):
        t, _ = tt.derive_llama({**TINY, "head_dim": 32})
        shapes = {x["name"]: x["shape"] for x in t}
        self.assertEqual(shapes["model.layers.0.self_attn.q_proj.weight"], [4 * 32, 100])
        self.assertEqual(shapes["model.layers.0.self_attn.k_proj.weight"], [2 * 32, 100])
        self.assertEqual(shapes["model.layers.0.self_attn.o_proj.weight"], [100, 4 * 32])

    def test_biases_add_tensors(self):
        base, _ = tt.derive_llama(TINY)
        biased, _ = tt.derive_llama({**TINY, "attention_bias": True, "mlp_bias": True})
        self.assertEqual(len(biased) - len(base), 2 * (4 + 3))

    def test_unknown_model_type_errors(self):
        with self.assertRaises(tt.TableError):
            tt.derive_llama({**TINY, "model_type": "gpt2"})


class CommittedTable(unittest.TestCase):
    def test_committed_real_table(self):
        d = json.loads((ROOT / "src/table/llama-3.2-1b.tensors.json").read_text())
        self.assertIs(d["verified"], True)
        self.assertEqual(d["total_params"], 1235814400)
        self.assertEqual(d["total_bytes"], 2471628800)
        self.assertEqual(d["tensor_count"], 146)


class VerifyHeader(unittest.TestCase):
    def test_verify_detects_mismatch(self):
        t, _ = tt.derive_llama(TINY)
        hdr = {x["name"]: {"dtype": "BF16", "shape": x["shape"], "data_offsets": [0, x["bytes"]]}
               for x in t}
        hdr["__metadata__"] = {"format": "pt"}
        tt.verify_against_header(t, hdr)
        bad = dict(hdr)
        bad["model.norm.weight"] = {"dtype": "F32", "shape": [100], "data_offsets": [0, 200]}
        with self.assertRaises(tt.TableError):
            tt.verify_against_header(t, bad)
        del hdr["lm_head.weight"]
        with self.assertRaises(tt.TableError):
            tt.verify_against_header(t, hdr)


if __name__ == "__main__":
    unittest.main()
