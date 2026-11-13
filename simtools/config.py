"""Paths, pinned versions, and model/network parameters."""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Set SIMAI_ROOT to override. Default: the third_party/SimAI submodule.
_DEFAULT_SIMAI = REPO_ROOT / "third_party" / "SimAI"
SIMAI_ROOT = Path(os.environ.get("SIMAI_ROOT", _DEFAULT_SIMAI)).resolve()
AICB_ROOT = SIMAI_ROOT / "aicb"
SIMAI_BIN = SIMAI_ROOT / "bin" / "SimAI_analytical"

WORKLOAD_DIR = REPO_ROOT / "workloads"
RAW_DIR = REPO_ROOT / "results" / "raw"
SUMMARY_DIR = REPO_ROOT / "results"
FIG_DIR = REPO_ROOT / "figures"

SIMAI_COMMIT = "a57d765c5478f319c5f5ab6fdc53f9cc7f461743"
AICB_COMMIT = "23eec3c48ca2d2d93dd888a4c7b22ab4421e782f"

# One NIC per GPU, so per-NIC bandwidth = per-GPU bandwidth. Gbit/s -> GB/s.
NIC_GBPS = {100: 12.5, 200: 25.0, 400: 50.0}

# NVLink bus bandwidth in GB/s (value from the SimAI README example).
# H100 and H20 have the same NVLink 4, so both use it.
NVLINK_GBPS = 360.0

GPU_TYPES = ("H100", "H20")

GPT3_175B = dict(
    model_name="gpt_175B", num_layers=96, hidden_size=12288,
    num_attention_heads=96, seq_length=2048, vocab_size=50257,
    ffn_hidden_size=4 * 12288,
)

# DeepSeek-V2-Lite. AICB's deepseek16 preset sets hidden_size=10944, which is
# the dense FFN width; the real hidden size is 2048.
DEEPSEEK_16B = dict(
    model_name="DeepSeek_16B", num_layers=27, hidden_size=2048,
    num_attention_heads=16, seq_length=4096, vocab_size=102400,
    moe_ffn_hidden_size=1408,
    dense_ffn_hidden_size=10944,
    num_experts=64, n_shared_expert=2, moe_router_topk=6, n_dense_layer=1,
    q_lora_rank=0, kv_lora_rank=512, qk_nope_dim=128, qk_rope_dim=64,
    v_head_dim=128,
)
