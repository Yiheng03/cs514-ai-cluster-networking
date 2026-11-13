"""Rough per-GPU memory for model state (no activations).

BF16 weights, FP32 grads, FP32 Adam state (12 B/param) sharded over the DP
group. Expert state is sharded over EP and the expert-DP group.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from simtools import config as C  # noqa: E402

GB = 1e9


def gpt3_params():
    m = C.GPT3_175B
    h, L, V = m["hidden_size"], m["num_layers"], m["vocab_size"]
    return 12 * L * h * h + V * h          # ~175B


def deepseek_params():
    m = C.DEEPSEEK_16B
    h, L, H = m["hidden_size"], m["num_layers"], m["num_attention_heads"]
    attn = (h * H * (m["qk_nope_dim"] + m["qk_rope_dim"])        # Q
            + h * (m["kv_lora_rank"] + m["qk_rope_dim"])         # KV down-proj
            + m["kv_lora_rank"] * H * (m["qk_nope_dim"] + m["v_head_dim"])
            + H * m["v_head_dim"] * h)                            # O
    n_moe = L - m["n_dense_layer"]
    dense_ffn = m["n_dense_layer"] * 3 * h * m["dense_ffn_hidden_size"]
    shared = n_moe * m["n_shared_expert"] * 3 * h * m["moe_ffn_hidden_size"]
    router = n_moe * h * m["num_experts"]
    emb = 2 * m["vocab_size"] * h       # untied input/output embeddings
    routed = n_moe * m["num_experts"] * 3 * h * m["moe_ffn_hidden_size"]
    return L * attn + dense_ffn + shared + router + emb, routed


def per_gpu(dense, expert, world, tp, pp, ep):
    d_local = dense / (tp * pp)
    e_local = expert / (tp * pp * ep)
    dp = world // (tp * pp)
    edp = max(1, world // (tp * pp * ep))
    w_g = (d_local + e_local) * (2 + 4)
    opt = d_local * 12 / dp + e_local * 12 / edp
    return (w_g + opt) / GB


def main():
    rows = [dict(model="GPT-3 175B", world=1024, tp=8, pp=8, ep=1,
                 params_B=gpt3_params() / 1e9,
                 state_GB=per_gpu(gpt3_params(), 0, 1024, 8, 8, 1))]
    d, e = deepseek_params()
    for world in (1024, 256):
        for ep in (8, 16, 32, 64):
            rows.append(dict(model="DeepSeek-V2-Lite", world=world, tp=1, pp=1,
                             ep=ep, params_B=(d + e) / 1e9,
                             state_GB=per_gpu(d, e, world, 1, 1, ep)))
    df = pd.DataFrame(rows).round(2)
    df["fits_80GB_H100"] = df.state_GB < 80
    df["fits_96GB_H20"] = df.state_GB < 96
    df.to_csv(C.SUMMARY_DIR / "memory_check.csv", index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
