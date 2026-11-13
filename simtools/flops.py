"""Per-row FLOP counts for AICB workloads.

Matmul (m x k)(k x n) = 2mkn FLOPs. Backward: input-grad = linear + 2x
attention core, weight-grad = linear, so a layer costs 3x its forward.
Attention core uses the full s x s score matrix. Element-wise ops are ignored.
All counts are per GPU and per micro-batch (one workload row).
"""

import numpy as np
import pandas as pd

from . import config as C


def _layer_index(df, marker, num_layers):
    # Wraps around so repeated micro-batches get the same layer indices.
    opens = (df["name"] == marker).to_numpy()
    return (np.cumsum(opens) - 1) % num_layers


def per_line_flops(df: pd.DataFrame, model: str, tp: int, micro_batch: int = 1):
    """Return (fwd, ig, wg) FLOP arrays, one value per row."""
    n = len(df)
    lin = np.zeros(n)    # weight matmuls
    core = np.zeros(n)   # attention QK^T and AV
    name = df["name"].to_numpy()

    if model == "gpt3":
        m = C.GPT3_175B
        s, h, b = m["seq_length"], m["hidden_size"], micro_batch
        is_attn = name == "attention_layer"
        lin[is_attn] = 8 * b * s * h**2 / tp          # QKV + output proj
        core[is_attn] = 4 * b * s**2 * h / tp
        lin[name == "mlp_layer"] = 16 * b * s * h**2 / tp   # h -> 4h -> h

    elif model == "deepseek16":
        m = C.DEEPSEEK_16B
        s, h, b = m["seq_length"], m["hidden_size"], micro_batch
        T = b * s
        H = m["num_attention_heads"]
        qk = m["qk_nope_dim"] + m["qk_rope_dim"]
        v = m["v_head_dim"]

        # MLA attention (q_lora_rank = 0, so Q is a direct projection).
        lin[name == "attention_q_column"] = 2 * T * h * H * qk / tp
        lin[name == "attention_linear_kv_lora"] = (
            2 * T * h * (m["kv_lora_rank"] + m["qk_rope_dim"]) / tp)
        is_kv = name == "attention_kv_column"
        lin[is_kv] = 2 * T * m["kv_lora_rank"] * H * (m["qk_nope_dim"] + v) / tp
        core[is_kv] = 2 * T * s * H * (qk + v) / tp
        lin[name == "attention_o_row"] = 2 * T * H * v * h / tp

        # mlp_column/mlp_row: dense FFN in the first layer, shared experts
        # elsewhere. SwiGLU: gate + up on the column row, down on the row row.
        layer = _layer_index(df, "attention_q_column", m["num_layers"])
        f = np.where(layer < m["n_dense_layer"], m["dense_ffn_hidden_size"],
                     m["n_shared_expert"] * m["moe_ffn_hidden_size"])
        is_col, is_row = name == "mlp_column", name == "mlp_row"
        lin[is_col] = (2 * T * h * 2 * f / tp)[is_col]
        lin[is_row] = (2 * T * h * f / tp)[is_row]

        # Each MoE block is 5 rows: router, dispatch, 2 TP rows, combine.
        # With even routing every GPU handles T * topk token-expert pairs.
        pos = moe_position(df)
        lin[pos == 0] = 2 * T * h * m["num_experts"] / tp
        lin[pos == 2] = 2 * T * m["moe_router_topk"] * 3 * h * m["moe_ffn_hidden_size"] / tp

        lin[name == "final_column"] = 2 * T * h * m["vocab_size"] / tp
    else:
        raise ValueError(model)

    return lin + core, lin + 2 * core, lin


def moe_position(df):
    """Position (0-4) of each row inside its MoE block, -1 elsewhere."""
    is_moe = (df["name"] == "mlp_moelayer").to_numpy()
    starts = is_moe & ~np.r_[False, is_moe[:-1]]
    block = np.cumsum(starts)
    pos = pd.Series(np.arange(len(df))).groupby(block).cumcount().to_numpy()
    return np.where(is_moe, pos, -1)


def expert_rows(df):
    return moe_position(df) == 2
