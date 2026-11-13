"""Generate AICB workloads and prepare them for SimAI-Analytical."""

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C
from . import flops

# Column layout of a workload line (see astra-sim Workload.cc).
COLUMNS = [
    "name", "reserved",
    "fwd_t", "fwd_type", "fwd_size",
    "ig_t", "ig_type", "ig_size",
    "wg_t", "wg_type", "wg_size",
    "update_t",
]
NUM_COLS = ["reserved", "fwd_t", "fwd_size", "ig_t", "ig_size",
            "wg_t", "wg_size", "update_t"]
TP_TYPES = {"ALLREDUCE", "ALLGATHER", "REDUCESCATTER", "ALLTOALL"}


def _run_aicb(args, out_path: Path) -> Path:
    # AICB always writes to aicb/results/workload/, so clear it first.
    wl_dir = C.AICB_ROOT / "results" / "workload"
    shutil.rmtree(wl_dir, ignore_errors=True)
    cmd = ["python3", "-m", "workload_generator.SimAI_training_workload_generator"] + args
    proc = subprocess.run(cmd, cwd=C.AICB_ROOT, capture_output=True, text=True)
    produced = list(wl_dir.glob("*.txt"))
    if proc.returncode != 0 or len(produced) != 1:
        raise RuntimeError(f"AICB failed for {args}:\n{proc.stderr[-2000:]}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(produced[0]), out_path)
    return out_path


def generate_gpt3(world_size, tp, pp, global_batch, micro_batch, out_path):
    m = C.GPT3_175B
    args = [
        "--frame=Megatron", f"--world_size={world_size}",
        f"--tensor_model_parallel_size={tp}", f"--pipeline_model_parallel={pp}",
        f"--global_batch={global_batch}", f"--micro_batch={micro_batch}",
        f"--num_layers={m['num_layers']}", f"--seq_length={m['seq_length']}",
        f"--hidden_size={m['hidden_size']}", "--epoch_num=1",
        f"--num_attention_heads={m['num_attention_heads']}",
        f"--model_name={m['model_name']}",
        f"--max_position_embeddings={m['seq_length']}",
        f"--vocab_size={m['vocab_size']}", "--use-distributed-optimizer",
    ]
    return _run_aicb(args, Path(out_path))


def generate_deepseek(world_size, ep, tp, global_batch, micro_batch, out_path):
    # AICB requires sequence parallelism when MoE is enabled.
    m = C.DEEPSEEK_16B
    args = [
        "--frame=DeepSeek", f"--world_size={world_size}",
        f"--tensor_model_parallel_size={tp}", "--pipeline_model_parallel=1",
        f"--global_batch={global_batch}", f"--micro_batch={micro_batch}",
        f"--num_layers={m['num_layers']}", f"--seq_length={m['seq_length']}",
        f"--hidden_size={m['hidden_size']}", "--epoch_num=1",
        f"--num_attention_heads={m['num_attention_heads']}",
        f"--model_name={m['model_name']}",
        f"--max_position_embeddings={m['seq_length']}",
        f"--vocab_size={m['vocab_size']}", "--use-distributed-optimizer",
        "--enable_sequence_parallel", "--moe_enable",
        f"--ffn_hidden_size={m['moe_ffn_hidden_size']}",
        f"--moe_router_topk={m['moe_router_topk']}",
        f"--num_experts={m['num_experts']}",
        f"--expert_model_parallel_size={ep}",
        f"--q_lora_rank={m['q_lora_rank']}", f"--kv_lora_rank={m['kv_lora_rank']}",
        f"--qk_nope_dim={m['qk_nope_dim']}", f"--qk_rope_dim={m['qk_rope_dim']}",
        f"--v_head_dim={m['v_head_dim']}",
        f"--n_shared_expert={m['n_shared_expert']}",
        f"--n_dense_layer={m['n_dense_layer']}",
    ]
    return _run_aicb(args, Path(out_path))


def read_workload(path):
    """Return (header lines, DataFrame) for a SimAI workload file."""
    header = Path(path).read_text().splitlines()[:2]
    df = pd.read_csv(path, sep="\t", skiprows=2, header=None, names=COLUMNS,
                     dtype=str, keep_default_na=False)
    df[NUM_COLS] = df[NUM_COLS].astype(np.int64)
    return header, df


def write_workload(header, df, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = [header[0], str(len(df))]   # second line is the row count
    body = df[COLUMNS].astype(str).agg("\t".join, axis=1)
    path.write_text("\n".join(header + body.tolist()) + "\n")
    return path


def optimizer_allgather_to_dp(df):
    """Move the distributed-optimizer parameter all-gather to the DP group.

    AICB puts it in the forward column of grad_norm, which SimAI maps to the
    TP group (NVLink). It actually runs over the DP group, so we put it on a
    separate row in the weight-gradient column.
    """
    idx = df.index[(df["name"] == "grad_norm") & (df["fwd_type"] == "ALLGATHER")]
    if not len(idx):
        return df
    df = df.copy()
    i = idx[0]
    ag = df.loc[i].copy()
    ag["name"] = "grad_norm_param_allgather"
    ag[["fwd_type", "ig_type"]] = "NONE"
    ag[["fwd_size", "ig_size", "fwd_t", "ig_t", "wg_t"]] = 0
    ag["wg_type"] = "ALLGATHER"
    ag["wg_size"] = df.at[i, "fwd_size"]
    df.loc[i, "fwd_type"] = "NONE"
    df.loc[i, "fwd_size"] = 0
    df = pd.concat([df.loc[:i], ag.to_frame().T, df.loc[i + 1:]], ignore_index=True)
    df[NUM_COLS] = df[NUM_COLS].astype(np.int64)
    return df


def fix_tp1(df):
    """Drop TP collectives when TP = 1.

    AICB still emits them for DeepSeek, and SimAI overflows on a TP group of
    size 1. Call optimizer_allgather_to_dp first so the DP all-gather is kept.
    """
    df = df.copy()
    for col in ("fwd", "ig"):
        is_tp = df[f"{col}_type"].isin(TP_TYPES)
        df.loc[is_tp, f"{col}_type"] = "NONE"
        df.loc[is_tp, f"{col}_size"] = 0
    return df


def fill_compute(df, model, throughput, **model_kw):
    """Set compute times (ns) to FLOPs / throughput."""
    df = df.copy()
    f_fwd, f_ig, f_wg = flops.per_line_flops(df, model, **model_kw)
    to_ns = 1e9 / throughput
    df["fwd_t"] = np.rint(f_fwd * to_ns).astype(np.int64)
    df["ig_t"] = np.rint(f_ig * to_ns).astype(np.int64)
    df["wg_t"] = np.rint(f_wg * to_ns).astype(np.int64)
    return df


def total_flops(df, model, **model_kw):
    return float(sum(x.sum() for x in flops.per_line_flops(df, model, **model_kw)))


def apply_skew(df, factor):
    """Scale expert compute and all-to-all bytes by a max/mean load factor.

    Models the most loaded EP rank, which everyone else waits for.
    """
    df = df.copy()
    exp = flops.expert_rows(df)
    t_cols = ["fwd_t", "ig_t", "wg_t"]
    df.loc[exp, t_cols] = np.rint(df.loc[exp, t_cols] * factor).astype(np.int64)
    for col in ("fwd", "ig", "wg"):
        a2a = df[f"{col}_type"] == "ALLTOALL_EP"
        df.loc[a2a, f"{col}_size"] = np.rint(
            df.loc[a2a, f"{col}_size"] * factor).astype(np.int64)
    return df


def placement_summary(ep, tp, gpus_per_server):
    """Fraction of a rank's all-to-all bytes that stay inside its host."""
    if ep * tp <= gpus_per_server:
        return 1.0
    peers_same_host = gpus_per_server // tp
    return (peers_same_host - 1) / (ep - 1)
