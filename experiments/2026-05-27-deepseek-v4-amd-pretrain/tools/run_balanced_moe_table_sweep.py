#!/usr/bin/env python3
"""Run and summarize the balanced-MoE backend comparison table.

This is intentionally a small, explicit experiment harness: it launches the
same TorchTitan canary for each model/MBS/backend row, keeps raw logs under
/scratch, then writes one compact JSON summary in run_artifacts/.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import shlex
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


EXP_DIR = Path(__file__).resolve().parents[1]
PLAYGROUND_DIR = EXP_DIR.parents[1]
LAUNCHER = EXP_DIR / "launchers" / "torchtitan_direct_deepseek_v4_4x8xmi350_canary.sh"
ARTIFACT_DIR = EXP_DIR / "run_artifacts"

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
STEP_RE = re.compile(
    r"step:\s*(?P<step>\d+)\s+"
    r"loss:\s*(?P<loss>[0-9.]+)\s+"
    r"grad_norm:\s*(?P<grad_norm>[0-9.]+)\s+"
    r"memory:\s*(?P<memory>[0-9.]+)GiB.*?"
    r"tps:\s*(?P<tps>[0-9,]+)\s+"
    r"tflops:\s*(?P<tflops>[0-9.]+)\s+"
    r"mfu:\s*(?P<mfu>[0-9.]+)%"
)
RESOURCE_FAILURE_RE = re.compile(r"Resource temporarily unavailable|\[spawn_error\]", re.IGNORECASE)
LOCAL_LOG_DIR = Path(os.environ.get("BALANCED_MOE_SWEEP_LOCAL_LOG_DIR", "/tmp"))


MODELS: dict[str, dict[str, Any]] = {
    "deepseek": {
        "label": "DeepSeek-V4 Flash 12-layer MTP1",
        "base": "deepseek",
        "module": "deepseek_v4",
        "config": "deepseek_v4_flash_12layer_ceiling_probe_bf16_canary",
        "env": {
            "CANARY_DSV4_NUM_MTP_MODULES": "1",
            "CANARY_DSV4_MTP_LOSS_WEIGHT": "0.3",
        },
    },
    "deepseek_mtp1": {
        "label": "DeepSeek-V4 Flash 12-layer MTP1",
        "base": "deepseek",
        "module": "deepseek_v4",
        "config": "deepseek_v4_flash_12layer_ceiling_probe_bf16_canary",
        "env": {
            "CANARY_DSV4_NUM_MTP_MODULES": "1",
            "CANARY_DSV4_MTP_LOSS_WEIGHT": "0.3",
        },
    },
    "deepseek_nomtp": {
        "label": "DeepSeek-V4 Flash 12-layer no-MTP historical control",
        "base": "deepseek",
        "module": "deepseek_v4",
        "config": "deepseek_v4_flash_12layer_ceiling_probe_bf16_canary",
        "env": {
            "CANARY_DSV4_NUM_MTP_MODULES": "0",
            "CANARY_DSV4_MTP_LOSS_WEIGHT": "0.3",
        },
    },
    "qwen": {
        "label": "Qwen3.5-397B-A17B 8-layer",
        "base": "qwen",
        "module": "qwen3",
        "config": "qwen3_5_397b_a17b_reduced_8layer_hothelper_gate",
    },
    "kimi": {
        "label": "Kimi-K2.6 6-layer",
        "base": "kimi",
        "module": "kimi_k2",
        "config": "kimi_k2_6_reduced_6layer_hothelper_gate",
    },
}


VARIANTS: dict[str, dict[str, Any]] = {
    "plain": {
        "label": "plain no-helper standard EP",
        "env": {
            "CANARY_MOE_COMM_BACKEND": "standard",
            "CANARY_STANDARD_EP_LOCAL_EXPERT_BACKEND": "grouped_mm",
            "CANARY_STANDARD_EP_WEIGHTED_SCATTER_BACKEND": "triton_bwd",
            "CANARY_WEIGHTED_SCATTER_BWD_SCORE_MODE": "torch",
        },
    },
    "primus": {
        "label": "Primus-Turbo TurboEPBackend no-helper raw top-k EP",
        "env": {
            "CANARY_MOE_COMM_BACKEND": "standard",
            "CANARY_STANDARD_EP_LOCAL_EXPERT_BACKEND": "grouped_mm",
            "CANARY_STANDARD_EP_BACKEND_NORMAL_TOPK_DISPATCH_TENSORS": "1",
            "CANARY_STANDARD_EP_BACKEND_NORMAL_TOPK_DISPATCH_VALIDATE": "0",
            "CANARY_STANDARD_EP_BACKEND_NORMAL_TOPK_DISPATCH_PERMUTE": "1",
            "CANARY_MORI_AITER_BALANCED_MOE_PLAN_BACKEND": "primus_turbo",
        },
    },
    "helper8": {
        "label": "standard-EP top-8 hot-helper retained SDMA",
        "env": {
            "CANARY_MOE_COMM_BACKEND": "standard",
            "CANARY_MOE_BALANCED_MOE_MODE": "execute",
            "CANARY_MOE_BALANCED_MOE_HOT_EXPERTS": "8",
            "CANARY_STANDARD_EP_LOCAL_EXPERT_BACKEND": "grouped_mm",
            "CANARY_STANDARD_EP_WEIGHTED_SCATTER_BACKEND": "triton_bwd",
            "CANARY_WEIGHTED_SCATTER_BWD_SCORE_MODE": "torch",
            "CANARY_STANDARD_EP_HOT_WEIGHT_FORWARD_BACKEND": "mori_sdma",
            "CANARY_STANDARD_EP_HOT_WEIGHT_REDUCE_BACKEND": "mori_sdma",
            "CANARY_STANDARD_EP_HOT_SPLIT_MATERIALIZE_ROWS": "owner_compact",
            "CANARY_STANDARD_EP_HOT_SPLIT_PRESORT_MODE": "owner_compact",
            "CANARY_STANDARD_EP_HOT_SPLIT_WEIGHT_LAYOUT": "selected",
            "CANARY_STANDARD_EP_HOT_SPLIT_WEIGHTED_SCATTER_BACKEND": "triton_bwd",
        },
    },
    "primus_helper8_native": {
        "label": "Primus-Turbo native top-8 hot-helper with MORI SDMA",
        "env": {
            "CANARY_MOE_COMM_BACKEND": "standard",
            "CANARY_MOE_BALANCED_MOE_MODE": "execute",
            "CANARY_MOE_BALANCED_MOE_HOT_EXPERTS": "8",
            "CANARY_MORI_AITER_BALANCED_MOE_PLAN_BACKEND": "primus_turbo",
            "CANARY_STANDARD_EP_LOCAL_EXPERT_BACKEND": "grouped_mm",
            "CANARY_STANDARD_EP_BACKEND_NORMAL_TOPK_DISPATCH_TENSORS": "1",
            "CANARY_STANDARD_EP_BACKEND_NORMAL_TOPK_DISPATCH_VALIDATE": "0",
            "CANARY_STANDARD_EP_BACKEND_NORMAL_TOPK_DISPATCH_PERMUTE": "1",
            "CANARY_STANDARD_EP_WEIGHTED_SCATTER_BACKEND": "triton_bwd",
            "CANARY_WEIGHTED_SCATTER_BWD_SCORE_MODE": "torch",
            "CANARY_STANDARD_EP_HOT_SPLIT_MATERIALIZE_ROWS": "owner_compact",
            "CANARY_STANDARD_EP_HOT_SPLIT_PRESORT_MODE": "owner_compact",
            "CANARY_STANDARD_EP_HOT_SPLIT_WEIGHT_LAYOUT": "needed",
            "CANARY_STANDARD_EP_NEEDED_WEIGHT_EXCHANGE_IMPL": "primus_turbo",
            "CANARY_STANDARD_EP_NEEDED_WEIGHT_EXCHANGE_TRANSPORT": "mori_sdma",
            "CANARY_STANDARD_EP_HOT_SPLIT_WEIGHTED_SCATTER_BACKEND": "triton_bwd",
        },
    },
}


DEEPSEEK_SPARSE_ENV = {
    "CANARY_SPARSE_MLA_MAX_SEQ_LEN": "4096",
    "CANARY_SPARSE_MLA_COMPRESS_RATIO_ALLOWLIST": "4,128",
    "CANARY_SPARSE_MLA_BACKWARD_MODE": "shared_kv",
    "CANARY_SPARSE_MLA_KERNEL_MODULE": "torchtitan.models.deepseek_v4.sparse_mla_shared_kv_triton_forward_backend",
    "CANARY_SHAREDKV_BACKWARD_MODE": "query_recompute",
    "CANARY_SHAREDKV_BLOCK_H": "64",
    "CANARY_SHAREDKV_BLOCK_N": "64",
    "CANARY_SHAREDKV_BLOCK_D": "128",
    "CANARY_LI_KL_BACKEND": "triton_tl_dot",
}


COMMON_ENV = {
    "GPUS_PER_NODE": "8",
    "CANARY_EP": "8",
    "CANARY_DP_SHARD": "-1",
    "CANARY_GBS": "128",
    "CANARY_SEQ_LEN": "4096",
    "CANARY_CHECKPOINT_ENABLE": "false",
    "CANARY_ACTIVATION_CHECKPOINT_MODE": "selective",
    "CANARY_ACTIVATION_CHECKPOINT_SCOPE": "attention_only",
    "CANARY_CHUNKED_CE_NUM_CHUNKS": "8",
    "CANARY_REQUIRE_IDLE_GPUS": "true",
    "CANARY_ENABLE_STRUCTURED_LOGGING": "true",
    "CANARY_PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
}


def parse_csv(value: str, allowed: set[str], name: str) -> list[str]:
    items = [part.strip() for part in value.split(",") if part.strip()]
    bad = [item for item in items if item not in allowed]
    if bad:
        raise SystemExit(f"Unknown {name}: {bad}; allowed={sorted(allowed)}")
    return items


def run_cmd(
    cmd: list[str],
    *,
    env: dict[str, str] | None = None,
    check: bool = True,
    timeout: float | None = None,
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            cmd,
            cwd=PLAYGROUND_DIR,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=check,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout if isinstance(exc.stdout, str) else ""
        stdout += f"\n[timeout] command exceeded {timeout}s: {' '.join(cmd)}\n"
        return subprocess.CompletedProcess(cmd, 124, stdout)
    except OSError as exc:
        return subprocess.CompletedProcess(cmd, 125, f"[spawn_error] {exc}\n")


def remote_read(host: str, path: str) -> str:
    proc = run_cmd(["ssh", "-o", "ConnectTimeout=15", host, "cat", path], check=False, timeout=60)
    if proc.returncode != 0:
        return ""
    return proc.stdout


def remote_preflight(host: str) -> dict[str, str]:
    checks = {
        "docker_ps": "docker ps --format '{{.ID}} {{.Names}} {{.Image}} {{.Status}}'",
        "rocm_smi": "rocm-smi --showmemuse --showuse 2>/dev/null | sed -n '1,80p'",
    }
    out: dict[str, str] = {}
    for key, remote_cmd in checks.items():
        proc = run_cmd(["ssh", "-o", "ConnectTimeout=15", host, remote_cmd], check=False, timeout=60)
        out[key] = proc.stdout
    return out


def _active_docker_rows(preflight: dict[str, str], allowed_name_regex: str | None = None) -> list[str]:
    pattern = re.compile(allowed_name_regex) if allowed_name_regex else None
    active: list[str] = []
    for raw in preflight.get("docker_ps", "").splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = line.split(maxsplit=3)
        name = parts[1] if len(parts) > 1 else line
        if pattern and pattern.search(name):
            continue
        active.append(line)
    return active


def preflight_idle(preflight: dict[str, str], allowed_docker_name_regex: str | None = None) -> bool:
    if _active_docker_rows(preflight, allowed_docker_name_regex):
        return False
    rocm = preflight.get("rocm_smi", "")
    busy = re.findall(r"GPU use \(%\):\s*(\d+)", rocm)
    mem = re.findall(r"GPU Memory Allocated \(VRAM%\):\s*(\d+)", rocm)
    if busy and any(int(v) != 0 for v in busy):
        return False
    if mem and any(int(v) != 0 for v in mem):
        return False
    return True


def wait_for_idle(
    host: str,
    *,
    allowed_docker_name_regex: str | None = None,
    attempts: int = 12,
    sleep_seconds: float = 5.0,
) -> tuple[dict[str, str], bool]:
    """Poll node idleness while Docker/ROCm tear down after a completed row."""
    latest: dict[str, str] = {}
    for idx in range(attempts):
        latest = remote_preflight(host)
        if preflight_idle(latest, allowed_docker_name_regex):
            return latest, True
        if idx + 1 < attempts:
            time.sleep(sleep_seconds)
    return latest, False


def run_launcher_with_retries(args: argparse.Namespace, env: dict[str, str], run_id: str) -> subprocess.CompletedProcess[str]:
    last_proc: subprocess.CompletedProcess[str] | None = None
    LOCAL_LOG_DIR.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, args.launch_attempts + 1):
        local_log = LOCAL_LOG_DIR / f"{run_id}.launcher.log"
        with local_log.open("a", encoding="utf-8", errors="replace") as log_f:
            log_f.write(f"\n[sweep-local] attempt={attempt} started={dt.datetime.now(dt.timezone.utc).isoformat()}\n")
            log_f.flush()
            try:
                raw_proc = subprocess.run(
                    ["bash", str(LAUNCHER), "run"],
                    cwd=PLAYGROUND_DIR,
                    env=env,
                    text=True,
                    stdout=log_f,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
                returncode = raw_proc.returncode
            except OSError as exc:
                log_f.write(f"[spawn_error] {exc}\n")
                returncode = 125
            log_f.write(f"\n[sweep-local] attempt={attempt} returncode={returncode}\n")
        text = local_log.read_text(errors="replace")
        proc = subprocess.CompletedProcess(
            ["bash", str(LAUNCHER), "run"],
            returncode,
            text[-16000:],
        )
        last_proc = proc
        if proc.returncode == 0:
            return proc
        if not RESOURCE_FAILURE_RE.search(proc.stdout):
            return proc
        print(
            f"[sweep] transient launcher resource failure for run_id={run_id} "
            f"attempt={attempt}/{args.launch_attempts}; retrying local_log={local_log}",
            flush=True,
        )
        wait_for_idle(args.host, allowed_docker_name_regex=args.allowed_nongpu_docker_name_regex)
        if attempt < args.launch_attempts:
            time.sleep(args.launch_retry_sleep)
    assert last_proc is not None
    return last_proc


def parse_log(text: str) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    warning_lines = 0
    oom_lines: list[str] = []
    for raw in text.splitlines():
        if "CUDA memory allocation retries" in raw:
            warning_lines += 1
        if (
            "out of memory" in raw.lower()
            or "HSA_STATUS_ERROR_OUT_OF_RESOURCES" in raw
            or "memory access fault" in raw.lower()
            or "RuntimeError:" in raw
        ):
            oom_lines.append(ANSI_RE.sub("", raw)[-500:])
        line = ANSI_RE.sub("", raw)
        match = STEP_RE.search(line)
        if match:
            rows.append(
                {
                    "step": int(match.group("step")),
                    "loss": float(match.group("loss")),
                    "grad_norm": float(match.group("grad_norm")),
                    "memory_gib": float(match.group("memory")),
                    "tps": float(match.group("tps").replace(",", "")),
                    "tflops": float(match.group("tflops")),
                    "mfu": float(match.group("mfu")),
                }
            )

    by_step: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        by_step.setdefault(row["step"], []).append(row)

    step_stats: list[dict[str, Any]] = []
    for step in sorted(by_step):
        vals = by_step[step]
        step_stats.append(
            {
                "step": step,
                "rank_rows": len(vals),
                "tokens_per_gpu_s": statistics.fmean(v["tps"] for v in vals),
                "tokens_per_gpu_s_min": min(v["tps"] for v in vals),
                "tokens_per_gpu_s_max": max(v["tps"] for v in vals),
                "tflops_per_gpu": statistics.fmean(v["tflops"] for v in vals),
                "mfu_pct": statistics.fmean(v["mfu"] for v in vals),
                "memory_max_gib": max(v["memory_gib"] for v in vals),
                "memory_min_gib": min(v["memory_gib"] for v in vals),
                "loss": vals[0]["loss"],
                "grad_norm": vals[0]["grad_norm"],
            }
        )

    def window(start: int, end: int) -> dict[str, Any] | None:
        xs = [s for s in step_stats if start <= s["step"] <= end]
        if not xs:
            return None
        return {
            "steps": [start, end],
            "count": len(xs),
            "tokens_per_gpu_s": round(statistics.fmean(s["tokens_per_gpu_s"] for s in xs), 4),
            "node_tokens_per_s": round(8 * statistics.fmean(s["tokens_per_gpu_s"] for s in xs), 4),
            "tflops_per_gpu": round(statistics.fmean(s["tflops_per_gpu"] for s in xs), 4),
            "mfu_pct": round(statistics.fmean(s["mfu_pct"] for s in xs), 4),
            "peak_logged_memory_gib": round(max(s["memory_max_gib"] for s in xs), 4),
            "min_step_tokens_per_gpu_s": round(min(s["tokens_per_gpu_s"] for s in xs), 4),
            "max_step_tokens_per_gpu_s": round(max(s["tokens_per_gpu_s"] for s in xs), 4),
        }

    return {
        "completed_steps": len(step_stats),
        "rank_metric_rows": len(rows),
        "cuda_memory_allocation_retry_warning_lines": warning_lines,
        "error_lines": oom_lines[-8:],
        "steps": step_stats,
        "windows": {
            "steps_2_to_20": window(2, 20),
            "steps_13_to_20": window(13, 20),
        },
        "loss": {
            "step_1": by_step.get(1, [{}])[0].get("loss"),
            "last_step": step_stats[-1]["loss"] if step_stats else None,
        },
    }


def make_env(args: argparse.Namespace, model_key: str, variant_key: str, mbs: int, run_id: str) -> dict[str, str]:
    env = os.environ.copy()
    env.update(COMMON_ENV)
    env.update(
        {
            "TARGET_HOST": args.host,
            "IMAGE": args.image,
            "REMOTE_ROOT": args.remote_root,
            "CANARY_TORCHTITAN_MODULE": MODELS[model_key]["module"],
            "CANARY_TORCHTITAN_CONFIG": MODELS[model_key]["config"],
            "CANARY_MBS": str(mbs),
            "CANARY_STEPS": str(args.steps),
            "RUN_ID": run_id,
            "LOG_DIR": f"{args.remote_root}/runs/{run_id}/logs",
            "DUMP_FOLDER": f"{args.remote_root}/runs/{run_id}/outputs",
        }
    )
    model = MODELS[model_key]
    env.update(model.get("env", {}))
    if model.get("base") == "deepseek":
        env.update(DEEPSEEK_SPARSE_ENV)
    env.update(VARIANTS[variant_key]["env"])
    return env


def common_surface(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "hardware": "8xMI350",
        "sequence_length": 4096,
        "global_batch_size": 128,
        "parallelism": "TP1_PP1_CP1_FSDP8_EP8",
        "activation_checkpointing": "selective attention_only",
        "checkpoint_enable": False,
        "pytorch_cuda_alloc_conf": "expandable_segments:True",
        "loss": "Chunked CE8",
        "optimizer": "AdamW",
        "steps": args.steps,
        "retained_windows": ["steps_2_to_20", "steps_13_to_20"],
    }


def start_remote_tmux_driver(
    args: argparse.Namespace,
    *,
    run_prefix: str,
    artifact_name: str,
) -> int:
    """Stage this experiment and run the row loop inside remote tmux."""

    remote_snapshot = (
        f"{args.remote_root}/sweep_snapshots/{run_prefix}/playground"
    )
    remote_exp = (
        f"{remote_snapshot}/experiments/2026-05-27-deepseek-v4-amd-pretrain"
    )
    remote_log = f"{args.remote_root}/runs/{run_prefix}/logs/table-driver.log"
    remote_exit = f"{args.remote_root}/runs/{run_prefix}/table-driver.exit"
    tmux_session = args.remote_tmux_session or f"bm_table_{run_prefix[-24:]}"

    subprocess.run(
        [
            "ssh",
            "-o",
            "ConnectTimeout=15",
            args.host,
            (
                f"mkdir -p {shlex.quote(remote_exp)} "
                f"{shlex.quote(str(Path(remote_log).parent))}"
            ),
        ],
        cwd=PLAYGROUND_DIR,
        check=True,
    )
    source_dirs = (
        "tools",
        "launchers",
        "sources/targets",
        "sources/wip/torchtitan",
        "sources/wip/primus-turbo",
        "sources/references/mori",
    )
    source_files = ("README.md", "recipe.yaml")
    for rel in source_dirs:
        local_path = EXP_DIR / rel
        if not local_path.is_dir():
            raise FileNotFoundError(local_path)
        remote_path = f"{remote_exp}/{rel}"
        subprocess.run(
            [
                "ssh",
                "-o",
                "ConnectTimeout=15",
                args.host,
                f"mkdir -p {shlex.quote(remote_path)}",
            ],
            cwd=PLAYGROUND_DIR,
            check=True,
        )
        subprocess.run(
            [
                "rsync",
                "-a",
                "--delete",
                "--exclude",
                ".git",
                "--exclude",
                "__pycache__/",
                "--exclude",
                "build/",
                f"{local_path}/",
                f"{args.host}:{remote_path}/",
            ],
            cwd=PLAYGROUND_DIR,
            check=True,
        )
    for rel in source_files:
        local_path = EXP_DIR / rel
        if not local_path.is_file():
            continue
        subprocess.run(
            [
                "rsync",
                "-a",
                str(local_path),
                f"{args.host}:{remote_exp}/{rel}",
            ],
            cwd=PLAYGROUND_DIR,
            check=True,
        )

    remote_args = [
        "python3",
        f"{remote_exp}/tools/run_balanced_moe_table_sweep.py",
        "--host",
        args.remote_self_host,
        "--image",
        args.image,
        "--remote-root",
        args.remote_root,
        "--models",
        args.models,
        "--variants",
        args.variants,
        "--mbs",
        args.mbs,
        "--steps",
        str(args.steps),
        "--run-prefix",
        run_prefix,
        "--artifact-name",
        artifact_name,
        "--launch-attempts",
        str(args.launch_attempts),
        "--launch-retry-sleep",
        str(args.launch_retry_sleep),
        "--compare-tolerance-pct",
        str(args.compare_tolerance_pct),
        "--allowed-nongpu-docker-name-regex",
        args.allowed_nongpu_docker_name_regex,
    ]
    if args.compare_artifact:
        remote_args.extend(["--compare-artifact", args.compare_artifact])
    if args.dry_run:
        remote_args.append("--dry-run")
    remote_cmd = " ".join(shlex.quote(part) for part in remote_args)
    shell_cmd = (
        f"cd {shlex.quote(remote_snapshot)} && "
        f"BALANCED_MOE_SWEEP_LOCAL_LOG_DIR={shlex.quote(args.remote_root + '/runs/' + run_prefix + '/local_launcher_logs')} "
        f"{remote_cmd} > {shlex.quote(remote_log)} 2>&1; "
        f"echo $? > {shlex.quote(remote_exit)}"
    )
    subprocess.run(
        [
            "ssh",
            "-o",
            "ConnectTimeout=15",
            args.host,
            (
                f"tmux kill-session -t {shlex.quote(tmux_session)} "
                "2>/dev/null || true; "
                f"tmux new-session -d -s {shlex.quote(tmux_session)} "
                f"{shlex.quote(shell_cmd)}"
            ),
        ],
        cwd=PLAYGROUND_DIR,
        check=True,
    )

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    launch_record = {
        "artifact": artifact_name,
        "status": "remote_tmux_started",
        "host": args.host,
        "remote_self_host": args.remote_self_host,
        "tmux_session": tmux_session,
        "remote_snapshot": remote_snapshot,
        "remote_log": remote_log,
        "remote_exit": remote_exit,
        "remote_artifact": f"{remote_exp}/run_artifacts/{artifact_name}",
        "poll_commands": {
            "tmux": f"ssh {args.host} 'tmux has-session -t {tmux_session}; tmux capture-pane -pt {tmux_session} | tail -n 80'",
            "log": f"ssh {args.host} 'tail -n 120 {remote_log}; cat {remote_exit} 2>/dev/null || true'",
            "artifact": f"scp {args.host}:{remote_exp}/run_artifacts/{artifact_name} {ARTIFACT_DIR / artifact_name}",
        },
    }
    (ARTIFACT_DIR / f"{Path(artifact_name).stem}_remote_launch.json").write_text(
        json.dumps(launch_record, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(launch_record, indent=2, sort_keys=True))
    return 0


def load_baseline_rows(path: str | None) -> dict[tuple[str, str, int], dict[str, Any]]:
    if not path:
        return {}
    artifact_path = Path(path)
    if not artifact_path.is_absolute():
        artifact_path = PLAYGROUND_DIR / artifact_path
    data = json.loads(artifact_path.read_text())
    rows: dict[tuple[str, str, int], dict[str, Any]] = {}
    for row in data.get("results", []):
        key = (row.get("model_key"), row.get("variant_key"), int(row.get("micro_batch_size", -1)))
        if key[0] and key[1] and key[2] >= 0:
            rows[key] = row
    return rows


def _late_window_tokens(row: dict[str, Any]) -> float | None:
    win = row.get("summary", {}).get("windows", {}).get("steps_13_to_20")
    if not win:
        return None
    value = win.get("tokens_per_gpu_s")
    return float(value) if value is not None else None


def trend_checks(
    results: list[dict[str, Any]],
    *,
    baseline_rows: dict[tuple[str, str, int], dict[str, Any]] | None = None,
    compare_tolerance_pct: float = 8.0,
) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    for model in sorted({r["model_key"] for r in results}):
        for variant in sorted({r["variant_key"] for r in results}):
            rows = [r for r in results if r["model_key"] == model and r["variant_key"] == variant]
            rows.sort(key=lambda r: r["micro_batch_size"])
            prev = None
            for row in rows:
                current = _late_window_tokens(row)
                if row["status"] != "pass" or current is None:
                    prev = None
                    continue
                if prev and current < prev["tokens_per_gpu_s"] * 0.97:
                    findings.append(
                        {
                            "type": "non_monotonic_late_window",
                            "model": model,
                            "variant": variant,
                            "previous_mbs": prev["mbs"],
                            "previous_tokens_per_gpu_s": prev["tokens_per_gpu_s"],
                            "current_mbs": row["micro_batch_size"],
                            "current_tokens_per_gpu_s": current,
                            "drop_pct": round((current / prev["tokens_per_gpu_s"] - 1.0) * 100.0, 2),
                        }
                    )
                prev = {"mbs": row["micro_batch_size"], "tokens_per_gpu_s": current}
    if baseline_rows:
        for row in results:
            if row.get("status") != "pass":
                continue
            current = _late_window_tokens(row)
            if current is None:
                continue
            key = (row["model_key"], row["variant_key"], int(row["micro_batch_size"]))
            baseline = baseline_rows.get(key)
            if not baseline or baseline.get("status") != "pass":
                continue
            baseline_value = _late_window_tokens(baseline)
            if baseline_value is None or baseline_value <= 0:
                continue
            delta_pct = (current / baseline_value - 1.0) * 100.0
            if abs(delta_pct) > compare_tolerance_pct:
                findings.append(
                    {
                        "type": "baseline_late_window_delta",
                        "model": key[0],
                        "variant": key[1],
                        "mbs": key[2],
                        "baseline_tokens_per_gpu_s": round(baseline_value, 4),
                        "current_tokens_per_gpu_s": round(current, 4),
                        "delta_pct": round(delta_pct, 2),
                        "tolerance_pct": compare_tolerance_pct,
                    }
                )
    for row in results:
        retries = row.get("summary", {}).get("cuda_memory_allocation_retry_warning_lines", 0)
        if retries:
            findings.append(
                {
                    "type": "allocator_retry_warning",
                    "model": row["model_key"],
                    "variant": row["variant_key"],
                    "mbs": row["micro_batch_size"],
                    "warning_lines": retries,
                }
            )
    return findings


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="do-vunguyen-mi350-gpu")
    parser.add_argument("--image", default="onenexus/nexus-titan:rocm722-pytorch-nightly-mori")
    parser.add_argument("--remote-root", default="/scratch/sonle5/dsv4_pretrain_canary_20260527")
    parser.add_argument("--models", default="deepseek_mtp1,deepseek_nomtp,qwen,kimi")
    parser.add_argument(
        "--variants",
        default="plain,primus,helper8",
    )
    parser.add_argument("--mbs", default="1,2,4,8")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--run-prefix", default=None)
    parser.add_argument("--artifact-name", default=None)
    parser.add_argument("--launch-attempts", type=int, default=3)
    parser.add_argument("--launch-retry-sleep", type=float, default=45.0)
    parser.add_argument("--compare-artifact", default=None)
    parser.add_argument("--compare-tolerance-pct", type=float, default=8.0)
    parser.add_argument(
        "--allowed-nongpu-docker-name-regex",
        default=r"(lighthouse|torchft_lighthouse)",
        help=(
            "Docker container name regex allowed by clean-node checks only when "
            "ROCm reports zero GPU and zero VRAM activity. Use an empty string "
            "for strict no-container checks."
        ),
    )
    parser.add_argument("--start-remote-tmux", action="store_true")
    parser.add_argument("--remote-self-host", default="localhost")
    parser.add_argument("--remote-tmux-session", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-fail", action="store_true", default=True)
    args = parser.parse_args()

    models = parse_csv(args.models, set(MODELS), "models")
    variants = parse_csv(args.variants, set(VARIANTS), "variants")
    mbs_values = [int(item) for item in args.mbs.split(",") if item.strip()]
    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_prefix = args.run_prefix or f"balanced_moe_table_cleanrerun_{timestamp}"
    artifact_name = args.artifact_name or f"balanced_moe_backend_table_cleanrerun_{timestamp}.json"
    if args.start_remote_tmux:
        return start_remote_tmux_driver(
            args,
            run_prefix=run_prefix,
            artifact_name=artifact_name,
        )
    baseline_rows = load_baseline_rows(args.compare_artifact)

    if args.allowed_nongpu_docker_name_regex == "":
        args.allowed_nongpu_docker_name_regex = None

    preflight = remote_preflight(args.host)
    results: list[dict[str, Any]] = []
    planned: list[dict[str, Any]] = []

    for model in models:
        for mbs in mbs_values:
            for variant in variants:
                run_id = f"{run_prefix}_{model}_{variant}_mbs{mbs}_steps{args.steps}"
                env = make_env(args, model, variant, mbs, run_id)
                planned.append(
                    {
                        "model": model,
                        "variant": variant,
                        "mbs": mbs,
                        "run_id": run_id,
                        "local_launcher_log": str(LOCAL_LOG_DIR / f"{run_id}.launcher.log"),
                        "env": {k: env[k] for k in sorted(env) if k.startswith("CANARY_") or k in {"TARGET_HOST", "IMAGE", "REMOTE_ROOT", "RUN_ID", "LOG_DIR", "DUMP_FOLDER", "GPUS_PER_NODE"}},
                    }
                )

    if args.dry_run:
        print(json.dumps({"planned": planned, "preflight": preflight}, indent=2))
        return 0

    for item in planned:
        model = item["model"]
        variant = item["variant"]
        mbs = item["mbs"]
        run_id = item["run_id"]
        env = make_env(args, model, variant, mbs, run_id)
        before, before_idle = wait_for_idle(
            args.host,
            allowed_docker_name_regex=args.allowed_nongpu_docker_name_regex,
        )
        if not before_idle:
            results.append(
                {
                    "model_key": model,
                    "model": MODELS[model]["label"],
                    "variant_key": variant,
                    "variant": VARIANTS[variant]["label"],
                    "micro_batch_size": mbs,
                    "run_id": run_id,
                    "status": "blocked_node_not_idle_before_run",
                    "preflight": before,
                    "active_docker_rows": _active_docker_rows(
                        before,
                        args.allowed_nongpu_docker_name_regex,
                    ),
                }
            )
            print(f"[sweep] blocked before run; node is not idle for run_id={run_id}", flush=True)
            break

        print(f"[sweep] start model={model} variant={variant} mbs={mbs} run_id={run_id}", flush=True)
        proc = run_launcher_with_retries(args, env, run_id)
        print(proc.stdout[-8000:], flush=True)
        log_path = f"{args.remote_root}/runs/{run_id}/logs/docker-run.log"
        log_text = remote_read(args.host, log_path)
        summary = parse_log(log_text)
        status = "pass" if proc.returncode == 0 and summary["completed_steps"] >= args.steps else "fail"
        after, after_idle = wait_for_idle(
            args.host,
            allowed_docker_name_regex=args.allowed_nongpu_docker_name_regex,
        )
        result = {
            "model_key": model,
            "model": MODELS[model]["label"],
            "variant_key": variant,
            "variant": VARIANTS[variant]["label"],
            "micro_batch_size": mbs,
            "gradient_accumulation_steps": 128 // mbs // 8,
            "run_id": run_id,
            "local_launcher_log": str(LOCAL_LOG_DIR / f"{run_id}.launcher.log"),
            "remote_log": log_path,
            "remote_output_dir": f"{args.remote_root}/runs/{run_id}/outputs",
            "returncode": proc.returncode,
            "status": status,
            "summary": summary,
            "preflight_before": before,
            "postflight_after": after,
            "postflight_idle": after_idle,
            "postflight_active_docker_rows": _active_docker_rows(
                after,
                args.allowed_nongpu_docker_name_regex,
            ),
            "tail": proc.stdout[-4000:],
        }
        results.append(result)

        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        partial = {
            "artifact": artifact_name,
            "created_utc": timestamp,
            "status": "in_progress",
            "host": args.host,
            "image": args.image,
            "common_surface": common_surface(args),
            "preflight": preflight,
            "planned": planned,
            "results": results,
            "completion": {
                "planned_rows": len(planned),
                "result_rows": len(results),
                "complete": False,
            },
            "compare_artifact": args.compare_artifact,
            "compare_tolerance_pct": args.compare_tolerance_pct,
            "findings": trend_checks(
                results,
                baseline_rows=baseline_rows,
                compare_tolerance_pct=args.compare_tolerance_pct,
            ),
        }
        (ARTIFACT_DIR / artifact_name).write_text(json.dumps(partial, indent=2, sort_keys=True) + "\n")

        print(
            f"[sweep] done status={status} completed_steps={summary['completed_steps']} "
            f"retry_warnings={summary['cuda_memory_allocation_retry_warning_lines']} "
            f"postflight_idle={after_idle}",
            flush=True,
        )
        if not after_idle:
            print(f"[sweep] stopping after run_id={run_id}; postflight is not idle", flush=True)
            break
        if status != "pass" and not args.continue_on_fail:
            break

    blocked_rows = [r for r in results if str(r.get("status", "")).startswith("blocked_")]
    completed_all_rows = len(results) == len(planned) and not blocked_rows
    failed_rows = [r for r in results if r.get("status") != "pass"]
    if not completed_all_rows:
        table_status = "incomplete"
    elif failed_rows:
        table_status = "complete_with_failures"
    else:
        table_status = "complete"

    final = {
        "artifact": artifact_name,
        "created_utc": timestamp,
        "status": table_status,
        "host": args.host,
        "image": args.image,
        "common_surface": common_surface(args),
        "preflight": preflight,
        "planned": planned,
        "results": results,
        "completion": {
            "planned_rows": len(planned),
            "result_rows": len(results),
            "complete": completed_all_rows,
            "failed_rows": len(failed_rows),
            "blocked_rows": len(blocked_rows),
        },
        "compare_artifact": args.compare_artifact,
        "compare_tolerance_pct": args.compare_tolerance_pct,
        "findings": trend_checks(
            results,
            baseline_rows=baseline_rows,
            compare_tolerance_pct=args.compare_tolerance_pct,
        ),
    }
    (ARTIFACT_DIR / artifact_name).write_text(json.dumps(final, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"artifact": str(ARTIFACT_DIR / artifact_name), "findings": final["findings"]}, indent=2))
    return 0 if completed_all_rows else 2


if __name__ == "__main__":
    sys.exit(main())
