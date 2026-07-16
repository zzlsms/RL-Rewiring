#!/usr/bin/env python3
"""Project an RL update into the base model spectral space.

For each target linear layer, this script:
  1. computes Delta = W_RL - W_base,
  2. extracts a low-rank Delta component,
  3. projects that component into the base model SVD coordinates,
  4. saves W_base + Delta_projected as the SAR model.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import gc
from typing import Any


def project_layer(
    base_weight_cpu: torch.Tensor,
    rl_weight_cpu: torch.Tensor,
    base_svd_rank: int,
    delta_fraction: float,
    worker_id: int,
) -> torch.Tensor:
    import torch

    device = torch.device(f"cuda:{worker_id}" if torch.cuda.is_available() else "cpu")
    try:
        w_base = base_weight_cpu.to(device=device, dtype=torch.float32).detach()
        w_rl = rl_weight_cpu.to(device=device, dtype=torch.float32).detach()
        delta = w_rl - w_base

        u_delta, s_delta, vh_delta = torch.linalg.svd(delta, full_matrices=False)
        delta_rank = max(1, int(len(s_delta) * delta_fraction))
        u_delta = u_delta[:, :delta_rank]
        s_delta = torch.diag(s_delta[:delta_rank])
        vh_delta = vh_delta[:delta_rank, :]

        u_base, s_base, vh_base = torch.linalg.svd(w_base, full_matrices=False)
        rank = min(base_svd_rank, len(s_base))
        u_base = u_base[:, :rank]
        vh_base = vh_base[:rank, :]

        # M = U_base^T Delta_k V_base in base spectral coordinates.
        rewiring = (u_base.T @ u_delta) @ s_delta @ (vh_delta @ vh_base.T)
        delta_projected = u_base @ rewiring @ vh_base
        w_final = w_base + delta_projected

        result = w_final.to("cpu", dtype=base_weight_cpu.dtype)
        del w_base, w_rl, delta, u_delta, s_delta, vh_delta, u_base, s_base, vh_base
        del rewiring, delta_projected, w_final
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return result
    except Exception as exc:
        raise RuntimeError(f"SAR projection failed on worker {worker_id}") from exc


def collect_target_layers(base_model: Any, rl_model: Any, target_modules: list[str]) -> list[dict[str, Any]]:
    import torch.nn as nn

    tasks: list[dict[str, Any]] = []
    for name, module in base_model.named_modules():
        if isinstance(module, nn.Linear) and any(target in name for target in target_modules):
            try:
                rl_module = rl_model.get_submodule(name)
            except AttributeError:
                print(f"[WARN] Layer {name} not found in RL model; skipping.")
                continue
            if not isinstance(rl_module, nn.Linear):
                raise TypeError(f"Matching RL module is not linear: {name}")
            if module.weight.shape != rl_module.weight.shape:
                raise ValueError(
                    f"Weight shape mismatch for {name}: "
                    f"base={tuple(module.weight.shape)}, rl={tuple(rl_module.weight.shape)}"
                )
            tasks.append(
                {
                    "name": name,
                    "module": module,
                    "base_weight": module.weight.data,
                    "rl_weight": rl_module.weight.data,
                }
            )
    return tasks


def process_model(args: argparse.Namespace) -> None:
    import torch
    from tqdm import tqdm
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading base model: {args.base_model}")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        device_map="cpu",
        trust_remote_code=args.trust_remote_code,
        torch_dtype=torch.float32,
    )
    print(f"Loading RL model: {args.rl_model}")
    rl_model = AutoModelForCausalLM.from_pretrained(
        args.rl_model,
        device_map="cpu",
        trust_remote_code=args.trust_remote_code,
        torch_dtype=torch.float32,
    )
    base_model.requires_grad_(False)
    rl_model.requires_grad_(False)

    if not 0 < args.delta_fraction <= 1:
        raise ValueError("--delta_fraction must be in the interval (0, 1].")
    if args.svd_rank <= 0:
        raise ValueError("--svd_rank must be positive.")

    tasks = collect_target_layers(base_model, rl_model, args.target_modules)
    if not tasks:
        raise ValueError("No target linear layers were found. Check --target_modules.")

    num_workers = max(1, torch.cuda.device_count())
    print(f"Projecting {len(tasks)} linear layers with {num_workers} worker(s).")
    print(f"Target modules: {args.target_modules}")

    # One single-thread executor per device prevents concurrent SVDs from
    # competing for memory on the same GPU.
    executors = [
        concurrent.futures.ThreadPoolExecutor(max_workers=1)
        for _ in range(num_workers)
    ]
    try:
        future_to_task = {}
        for index, task in enumerate(tasks):
            worker_id = index % num_workers
            future = executors[worker_id].submit(
                project_layer,
                task["base_weight"],
                task["rl_weight"],
                args.svd_rank,
                args.delta_fraction,
                worker_id,
            )
            future_to_task[future] = task

        for future in tqdm(
            concurrent.futures.as_completed(future_to_task),
            total=len(future_to_task),
            desc="Projecting layers",
        ):
            task = future_to_task[future]
            task["module"].weight.data.copy_(future.result())
    finally:
        for executor in executors:
            executor.shutdown(wait=True, cancel_futures=True)

    del rl_model
    gc.collect()

    if hasattr(base_model, "generation_config"):
        base_model.generation_config.do_sample = True

    print(f"Saving SAR model to {args.save_path}")
    base_model.save_pretrained(args.save_path, safe_serialization=True)

    tokenizer_source = args.tokenizer_source or args.rl_model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=args.trust_remote_code)
    tokenizer.save_pretrained(args.save_path)
    print("SAR projection completed.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--rl_model", required=True)
    parser.add_argument("--save_path", required=True)
    parser.add_argument("--svd_rank", type=int, default=1_000_000)
    parser.add_argument("--delta_fraction", type=float, default=0.01)
    parser.add_argument("--target_modules", nargs="+", default=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ])
    parser.add_argument("--tokenizer_source", default=None)
    parser.add_argument("--trust_remote_code", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    process_model(parse_args())
