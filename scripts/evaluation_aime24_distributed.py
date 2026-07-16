#!/usr/bin/env python3
"""Distributed AIME evaluator for vLLM OpenAI-compatible servers.

Each process evaluates one shard of the dataset and writes a shard-level JSON file.
Use `scripts/run_aime.sh` to launch multiple vLLM instances and evaluator
workers across nodes.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


def _fix_fracs(string: str) -> str:
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if substr and substr[0] == "{":
                new_str += substr
            else:
                if len(substr) < 2:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    new_str += "{" + a + "}{" + b + "}" + substr[2:]
                else:
                    new_str += "{" + a + "}" + b + substr[2:]
    return new_str


def _fix_a_slash_b(string: str) -> str:
    if len(string.split("/")) != 2:
        return string
    a, b = string.split("/")
    try:
        a_int = int(a)
        b_int = int(b)
        if string == f"{a_int}/{b_int}":
            return "\\frac{" + str(a_int) + "}{" + str(b_int) + "}"
    except ValueError:
        pass
    return string


def _remove_right_units(string: str) -> str:
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        if len(splits) == 2:
            return splits[0]
    return string


def _fix_sqrt(string: str) -> str:
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if split and split[0] != "{":
            new_string += "\\sqrt{" + split[0] + "}" + split[1:]
        else:
            new_string += "\\sqrt" + split
    return new_string


def strip_math_answer(string: Any) -> str:
    string = str(string).replace("\n", "")
    string = string.replace("\\!", "")
    string = string.replace("\\\\", "\\")
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")
    string = string.replace("\\$", "")
    string = _remove_right_units(string)
    string = string.replace("\\%", "")
    string = string.replace("\%", "")
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")

    if not string:
        return string
    if string[0] == ".":
        string = "0" + string

    if len(string.split("=")) == 2 and len(string.split("=")[0]) <= 2:
        string = string.split("=")[1]

    string = _fix_sqrt(string)
    string = string.replace(" ", "")
    string = _fix_fracs(string)
    if string == "0.5":
        string = "\\frac{1}{2}"
    string = _fix_a_slash_b(string)
    if re.fullmatch(r"-?\d+", string):
        sign = "-" if string.startswith("-") else ""
        digits = string.lstrip("-").lstrip("0") or "0"
        string = sign + digits
    return string


def is_equiv(prediction: str | None, answer: str | None) -> bool:
    if prediction is None and answer is None:
        return True
    if prediction is None or answer is None:
        return False
    try:
        return strip_math_answer(prediction) == strip_math_answer(answer)
    except Exception:
        return prediction == answer


def remove_boxed(s: str | None) -> str | None:
    if not s:
        return None
    left = "\\boxed{"
    if s.startswith(left) and s.endswith("}"):
        return s[len(left) : -1]
    return None


def last_boxed_only_string(string: str) -> str | None:
    idx = string.rfind("\\boxed")
    if idx < 0:
        return None
    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1
    if right_brace_idx is None:
        return None
    return string[idx : right_brace_idx + 1]


def extract_candidates(text: str) -> list[str]:
    if not text:
        return []

    answers: list[str] = []
    boxed_answer = last_boxed_only_string(text)
    if boxed_answer is not None:
        content = remove_boxed(boxed_answer)
        if content:
            answers.append(content)

    rl_zero_matches = re.findall(r"Answer:\s*(\$)?\s*(-?[\d\.]+)", text, re.IGNORECASE)
    if rl_zero_matches:
        answers.append(rl_zero_matches[-1][1])
    else:
        numbers = re.findall(r"-?\d+(?:\.\d+)?", text)
        if numbers:
            answers.append(numbers[-1])

    if not answers:
        dollars = [m.start() for m in re.finditer(r"\\\$|\$", text)]
        if len(dollars) > 1:
            answers.append(text[dollars[-2] + 1 : dollars[-1]])

    if not answers:
        answers.append(text)
    return answers


def generate_batch(
    messages: list[dict[str, str]],
    model_name: str,
    url: str,
    n_samples: int,
    temperature: float,
    top_p: float,
    max_tokens: int,
    request_timeout: int,
) -> list[str]:
    import requests

    payload = {
        "model": model_name,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "n": n_samples,
    }
    response = requests.post(
        f"{url.rstrip('/')}/v1/chat/completions",
        headers={"Content-Type": "application/json"},
        json=payload,
        timeout=request_timeout,
    )
    response.raise_for_status()
    return [choice["message"]["content"] for choice in response.json()["choices"]]


def get_generation_lengths(outputs: list[str], tokenizer: Any) -> list[int]:
    return [len(tokenizer.encode(output, add_special_tokens=False)) for output in outputs]


def load_dataset_with_retries(dataset_name: str, split: str, attempts: int, wait_seconds: int) -> Any:
    from datasets import load_dataset

    for attempt in range(attempts):
        try:
            return load_dataset(dataset_name, split=split)
        except Exception as exc:
            if attempt == attempts - 1:
                raise
            print(f"Dataset load failed ({exc}). Retrying in {wait_seconds}s...", flush=True)
            time.sleep(wait_seconds)
    raise RuntimeError("unreachable")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_id", required=True, help="Tokenizer path/id, usually the evaluated model path.")
    parser.add_argument("--served_model_name", default="sar-model", help="vLLM served model name.")
    parser.add_argument("--vllm_url", default="http://localhost:8000")
    parser.add_argument("--dataset", default="HuggingFaceH4/aime_2024")
    parser.add_argument("--split", default="train")
    parser.add_argument("--num_problems", type=int, default=-1)
    parser.add_argument("--k", type=int, default=32, dest="n_samples")
    parser.add_argument("--sub_batch_size", type=int, default=16, help="Number of samples per API call.")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_tokens", type=int, default=28672)
    parser.add_argument("--request_timeout", type=int, default=3600)
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/aime24"))
    parser.add_argument("--hf_endpoint", default=os.environ.get("HF_ENDPOINT"))
    parser.add_argument("--hf_hub_cache", default=os.environ.get("HF_HUB_CACHE"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from tqdm import tqdm
    from transformers import AutoTokenizer

    if args.hf_endpoint:
        os.environ["HF_ENDPOINT"] = args.hf_endpoint
    if args.hf_hub_cache:
        os.environ["HF_HUB_CACHE"] = args.hf_hub_cache

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / f"shard_{args.shard_id}_of_{args.num_shards}_k{args.n_samples}.json"

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)

    wait_time = args.shard_id * 5
    print(f"[Worker {args.shard_id}] waiting {wait_time}s before dataset load...", flush=True)
    time.sleep(wait_time)

    dataset = load_dataset_with_retries(args.dataset, args.split, attempts=5, wait_seconds=20)
    if args.num_problems > 0:
        dataset = dataset.select(range(min(len(dataset), args.num_problems)))
    expected_problems = len(dataset)
    dataset = dataset.shard(num_shards=args.num_shards, index=args.shard_id)

    print(
        f"Worker {args.shard_id}/{args.num_shards}: processing {len(dataset)} problems.",
        flush=True,
    )

    def process_example(example: dict[str, Any]) -> dict[str, Any]:
        q_key = "problem" if "problem" in example else "question"
        a_key = "answer" if "answer" in example else "solution"
        raw_answer = str(example[a_key]).strip()
        ground_truth = remove_boxed(last_boxed_only_string(raw_answer) or "") or raw_answer
        return {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        f"Problem: {example[q_key]}\n\n"
                        "Solve this math problem step by step, and put your final answer in \\boxed{}."
                    ),
                }
            ],
            "ground_truth": ground_truth,
            "problem_text": example[q_key],
        }

    dataset = dataset.map(process_example)

    detailed_results: list[dict[str, Any]] = []
    total_tokens = 0
    total_samples = 0
    failed_problems = 0

    def write_output() -> None:
        output_data = {
            "config": {
                "dataset": args.dataset,
                "split": args.split,
                "expected_problems": expected_problems,
                "shard_id": args.shard_id,
                "num_shards": args.num_shards,
                "samples_per_problem": args.n_samples,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "max_tokens": args.max_tokens,
            },
            "metrics": {
                "avg_length_overall": total_tokens / total_samples if total_samples else 0,
                "problems_completed": len(detailed_results),
                "failed_problems": failed_problems,
                "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            },
            "detailed_results": detailed_results,
        }
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

    # Create a shard file even when this worker receives no examples.
    write_output()

    for idx, row in enumerate(tqdm(dataset, desc=f"Shard {args.shard_id}")):
        try:
            generations: list[str] = []
            remaining = args.n_samples
            while remaining > 0:
                batch_n = min(args.sub_batch_size, remaining)
                generations.extend(
                    generate_batch(
                        messages=row["messages"],
                        model_name=args.served_model_name,
                        url=args.vllm_url,
                        n_samples=batch_n,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        max_tokens=args.max_tokens,
                        request_timeout=args.request_timeout,
                    )
                )
                remaining -= batch_n

            gen_lens = get_generation_lengths(generations, tokenizer)
            total_tokens += sum(gen_lens)
            total_samples += len(gen_lens)
            avg_tokens_problem = sum(gen_lens) / len(gen_lens) if gen_lens else 0

            correct_count = 0
            extracted_answers: list[str] = []
            for generation in generations:
                candidates = extract_candidates(generation)
                matched = False
                for candidate in candidates:
                    if is_equiv(candidate, row["ground_truth"]):
                        matched = True
                        extracted_answers.append(candidate)
                        break
                if matched:
                    correct_count += 1
                elif candidates:
                    extracted_answers.append(candidates[0])
                else:
                    extracted_answers.append("[No answer found]")

            problem_acc = correct_count / args.n_samples
            print(
                f"[{datetime.now().strftime('%H:%M:%S')}] "
                f"Shard {args.shard_id} | Problem {idx + 1}/{len(dataset)} | "
                f"GT={row['ground_truth']} | Correct={correct_count}/{args.n_samples} "
                f"({problem_acc:.1%})",
                flush=True,
            )

            detailed_results.append(
                {
                    "problem": row["problem_text"],
                    "ground_truth": row["ground_truth"],
                    "correct_count": correct_count,
                    "total_samples": args.n_samples,
                    "avg_tokens": avg_tokens_problem,
                    "generated_answers": extracted_answers,
                }
            )

            write_output()

        except Exception as exc:
            failed_problems += 1
            print(f"Error at shard {args.shard_id}, problem {idx}: {exc}", file=sys.stderr, flush=True)
            write_output()

    print(f"Worker {args.shard_id} finished. Results saved to {output_path}")
    if failed_problems:
        raise RuntimeError(f"{failed_problems} problem(s) failed; inspect the shard log.")


if __name__ == "__main__":
    main()
