import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import torch

import extract_semantics_qwen_structured_all_indicator as base


def parse_gpus(gpus_arg: str | None) -> list[int]:
    if gpus_arg is None or str(gpus_arg).strip() == "":
        count = int(torch.cuda.device_count())
        if count <= 0:
            raise RuntimeError("No visible CUDA devices found. Pass --gpus explicitly only after GPUs are available.")
        return list(range(count))
    return [int(x.strip()) for x in str(gpus_arg).split(",") if x.strip() != ""]


def count_items(json_path: str) -> int:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{json_path} does not contain a list.")
    return len(data)


def shard_ranges(start_idx: int, end_idx: int, n_shards: int) -> list[tuple[int, int]]:
    total = max(0, end_idx - start_idx)
    if total <= 0:
        return []
    step = int(math.ceil(total / float(n_shards)))
    out = []
    for shard_id in range(n_shards):
        s = start_idx + shard_id * step
        e = min(end_idx, s + step)
        if s < e:
            out.append((s, e))
    return out


def shard_output_path(out_json: str, shard_id: int, n_shards: int) -> str:
    out_path = Path(out_json)
    return str(out_path.with_name(f"{out_path.name}.shard{shard_id}of{n_shards}.json"))


def merge_shards(out_json: str, shard_paths: list[str]) -> None:
    merged: dict[int, dict] = {}
    for path in shard_paths:
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            continue
        for row in data:
            if not isinstance(row, dict):
                continue
            sample_index = int(row.get("sample_index", -1))
            if sample_index < 0:
                continue
            merged[sample_index] = row
    ordered = [merged[idx] for idx in sorted(merged.keys())]
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(ordered, f, ensure_ascii=False, indent=2)


def run_worker(args) -> None:
    base.MODEL_NAME = args.model_name
    base.BASE_IMG = args.base_img
    base.main(
        json_path=args.json_path,
        out_json=args.out_json,
        start_idx=int(args.start_idx),
        end_idx=int(args.end_idx),
        max_new_tokens=int(args.max_new_tokens),
        flush_every=int(args.flush_every),
        resume=bool(args.resume),
    )


def run_launcher(args) -> None:
    gpus = parse_gpus(args.gpus)
    total_items = count_items(args.json_path)
    start_idx = int(args.start_idx)
    end_idx = total_items if args.end_idx is None else min(int(args.end_idx), total_items)
    ranges = shard_ranges(start_idx, end_idx, len(gpus))
    if not ranges:
        raise ValueError(f"Empty range: start_idx={start_idx}, end_idx={end_idx}, total_items={total_items}")

    shard_paths = []
    procs = []

    for shard_id, ((s, e), gpu_id) in enumerate(zip(ranges, gpus, strict=False)):
        shard_out = shard_output_path(args.out_json, shard_id, len(ranges))
        shard_paths.append(shard_out)
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            "--json_path",
            args.json_path,
            "--out_json",
            shard_out,
            "--model_name",
            args.model_name,
            "--base_img",
            args.base_img,
            "--start_idx",
            str(s),
            "--end_idx",
            str(e),
            "--max_new_tokens",
            str(args.max_new_tokens),
            "--flush_every",
            str(args.flush_every),
        ]
        if args.resume:
            cmd.append("--resume")
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        print(f"[INFO] launching shard {shard_id}/{len(ranges)} on GPU {gpu_id}: [{s}, {e}) -> {shard_out}")
        procs.append(subprocess.Popen(cmd, env=env))

    failed = False
    for proc in procs:
        code = proc.wait()
        if code != 0:
            failed = True

    if failed:
        raise RuntimeError("At least one shard process failed. Inspect shard logs above and resume if needed.")

    merge_shards(args.out_json, shard_paths)
    print(f"[INFO] merged {len(shard_paths)} shards -> {args.out_json}")

    if not args.keep_shards:
        for path in shard_paths:
            if os.path.exists(path):
                os.remove(path)
        print("[INFO] shard files removed")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the structured Qwen semantic extractor in parallel across multiple GPUs and merge outputs."
    )
    parser.add_argument("--json_path", type=str, required=True)
    parser.add_argument("--out_json", type=str, required=True)
    parser.add_argument("--model_name", type=str, default=base.MODEL_NAME)
    parser.add_argument("--base_img", type=str, default=base.BASE_IMG)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--flush_every", type=int, default=10)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--gpus", type=str, default=None, help="Comma-separated GPU ids. Default: all visible GPUs.")
    parser.add_argument("--keep_shards", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    if args.worker:
        run_worker(args)
    else:
        run_launcher(args)
