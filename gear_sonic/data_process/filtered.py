#!/usr/bin/env python3
"""Filter motion-lib PKLs with Humanoid_Batch FK.

This script recursively scans a robot_filtered motion directory, rejects clips
where both feet float above a z threshold for a continuous duration, rejects
clips whose FK-derived joint velocity exceeds a threshold, and copies passing
PKLs to an output directory while preserving the input-relative layout.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import math
import multiprocessing as mp
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import shutil
import sys
import traceback
from types import SimpleNamespace

try:
    from tqdm import tqdm
except ImportError:

    def tqdm(iterable=None, *args, **kwargs):
        return iterable if iterable is not None else []


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


LEFT_FOOT_BODY = "left_ankle_roll_link"
RIGHT_FOOT_BODY = "right_ankle_roll_link"

_worker_humanoid = None
_worker_joblib = None
_worker_torch = None
_worker_config = None
_worker_left_foot_idx = None
_worker_right_foot_idx = None


@dataclass(frozen=True)
class WorkerConfig:
    input_dir: str
    output_dir: str
    asset_root: str
    asset_file: str
    foot_z_threshold: float
    float_duration: float
    vel_threshold: float
    target_fps: int
    dry_run: bool
    torch_threads: int


@dataclass(frozen=True)
class FilterResult:
    rel_path: str
    status: str
    reasons: tuple[str, ...]
    motion_key: str = ""
    max_joint_velocity: float = 0.0
    longest_float_duration: float = 0.0
    copied: bool = False
    error: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Filter robot motion PKLs using Humanoid_Batch FK. Rejects motions with "
            "both feet floating for too long or joint velocities above a threshold."
        )
    )
    parser.add_argument(
        "--input_dir",
        default="data/motion_lib_bones_seed/robot_filtered",
        help="Input directory containing motion-lib PKL files.",
    )
    parser.add_argument(
        "--output_dir",
        default="data/motion_lib_bones_seed/robot_filtered_clean",
        help="Output directory for copied passing PKLs.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=os.cpu_count() or 1,
        help="Number of worker processes.",
    )
    parser.add_argument(
        "--foot_z_threshold",
        type=float,
        default=0.1,
        help="Reject when both feet stay above this world-z threshold.",
    )
    parser.add_argument(
        "--float_duration",
        type=float,
        default=1.0,
        help="Continuous floating duration in seconds required for rejection.",
    )
    parser.add_argument(
        "--vel_threshold",
        type=float,
        default=20.0,
        help="Reject when any absolute FK-derived joint velocity exceeds this value.",
    )
    parser.add_argument(
        "--target_fps",
        type=int,
        default=50,
        help="Target FPS passed to Humanoid_Batch.fk_batch interpolation.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Run filtering and reporting without copying passing files.",
    )
    parser.add_argument(
        "--report_path",
        default=None,
        help="Optional CSV path for rejected/error files and diagnostics.",
    )
    parser.add_argument(
        "--torch_threads",
        type=int,
        default=1,
        help="Torch CPU threads per worker; keep low to avoid oversubscription.",
    )
    return parser.parse_args()


def init_worker(config: WorkerConfig) -> None:
    """Initialize heavyweight FK state once per worker process."""
    global _worker_config
    global _worker_humanoid
    global _worker_joblib
    global _worker_torch
    global _worker_left_foot_idx
    global _worker_right_foot_idx

    _worker_config = config

    import joblib
    import torch

    from gear_sonic.utils.motion_lib import torch_humanoid_batch

    if config.torch_threads > 0:
        torch.set_num_threads(config.torch_threads)

    motion_cfg = SimpleNamespace(
        asset=SimpleNamespace(
            assetRoot=config.asset_root,
            assetFileName=config.asset_file,
            urdfFileName="",
        ),
        extend_config=[],
    )
    humanoid = torch_humanoid_batch.Humanoid_Batch(motion_cfg, device=torch.device("cpu"))

    _worker_joblib = joblib
    _worker_torch = torch
    _worker_humanoid = humanoid
    _worker_left_foot_idx = humanoid.body_names.index(LEFT_FOOT_BODY)
    _worker_right_foot_idx = humanoid.body_names.index(RIGHT_FOOT_BODY)


def load_motion_entries(pkl_path: Path) -> list[tuple[str, dict]]:
    """Load one PKL and return motion entries.

    The expected robot_filtered format is a single-motion dict:
    ``{motion_name: entry}``. For robustness, this also accepts a bare entry
    dict with ``root_trans_offset`` and ``pose_aa`` keys.
    """
    data = _worker_joblib.load(pkl_path)

    if isinstance(data, dict) and {"root_trans_offset", "pose_aa"}.issubset(data):
        return [(pkl_path.stem, data)]

    if not isinstance(data, dict) or not data:
        raise ValueError("PKL does not contain a non-empty motion dict")

    entries = []
    for motion_key, entry in data.items():
        if not isinstance(entry, dict):
            raise ValueError(f"Motion entry {motion_key!r} is not a dict")
        if "root_trans_offset" not in entry or "pose_aa" not in entry:
            raise KeyError(f"Motion entry {motion_key!r} missing root_trans_offset or pose_aa")
        entries.append((str(motion_key), entry))

    return entries


def longest_true_run(mask) -> int:
    longest = 0
    current = 0
    for value in mask:
        if bool(value):
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def evaluate_entry(entry: dict) -> tuple[list[str], float, float, int]:
    """Run FK and return reject reasons plus diagnostics for one motion entry."""
    torch = _worker_torch
    humanoid = _worker_humanoid
    config = _worker_config

    trans = torch.as_tensor(entry["root_trans_offset"], dtype=torch.float32)
    pose_aa = torch.as_tensor(entry["pose_aa"], dtype=torch.float32)
    fps = float(entry.get("fps", 30.0))

    if trans.ndim != 2 or trans.shape[-1] != 3:
        raise ValueError(f"root_trans_offset must have shape (T, 3), got {tuple(trans.shape)}")
    if pose_aa.ndim != 3 or pose_aa.shape[-1] != 3:
        raise ValueError(f"pose_aa must have shape (T, J, 3), got {tuple(pose_aa.shape)}")
    if trans.shape[0] != pose_aa.shape[0]:
        raise ValueError(
            "root_trans_offset and pose_aa frame counts differ: "
            f"{trans.shape[0]} vs {pose_aa.shape[0]}"
        )
    if trans.shape[0] == 0:
        raise ValueError("Motion has zero frames")

    with torch.no_grad():
        fk_result = humanoid.fk_batch(
            pose_aa[None],
            trans[None],
            return_full=True,
            fps=fps,
            target_fps=config.target_fps,
            interpolate_data=True,
        )

        fk_fps = int(
            getattr(fk_result, "fps", config.target_fps if fps != config.target_fps else fps)
        )
        foot_positions = fk_result.global_translation[
            0, :, [_worker_left_foot_idx, _worker_right_foot_idx], 2
        ]
        floating_mask = (
            (foot_positions[:, 0] > config.foot_z_threshold)
            & (foot_positions[:, 1] > config.foot_z_threshold)
        )
        longest_float_frames = longest_true_run(floating_mask.detach().cpu().tolist())
        longest_float_duration = longest_float_frames / max(fk_fps, 1)

        dof_vels = fk_result.dof_vels
        max_joint_velocity = float(dof_vels.abs().max().item()) if dof_vels.numel() else 0.0

    min_float_frames = int(math.ceil(config.float_duration * fk_fps))
    reasons = []
    if longest_float_frames >= min_float_frames:
        reasons.append("floating_feet")
    if max_joint_velocity > config.vel_threshold:
        reasons.append("joint_velocity")

    return reasons, max_joint_velocity, longest_float_duration, fk_fps


def process_file(path_text: str) -> FilterResult:
    config = _worker_config
    pkl_path = Path(path_text)
    input_dir = Path(config.input_dir)
    rel_path = pkl_path.relative_to(input_dir)

    try:
        entries = load_motion_entries(pkl_path)

        all_reasons = set()
        motion_keys = []
        max_joint_velocity = 0.0
        longest_float_duration = 0.0

        for motion_key, entry in entries:
            reasons, entry_max_vel, entry_float_duration, _fk_fps = evaluate_entry(entry)
            motion_keys.append(motion_key)
            all_reasons.update(reasons)
            max_joint_velocity = max(max_joint_velocity, entry_max_vel)
            longest_float_duration = max(longest_float_duration, entry_float_duration)

        if all_reasons:
            return FilterResult(
                rel_path=rel_path.as_posix(),
                status="rejected",
                reasons=tuple(sorted(all_reasons)),
                motion_key=";".join(motion_keys),
                max_joint_velocity=max_joint_velocity,
                longest_float_duration=longest_float_duration,
            )

        copied = False
        if not config.dry_run:
            dest_path = Path(config.output_dir) / rel_path
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(pkl_path, dest_path)
            copied = True

        return FilterResult(
            rel_path=rel_path.as_posix(),
            status="kept",
            reasons=(),
            motion_key=";".join(motion_keys),
            max_joint_velocity=max_joint_velocity,
            longest_float_duration=longest_float_duration,
            copied=copied,
        )
    except Exception as exc:  # noqa: BLE001
        return FilterResult(
            rel_path=rel_path.as_posix(),
            status="error",
            reasons=("error",),
            error=f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
        )


def is_relative_to(path: Path, maybe_parent: Path) -> bool:
    try:
        path.relative_to(maybe_parent)
        return True
    except ValueError:
        return False


def discover_motion_files(input_dir: Path, output_dir: Path) -> list[Path]:
    output_resolved = output_dir.resolve()
    files = []
    for pkl_path in input_dir.rglob("*.pkl"):
        if pkl_path.name == "metadata.pkl":
            continue
        if output_dir.exists() and is_relative_to(pkl_path.resolve(), output_resolved):
            continue
        files.append(pkl_path)
    return sorted(files)


def check_runtime_dependencies() -> list[str]:
    """Return missing modules needed by Humanoid_Batch and PKL loading."""
    required_modules = [
        "easydict",
        "hydra",
        "joblib",
        "loguru",
        "lxml",
        "numpy",
        "omegaconf",
        "open3d",
        "rich",
        "scipy",
        "torch",
    ]
    return [name for name in required_modules if importlib.util.find_spec(name) is None]


def write_report(report_path: Path, results: list[FilterResult]) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "status",
                "rel_path",
                "motion_key",
                "reasons",
                "max_joint_velocity",
                "longest_float_duration",
                "copied",
                "error",
            ],
        )
        writer.writeheader()
        for result in results:
            if result.status == "kept":
                continue
            writer.writerow(
                {
                    "status": result.status,
                    "rel_path": result.rel_path,
                    "motion_key": result.motion_key,
                    "reasons": ";".join(result.reasons),
                    "max_joint_velocity": f"{result.max_joint_velocity:.6f}",
                    "longest_float_duration": f"{result.longest_float_duration:.6f}",
                    "copied": int(result.copied),
                    "error": result.error,
                }
            )


def print_summary(args: argparse.Namespace, results: list[FilterResult]) -> None:
    status_counts = Counter(result.status for result in results)
    reason_counts = Counter()
    both_rules = 0

    for result in results:
        reasons = set(result.reasons)
        for reason in reasons:
            if reason != "error":
                reason_counts[reason] += 1
        if {"floating_feet", "joint_velocity"}.issubset(reasons):
            both_rules += 1

    print("\n" + "=" * 72)
    print("FK MOTION FILTER SUMMARY")
    print("=" * 72)
    print(f"Input directory:  {Path(args.input_dir)}")
    print(f"Output directory: {Path(args.output_dir)}")
    print(f"Dry run:          {args.dry_run}")
    print(f"Total scanned:    {len(results)}")
    print(f"Kept:             {status_counts.get('kept', 0)}")
    print(f"Rejected:         {status_counts.get('rejected', 0)}")
    print(f"Errors:           {status_counts.get('error', 0)}")
    print(f"floating_feet:    {reason_counts.get('floating_feet', 0)}")
    print(f"joint_velocity:   {reason_counts.get('joint_velocity', 0)}")
    print(f"Both rules:       {both_rules}")
    print(f"foot_z_threshold: {args.foot_z_threshold}")
    print(f"float_duration:   {args.float_duration}")
    print(f"vel_threshold:    {args.vel_threshold}")
    print(f"target_fps:       {args.target_fps}")

    if args.report_path:
        print(f"Report:           {Path(args.report_path)}")


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        print(f"ERROR: input_dir does not exist: {input_dir}", file=sys.stderr)
        return 1
    if not input_dir.is_dir():
        print(f"ERROR: input_dir is not a directory: {input_dir}", file=sys.stderr)
        return 1
    if input_dir.resolve() == output_dir.resolve():
        print("ERROR: output_dir must be different from input_dir", file=sys.stderr)
        return 1
    if args.workers < 1:
        print("ERROR: --workers must be >= 1", file=sys.stderr)
        return 1
    if args.target_fps < 1:
        print("ERROR: --target_fps must be >= 1", file=sys.stderr)
        return 1
    if args.float_duration <= 0:
        print("ERROR: --float_duration must be > 0", file=sys.stderr)
        return 1

    asset_root = REPO_ROOT / "gear_sonic/data/assets/robot_description/mjcf"
    asset_file = "g1_29dof_rev_1_0.xml"
    if not (asset_root / asset_file).exists():
        print(f"ERROR: MJCF file not found: {asset_root / asset_file}", file=sys.stderr)
        return 1

    missing_modules = check_runtime_dependencies()
    if missing_modules:
        print(
            "ERROR: missing runtime dependencies required by Humanoid_Batch: "
            + ", ".join(missing_modules),
            file=sys.stderr,
        )
        print(
            'Activate the project training environment or install with '
            '`pip install -e "gear_sonic/[training]"`.',
            file=sys.stderr,
        )
        return 1

    motion_files = discover_motion_files(input_dir, output_dir)
    if not motion_files:
        print(f"No motion PKL files found under {input_dir}")
        return 0

    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    workers = max(1, min(args.workers, len(motion_files)))
    config = WorkerConfig(
        input_dir=str(input_dir),
        output_dir=str(output_dir),
        asset_root=str(asset_root),
        asset_file=asset_file,
        foot_z_threshold=args.foot_z_threshold,
        float_duration=args.float_duration,
        vel_threshold=args.vel_threshold,
        target_fps=args.target_fps,
        dry_run=args.dry_run,
        torch_threads=args.torch_threads,
    )

    print(f"Found {len(motion_files)} motion PKLs")
    print(f"Using {workers} worker process(es)")
    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}")

    path_texts = [str(path) for path in motion_files]
    if workers == 1:
        init_worker(config)
        results = [
            process_file(path_text)
            for path_text in tqdm(path_texts, total=len(path_texts), desc="Filtering motions")
        ]
    else:
        with mp.Pool(processes=workers, initializer=init_worker, initargs=(config,)) as pool:
            results = list(
                tqdm(
                    pool.imap_unordered(process_file, path_texts, chunksize=16),
                    total=len(path_texts),
                    desc="Filtering motions",
                )
            )

    if args.report_path:
        write_report(Path(args.report_path), results)

    print_summary(args, results)
    return 1 if any(result.status == "error" for result in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
