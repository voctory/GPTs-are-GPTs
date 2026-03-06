#!/usr/bin/env python3
"""Batch GPT-5.4 exposure rerun pipeline using the Codex CLI.

This repo only contains the downstream analysis notebooks and saved GPT-4-era
labels. The original model-calling code is not present, so this script adds a
resume-safe rerun path that:

1. Reads the existing task universe from ``data/full_onet_data.tsv``.
2. Creates batch JSON files for Codex classification.
3. Calls ``codex exec`` once per batch using a strict JSON schema.
4. Merges normalized results back into a ``full_labelset``-style TSV.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from collections import Counter
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_PATH = REPO_ROOT / "data" / "full_onet_data.tsv"
DEFAULT_BASE_LABELSET = REPO_ROOT / "data" / "full_labelset.tsv"
DEFAULT_PROMPT_PATH = REPO_ROOT / "code" / "gpt54_exposure_prompt.txt"
DEFAULT_SCHEMA_PATH = REPO_ROOT / "code" / "gpt54_exposure_schema.json"
DEFAULT_WORK_DIR = REPO_ROOT / ".gpt54_rerun"
DEFAULT_OUTPUT_PATH = REPO_ROOT / "data" / "full_labelset_gpt54.tsv"
DEFAULT_TASK_DELTA_PATH = REPO_ROOT / "data" / "gpt54_vs_gpt4_task_delta.tsv"
DEFAULT_OCC_DELTA_PATH = REPO_ROOT / "data" / "gpt54_vs_gpt4_occ_delta.tsv"
DEFAULT_SUMMARY_PATH = REPO_ROOT / "data" / "gpt54_vs_gpt4_summary.json"

DEFAULT_PROMPT_VERSION = "gpt54_early2026_agentic_rubric_v2"
DEFAULT_MODEL = "gpt-5.4"
DEFAULT_REASONING_EFFORT = "medium"
DEFAULT_SHARD_SIZE = 100

RAW_ID_FIELD = ""
NORMALIZED_ID_FIELD = "row_id"
TASK_COLUMNS = [
    "O*NET-SOC Code",
    "Task ID",
    "Task",
    "Task Type",
    "Title",
]
EXTRA_RESULT_COLUMNS = [
    "gpt54_exposure",
    "gpt54_reason",
    "gpt54_confidence",
    "gpt54_model",
    "gpt54_reasoning_effort",
    "gpt54_prompt_version",
    "gpt54_batch_id",
    "gpt54_alpha",
    "gpt54_beta",
    "gpt54_gamma",
]
LABEL_TO_SCORES = {
    "E0": {"alpha": "0.0", "beta": "0.0", "gamma": "0.0"},
    "E1": {"alpha": "1.0", "beta": "1.0", "gamma": "1.0"},
    "E2": {"alpha": "0.0", "beta": "0.5", "gamma": "1.0"},
}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_tsv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        rows: List[Dict[str, str]] = []
        for index, raw_row in enumerate(reader):
            normalized: Dict[str, str] = {}
            for key, value in raw_row.items():
                normalized_key = key if key not in ("", None) else NORMALIZED_ID_FIELD
                normalized[normalized_key] = value if value is not None else ""
            if not normalized.get(NORMALIZED_ID_FIELD):
                normalized[NORMALIZED_ID_FIELD] = str(index)
            rows.append(normalized)
    return rows


def to_tsv_fieldnames(rows: Sequence[Dict[str, str]]) -> List[str]:
    if not rows:
        return [NORMALIZED_ID_FIELD]
    seen: List[str] = []
    for key in rows[0].keys():
        if key == RAW_ID_FIELD:
            continue
        normalized_key = key if key else NORMALIZED_ID_FIELD
        if normalized_key not in seen:
            seen.append(normalized_key)
    if NORMALIZED_ID_FIELD not in seen:
        seen.insert(0, NORMALIZED_ID_FIELD)
    return seen


def load_row_ids(path: Path | None) -> set[str] | None:
    if path is None:
        return None
    with path.open("r", encoding="utf-8") as handle:
        return {line.strip() for line in handle if line.strip()}


def make_batch_payload(rows: Sequence[Dict[str, str]]) -> List[Dict[str, str]]:
    payload = []
    for row in rows:
        payload.append(
            {
                "row_id": row[NORMALIZED_ID_FIELD],
                "onet_soc_code": row["O*NET-SOC Code"],
                "task_id": row["Task ID"],
                "title": row["Title"],
                "task_type": row["Task Type"],
                "task": row["Task"],
            }
        )
    return payload


def slug_for_batch(index: int) -> str:
    return f"batch-{index:05d}"


def create_batches(
    input_path: Path,
    work_dir: Path,
    shard_size: int,
    start_index: int,
    limit: int | None,
    row_id_filter: set[str] | None,
    prompt_version: str,
) -> None:
    rows = read_tsv_rows(input_path)
    selected_rows = rows[start_index:]
    if row_id_filter is not None:
        selected_rows = [
            row for row in selected_rows if row[NORMALIZED_ID_FIELD] in row_id_filter
        ]
    if limit is not None:
        selected_rows = selected_rows[:limit]

    batches_dir = work_dir / "batches"
    ensure_dir(batches_dir)

    # Remove stale batch definitions so repeated runs are deterministic.
    for existing in batches_dir.glob("batch-*.json"):
        existing.unlink()

    batch_count = 0
    for offset in range(0, len(selected_rows), shard_size):
        batch_rows = selected_rows[offset : offset + shard_size]
        batch_id = slug_for_batch(batch_count)
        payload = {
            "batch_id": batch_id,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "prompt_version": prompt_version,
            "source_path": str(input_path),
            "rows": make_batch_payload(batch_rows),
        }
        batch_path = batches_dir / f"{batch_id}.json"
        batch_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        batch_count += 1

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_path": str(input_path),
        "prompt_version": prompt_version,
        "total_source_rows": len(rows),
        "selected_rows": len(selected_rows),
        "shard_size": shard_size,
        "batch_count": batch_count,
        "start_index": start_index,
        "limit": limit,
        "row_id_file_applied": row_id_filter is not None,
    }
    (work_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(
        f"Wrote {batch_count} batch file(s) covering {len(selected_rows)} row(s) to "
        f"{batches_dir}"
    )


def build_prompt(prompt_template_path: Path, batch_rows: Sequence[Dict[str, str]]) -> str:
    template = prompt_template_path.read_text(encoding="utf-8").strip()
    rows_json = json.dumps(batch_rows, ensure_ascii=False, indent=2)
    return (
        f"{template}\n\n"
        "Batch rows (JSON):\n"
        f"{rows_json}\n"
    )


def run_codex(
    prompt_text: str,
    schema_path: Path,
    output_path: Path,
    stdout_log_path: Path,
    stderr_log_path: Path,
    model: str,
    reasoning_effort: str,
) -> None:
    command = [
        "codex",
        "exec",
        "--ephemeral",
        "-m",
        model,
        "-c",
        f'model_reasoning_effort="{reasoning_effort}"',
        "--output-schema",
        str(schema_path),
        "-o",
        str(output_path),
        "-",
    ]
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        input=prompt_text,
        text=True,
        capture_output=True,
        check=False,
    )
    stdout_log_path.write_text(completed.stdout, encoding="utf-8")
    stderr_log_path.write_text(completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(
            f"codex exec failed with exit code {completed.returncode}. "
            f"See {stdout_log_path} and {stderr_log_path}."
        )


def validate_batch_result(
    batch_rows: Sequence[Dict[str, str]],
    payload: Dict[str, object],
) -> List[Dict[str, str]]:
    if not isinstance(payload, dict) or "results" not in payload:
        raise ValueError("Codex response is missing a top-level 'results' array.")
    raw_results = payload["results"]
    if not isinstance(raw_results, list):
        raise ValueError("Codex response 'results' must be an array.")

    expected_by_row_id = {row["row_id"]: row for row in batch_rows}
    normalized_results: List[Dict[str, str]] = []
    seen_row_ids: set[str] = set()
    for item in raw_results:
        if not isinstance(item, dict):
            raise ValueError("Every result item must be an object.")
        required_fields = [
            "row_id",
            "onet_soc_code",
            "task_id",
            "label",
            "reason",
            "confidence",
        ]
        missing = [field for field in required_fields if field not in item]
        if missing:
            raise ValueError(f"Result is missing required fields: {missing}")
        row_id = str(item["row_id"])
        if row_id in seen_row_ids:
            raise ValueError(f"Duplicate row_id in result payload: {row_id}")
        if row_id not in expected_by_row_id:
            raise ValueError(f"Unexpected row_id in result payload: {row_id}")

        source = expected_by_row_id[row_id]
        if str(item["onet_soc_code"]) != source["onet_soc_code"]:
            raise ValueError(f"onet_soc_code mismatch for row_id {row_id}")
        if str(item["task_id"]) != source["task_id"]:
            raise ValueError(f"task_id mismatch for row_id {row_id}")
        label = str(item["label"])
        confidence = str(item["confidence"])
        if label not in LABEL_TO_SCORES:
            raise ValueError(f"Unsupported label {label!r} for row_id {row_id}")
        if confidence not in {"low", "medium", "high"}:
            raise ValueError(f"Unsupported confidence {confidence!r} for row_id {row_id}")
        normalized_results.append(
            {
                "row_id": row_id,
                "onet_soc_code": str(item["onet_soc_code"]),
                "task_id": str(item["task_id"]),
                "label": label,
                "reason": str(item["reason"]).strip(),
                "confidence": confidence,
            }
        )
        seen_row_ids.add(row_id)

    missing_row_ids = set(expected_by_row_id) - seen_row_ids
    if missing_row_ids:
        sample = ", ".join(sorted(missing_row_ids)[:10])
        raise ValueError(f"Missing result rows: {sample}")
    return normalized_results


def load_batch(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def classify_batch(
    batch_path: Path,
    prompt_template_path: Path,
    schema_path: Path,
    work_dir: Path,
    model: str,
    reasoning_effort: str,
    prompt_version: str,
    force: bool,
) -> None:
    payload = load_batch(batch_path)
    batch_id = str(payload["batch_id"])
    batch_rows = payload["rows"]
    if not isinstance(batch_rows, list):
        raise ValueError(f"{batch_path} is missing a 'rows' array.")

    results_dir = work_dir / "results"
    raw_dir = work_dir / "raw_last_messages"
    logs_dir = work_dir / "logs"
    ensure_dir(results_dir)
    ensure_dir(raw_dir)
    ensure_dir(logs_dir)

    normalized_result_path = results_dir / f"{batch_id}.json"
    if normalized_result_path.exists() and not force:
        print(f"Skipping {batch_id}; normalized result already exists.")
        return

    prompt_text = build_prompt(prompt_template_path, batch_rows)
    raw_last_message_path = raw_dir / f"{batch_id}.json"
    stdout_log_path = logs_dir / f"{batch_id}.stdout.log"
    stderr_log_path = logs_dir / f"{batch_id}.stderr.log"

    run_codex(
        prompt_text=prompt_text,
        schema_path=schema_path,
        output_path=raw_last_message_path,
        stdout_log_path=stdout_log_path,
        stderr_log_path=stderr_log_path,
        model=model,
        reasoning_effort=reasoning_effort,
    )

    raw_payload = json.loads(raw_last_message_path.read_text(encoding="utf-8"))
    normalized_results = validate_batch_result(batch_rows, raw_payload)
    output_payload = {
        "batch_id": batch_id,
        "model": model,
        "reasoning_effort": reasoning_effort,
        "prompt_version": prompt_version,
        "source_batch_path": str(batch_path),
        "results": normalized_results,
    }
    normalized_result_path.write_text(
        json.dumps(output_payload, indent=2), encoding="utf-8"
    )
    print(f"Saved normalized results for {batch_id} -> {normalized_result_path}")


def iter_batch_paths(batches_dir: Path) -> Iterable[Path]:
    return sorted(batches_dir.glob("batch-*.json"))


def classify_all(
    work_dir: Path,
    prompt_template_path: Path,
    schema_path: Path,
    model: str,
    reasoning_effort: str,
    prompt_version: str,
    force: bool,
    max_batches: int | None,
) -> None:
    batches_dir = work_dir / "batches"
    batch_paths = list(iter_batch_paths(batches_dir))
    if max_batches is not None:
        batch_paths = batch_paths[:max_batches]
    if not batch_paths:
        raise FileNotFoundError(
            f"No batch files found in {batches_dir}. Run 'make-batches' first."
        )
    for batch_path in batch_paths:
        classify_batch(
            batch_path=batch_path,
            prompt_template_path=prompt_template_path,
            schema_path=schema_path,
            work_dir=work_dir,
            model=model,
            reasoning_effort=reasoning_effort,
            prompt_version=prompt_version,
            force=force,
        )


def load_results(results_dirs: Sequence[Path]) -> Dict[str, Dict[str, str]]:
    result_map: Dict[str, Dict[str, str]] = {}
    for results_dir in results_dirs:
        for path in sorted(results_dir.glob("batch-*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            batch_id = str(payload["batch_id"])
            model = str(payload["model"])
            reasoning_effort = str(payload["reasoning_effort"])
            prompt_version = str(payload["prompt_version"])
            for item in payload["results"]:
                row_id = item["row_id"]
                result_map[row_id] = {
                    "gpt54_exposure": item["label"],
                    "gpt54_reason": item["reason"],
                    "gpt54_confidence": item["confidence"],
                    "gpt54_model": model,
                    "gpt54_reasoning_effort": reasoning_effort,
                    "gpt54_prompt_version": prompt_version,
                    "gpt54_batch_id": batch_id,
                }
    return result_map


def attach_scores(row: Dict[str, str]) -> None:
    label = row.get("gpt54_exposure", "")
    if label not in LABEL_TO_SCORES:
        row["gpt54_alpha"] = ""
        row["gpt54_beta"] = ""
        row["gpt54_gamma"] = ""
        return
    scores = LABEL_TO_SCORES[label]
    row["gpt54_alpha"] = scores["alpha"]
    row["gpt54_beta"] = scores["beta"]
    row["gpt54_gamma"] = scores["gamma"]


def float_string(value: float) -> str:
    return f"{value:.6f}".rstrip("0").rstrip(".") if "." in f"{value:.6f}" else f"{value:.6f}"


def label_scores(label: str) -> Dict[str, float]:
    if label not in LABEL_TO_SCORES:
        raise ValueError(f"Unsupported label {label!r}")
    raw = LABEL_TO_SCORES[label]
    return {key: float(raw[key]) for key in ("alpha", "beta", "gamma")}


def merge_results(
    base_labelset_path: Path,
    input_path: Path,
    work_dir: Path,
    results_dirs: Sequence[Path],
    output_path: Path,
    allow_missing: bool,
) -> None:
    if base_labelset_path.exists():
        base_rows = read_tsv_rows(base_labelset_path)
    else:
        base_rows = read_tsv_rows(input_path)

    candidate_results_dirs = list(results_dirs)
    if not candidate_results_dirs:
        candidate_results_dirs = [work_dir / "results"]
    result_map = load_results(candidate_results_dirs)
    if not result_map:
        raise FileNotFoundError(
            f"No normalized results found in {candidate_results_dirs}. "
            "Run 'classify-batch' or 'classify-all' first."
        )

    merged_rows: List[Dict[str, str]] = []
    for row in base_rows:
        row_id = row[NORMALIZED_ID_FIELD]
        merged = dict(row)
        result = result_map.get(row_id)
        if result is None:
            if not allow_missing:
                raise ValueError(
                    f"Missing GPT-5.4 result for row_id {row_id}. "
                    "Use --allow-missing to write blanks instead."
                )
            for key in EXTRA_RESULT_COLUMNS:
                merged[key] = ""
        else:
            merged.update(result)
            attach_scores(merged)
        merged_rows.append(merged)

    output_fieldnames = to_tsv_fieldnames(merged_rows)
    for column in EXTRA_RESULT_COLUMNS:
        if column not in output_fieldnames:
            output_fieldnames.append(column)

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_fieldnames, delimiter="\t")
        writer.writeheader()
        for row in merged_rows:
            writer.writerow({key: row.get(key, "") for key in output_fieldnames})

    label_counts = Counter(
        row["gpt54_exposure"] for row in merged_rows if row.get("gpt54_exposure")
    )
    confidence_counts = Counter(
        row["gpt54_confidence"] for row in merged_rows if row.get("gpt54_confidence")
    )
    print(f"Wrote merged TSV to {output_path}")
    if label_counts:
        print("Label counts:", dict(label_counts))
    if confidence_counts:
        print("Confidence counts:", dict(confidence_counts))


def task_weight(task_type: str, weighting: str) -> float:
    if weighting == "equal":
        return 1.0
    if weighting == "core":
        return 2.0 if task_type == "Core" else 1.0
    raise ValueError(f"Unsupported weighting {weighting!r}")


def compare_results(
    merged_path: Path,
    baseline_label_column: str,
    task_delta_path: Path,
    occ_delta_path: Path,
    summary_path: Path,
    weighting: str,
    allow_missing: bool,
) -> None:
    rows = read_tsv_rows(merged_path)
    if not rows:
        raise ValueError(f"No rows found in {merged_path}")
    required_columns = [
        baseline_label_column,
        "gpt54_exposure",
        "Task Type",
        "Title",
        "O*NET-SOC Code",
        "Task ID",
        "Task",
    ]
    missing_columns = [column for column in required_columns if column not in rows[0]]
    if missing_columns:
        raise ValueError(
            f"Merged file is missing required comparison columns: {missing_columns}"
        )

    task_deltas: List[Dict[str, str]] = []
    transition_counts: Counter[str] = Counter()
    confidence_counts: Counter[str] = Counter()
    occupation_buckets: Dict[tuple[str, str], Dict[str, float]] = defaultdict(
        lambda: {
            "weight_total": 0.0,
            "legacy_alpha_sum": 0.0,
            "legacy_beta_sum": 0.0,
            "legacy_gamma_sum": 0.0,
            "gpt54_alpha_sum": 0.0,
            "gpt54_beta_sum": 0.0,
            "gpt54_gamma_sum": 0.0,
            "task_count": 0.0,
            "changed_task_count": 0.0,
        }
    )

    missing_row_count = 0
    for row in rows:
        legacy_label = row.get(baseline_label_column, "")
        new_label = row.get("gpt54_exposure", "")
        if not new_label:
            missing_row_count += 1
            if allow_missing:
                continue
            raise ValueError(
                f"Missing gpt54_exposure for row_id {row[NORMALIZED_ID_FIELD]}. "
                "Use --allow-missing to skip unfinished rows."
            )
        legacy_scores = label_scores(legacy_label)
        new_scores = label_scores(new_label)
        changed = legacy_label != new_label
        transition = f"{legacy_label}->{new_label}"
        transition_counts[transition] += 1
        confidence = row.get("gpt54_confidence", "")
        if confidence:
            confidence_counts[confidence] += 1

        task_delta = {
            NORMALIZED_ID_FIELD: row[NORMALIZED_ID_FIELD],
            "O*NET-SOC Code": row["O*NET-SOC Code"],
            "Task ID": row["Task ID"],
            "Title": row["Title"],
            "Task Type": row["Task Type"],
            "Task": row["Task"],
            "legacy_label_column": baseline_label_column,
            "legacy_exposure": legacy_label,
            "gpt54_exposure": new_label,
            "label_transition": transition,
            "changed_exposure": "1" if changed else "0",
            "legacy_alpha": float_string(legacy_scores["alpha"]),
            "legacy_beta": float_string(legacy_scores["beta"]),
            "legacy_gamma": float_string(legacy_scores["gamma"]),
            "gpt54_alpha": float_string(new_scores["alpha"]),
            "gpt54_beta": float_string(new_scores["beta"]),
            "gpt54_gamma": float_string(new_scores["gamma"]),
            "delta_alpha": float_string(new_scores["alpha"] - legacy_scores["alpha"]),
            "delta_beta": float_string(new_scores["beta"] - legacy_scores["beta"]),
            "delta_gamma": float_string(new_scores["gamma"] - legacy_scores["gamma"]),
            "gpt54_confidence": confidence,
            "gpt54_reason": row.get("gpt54_reason", ""),
            "gpt54_model": row.get("gpt54_model", ""),
            "gpt54_reasoning_effort": row.get("gpt54_reasoning_effort", ""),
            "gpt54_prompt_version": row.get("gpt54_prompt_version", ""),
        }
        task_deltas.append(task_delta)

        bucket_key = (row["O*NET-SOC Code"], row["Title"])
        bucket = occupation_buckets[bucket_key]
        weight = task_weight(row["Task Type"], weighting)
        bucket["weight_total"] += weight
        bucket["legacy_alpha_sum"] += legacy_scores["alpha"] * weight
        bucket["legacy_beta_sum"] += legacy_scores["beta"] * weight
        bucket["legacy_gamma_sum"] += legacy_scores["gamma"] * weight
        bucket["gpt54_alpha_sum"] += new_scores["alpha"] * weight
        bucket["gpt54_beta_sum"] += new_scores["beta"] * weight
        bucket["gpt54_gamma_sum"] += new_scores["gamma"] * weight
        bucket["task_count"] += 1.0
        if changed:
            bucket["changed_task_count"] += 1.0

    task_fieldnames = [
        NORMALIZED_ID_FIELD,
        "O*NET-SOC Code",
        "Task ID",
        "Title",
        "Task Type",
        "Task",
        "legacy_label_column",
        "legacy_exposure",
        "gpt54_exposure",
        "label_transition",
        "changed_exposure",
        "legacy_alpha",
        "legacy_beta",
        "legacy_gamma",
        "gpt54_alpha",
        "gpt54_beta",
        "gpt54_gamma",
        "delta_alpha",
        "delta_beta",
        "delta_gamma",
        "gpt54_confidence",
        "gpt54_reason",
        "gpt54_model",
        "gpt54_reasoning_effort",
        "gpt54_prompt_version",
    ]
    with task_delta_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=task_fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(task_deltas)

    occupation_rows: List[Dict[str, str]] = []
    for (onet_soc_code, title), bucket in sorted(occupation_buckets.items()):
        weight_total = bucket["weight_total"]
        legacy_alpha = bucket["legacy_alpha_sum"] / weight_total
        legacy_beta = bucket["legacy_beta_sum"] / weight_total
        legacy_gamma = bucket["legacy_gamma_sum"] / weight_total
        new_alpha = bucket["gpt54_alpha_sum"] / weight_total
        new_beta = bucket["gpt54_beta_sum"] / weight_total
        new_gamma = bucket["gpt54_gamma_sum"] / weight_total
        occupation_rows.append(
            {
                "O*NET-SOC Code": onet_soc_code,
                "Title": title,
                "weighting": weighting,
                "task_count": str(int(bucket["task_count"])),
                "changed_task_count": str(int(bucket["changed_task_count"])),
                "legacy_alpha": float_string(legacy_alpha),
                "legacy_beta": float_string(legacy_beta),
                "legacy_gamma": float_string(legacy_gamma),
                "gpt54_alpha": float_string(new_alpha),
                "gpt54_beta": float_string(new_beta),
                "gpt54_gamma": float_string(new_gamma),
                "delta_alpha": float_string(new_alpha - legacy_alpha),
                "delta_beta": float_string(new_beta - legacy_beta),
                "delta_gamma": float_string(new_gamma - legacy_gamma),
            }
        )
    occupation_rows.sort(
        key=lambda row: abs(float(row["delta_beta"])),
        reverse=True,
    )
    occ_fieldnames = [
        "O*NET-SOC Code",
        "Title",
        "weighting",
        "task_count",
        "changed_task_count",
        "legacy_alpha",
        "legacy_beta",
        "legacy_gamma",
        "gpt54_alpha",
        "gpt54_beta",
        "gpt54_gamma",
        "delta_alpha",
        "delta_beta",
        "delta_gamma",
    ]
    with occ_delta_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=occ_fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(occupation_rows)

    changed_row_count = sum(1 for row in task_deltas if row["changed_exposure"] == "1")
    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "merged_path": str(merged_path),
        "baseline_label_column": baseline_label_column,
        "weighting": weighting,
        "task_rows_compared": len(task_deltas),
        "task_rows_changed": changed_row_count,
        "task_rows_unchanged": len(task_deltas) - changed_row_count,
        "task_rows_missing_gpt54": missing_row_count,
        "label_transitions": dict(transition_counts),
        "gpt54_confidence_counts": dict(confidence_counts),
        "top_occupation_beta_deltas": occupation_rows[:20],
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote task delta TSV to {task_delta_path}")
    print(f"Wrote occupation delta TSV to {occ_delta_path}")
    print(f"Wrote comparison summary JSON to {summary_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="GPT-5.4 exposure rerun pipeline using codex exec.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    make_batches_parser = subparsers.add_parser(
        "make-batches",
        help="Create Codex batch files from full_onet_data.tsv.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    make_batches_parser.add_argument(
        "--input-path",
        type=Path,
        default=DEFAULT_INPUT_PATH,
    )
    make_batches_parser.add_argument(
        "--work-dir",
        type=Path,
        default=DEFAULT_WORK_DIR,
    )
    make_batches_parser.add_argument(
        "--shard-size",
        type=int,
        default=DEFAULT_SHARD_SIZE,
    )
    make_batches_parser.add_argument(
        "--start-index",
        type=int,
        default=0,
    )
    make_batches_parser.add_argument(
        "--limit",
        type=int,
        default=None,
    )
    make_batches_parser.add_argument(
        "--row-id-file",
        type=Path,
        default=None,
        help="Optional newline-delimited list of row_id values to include.",
    )
    make_batches_parser.add_argument(
        "--prompt-version",
        default=DEFAULT_PROMPT_VERSION,
        help="Prompt version label written into the batch manifest.",
    )

    classify_batch_parser = subparsers.add_parser(
        "classify-batch",
        help="Run codex exec for one batch file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    classify_batch_parser.add_argument("batch_path", type=Path)
    classify_batch_parser.add_argument(
        "--prompt-template-path",
        type=Path,
        default=DEFAULT_PROMPT_PATH,
    )
    classify_batch_parser.add_argument(
        "--schema-path",
        type=Path,
        default=DEFAULT_SCHEMA_PATH,
    )
    classify_batch_parser.add_argument(
        "--work-dir",
        type=Path,
        default=DEFAULT_WORK_DIR,
    )
    classify_batch_parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
    )
    classify_batch_parser.add_argument(
        "--reasoning-effort",
        default=DEFAULT_REASONING_EFFORT,
    )
    classify_batch_parser.add_argument(
        "--prompt-version",
        default=DEFAULT_PROMPT_VERSION,
    )
    classify_batch_parser.add_argument(
        "--force",
        action="store_true",
    )

    classify_all_parser = subparsers.add_parser(
        "classify-all",
        help="Run codex exec across every pending batch file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    classify_all_parser.add_argument(
        "--work-dir",
        type=Path,
        default=DEFAULT_WORK_DIR,
    )
    classify_all_parser.add_argument(
        "--prompt-template-path",
        type=Path,
        default=DEFAULT_PROMPT_PATH,
    )
    classify_all_parser.add_argument(
        "--schema-path",
        type=Path,
        default=DEFAULT_SCHEMA_PATH,
    )
    classify_all_parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
    )
    classify_all_parser.add_argument(
        "--reasoning-effort",
        default=DEFAULT_REASONING_EFFORT,
    )
    classify_all_parser.add_argument(
        "--prompt-version",
        default=DEFAULT_PROMPT_VERSION,
    )
    classify_all_parser.add_argument(
        "--force",
        action="store_true",
    )
    classify_all_parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Optional cap for smoke tests or staged runs.",
    )

    merge_parser = subparsers.add_parser(
        "merge",
        help="Merge normalized batch outputs into a full_labelset-style TSV.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    merge_parser.add_argument(
        "--base-labelset-path",
        type=Path,
        default=DEFAULT_BASE_LABELSET,
    )
    merge_parser.add_argument(
        "--input-path",
        type=Path,
        default=DEFAULT_INPUT_PATH,
    )
    merge_parser.add_argument(
        "--work-dir",
        type=Path,
        default=DEFAULT_WORK_DIR,
    )
    merge_parser.add_argument(
        "--results-dir",
        type=Path,
        action="append",
        default=[],
        help="Optional results directory. Repeat to layer reruns; later paths override earlier ones by row_id.",
    )
    merge_parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
    )
    merge_parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Write blank GPT-5.4 columns for rows that do not have results yet.",
    )

    compare_parser = subparsers.add_parser(
        "compare",
        help="Create task-level and occupation-level deltas versus the legacy GPT-4 labels.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    compare_parser.add_argument(
        "--merged-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
    )
    compare_parser.add_argument(
        "--baseline-label-column",
        default="gpt4_exposure",
        help="Legacy label column to compare against.",
    )
    compare_parser.add_argument(
        "--task-delta-path",
        type=Path,
        default=DEFAULT_TASK_DELTA_PATH,
    )
    compare_parser.add_argument(
        "--occ-delta-path",
        type=Path,
        default=DEFAULT_OCC_DELTA_PATH,
    )
    compare_parser.add_argument(
        "--summary-path",
        type=Path,
        default=DEFAULT_SUMMARY_PATH,
    )
    compare_parser.add_argument(
        "--weighting",
        choices=("equal", "core"),
        default="core",
        help="Weighting to use for occupation-level deltas.",
    )
    compare_parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Skip rows that do not have GPT-5.4 labels yet.",
    )

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "make-batches":
            row_id_filter = load_row_ids(args.row_id_file)
            create_batches(
                input_path=args.input_path,
                work_dir=args.work_dir,
                shard_size=args.shard_size,
                start_index=args.start_index,
                limit=args.limit,
                row_id_filter=row_id_filter,
                prompt_version=args.prompt_version,
            )
        elif args.command == "classify-batch":
            classify_batch(
                batch_path=args.batch_path,
                prompt_template_path=args.prompt_template_path,
                schema_path=args.schema_path,
                work_dir=args.work_dir,
                model=args.model,
                reasoning_effort=args.reasoning_effort,
                prompt_version=args.prompt_version,
                force=args.force,
            )
        elif args.command == "classify-all":
            classify_all(
                work_dir=args.work_dir,
                prompt_template_path=args.prompt_template_path,
                schema_path=args.schema_path,
                model=args.model,
                reasoning_effort=args.reasoning_effort,
                prompt_version=args.prompt_version,
                force=args.force,
                max_batches=args.max_batches,
            )
        elif args.command == "merge":
            merge_results(
                base_labelset_path=args.base_labelset_path,
                input_path=args.input_path,
                work_dir=args.work_dir,
                results_dirs=args.results_dir,
                output_path=args.output_path,
                allow_missing=args.allow_missing,
            )
        elif args.command == "compare":
            compare_results(
                merged_path=args.merged_path,
                baseline_label_column=args.baseline_label_column,
                task_delta_path=args.task_delta_path,
                occ_delta_path=args.occ_delta_path,
                summary_path=args.summary_path,
                weighting=args.weighting,
                allow_missing=args.allow_missing,
            )
        else:
            parser.error(f"Unhandled command {args.command!r}")
    except Exception as exc:  # pragma: no cover - CLI surface
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
