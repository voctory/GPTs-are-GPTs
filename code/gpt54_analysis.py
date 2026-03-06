#!/usr/bin/env python3
"""Executable paper-style analysis for the GPTs-are-GPTs rerun.

This replaces the first notebook (`gpts_are_gpts_script1.ipynb`) with a
standard-library CLI that can run in this environment without Jupyter or
third-party Python packages.

Scope:
- task-level label agreement summaries
- occupation-level aggregation
- exposure-share tables and threshold curves
- GPT-5.4 vs GPT-4 delta exports

Non-goals:
- plotting cells
- the second notebook's network/community analysis
  (`gpts_are_gpts_script2.ipynb`), whose required input files are not present
  in this repository
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

try:
    from gpt54_rerun import (
        DEFAULT_OCC_DELTA_PATH,
        DEFAULT_OUTPUT_PATH as DEFAULT_MERGED_PATH,
        DEFAULT_SUMMARY_PATH as DEFAULT_COMPARE_SUMMARY_PATH,
        DEFAULT_TASK_DELTA_PATH,
        compare_results,
    )
except ImportError:  # pragma: no cover - CLI import guard
    DEFAULT_MERGED_PATH = Path(__file__).resolve().parent.parent / "data" / "full_labelset_gpt54.tsv"
    DEFAULT_TASK_DELTA_PATH = Path(__file__).resolve().parent.parent / "data" / "gpt54_vs_gpt4_task_delta.tsv"
    DEFAULT_OCC_DELTA_PATH = Path(__file__).resolve().parent.parent / "data" / "gpt54_vs_gpt4_occ_delta.tsv"
    DEFAULT_COMPARE_SUMMARY_PATH = Path(__file__).resolve().parent.parent / "data" / "gpt54_vs_gpt4_summary.json"
    compare_results = None


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BLS_PATH = REPO_ROOT / "data" / "national_May2021_dl.csv"
DEFAULT_OCC_LEVEL_OUTPUT = REPO_ROOT / "data" / "occ_level_gpt54.csv"
DEFAULT_WEIGHTED_OCC_OUTPUT = REPO_ROOT / "data" / "occ_level_gpt54_weighted.tsv"
DEFAULT_OCC_SHARE_OUTPUT = REPO_ROOT / "data" / "occupation_task_shares_gpt54.tsv"
DEFAULT_CURVES_OUTPUT = REPO_ROOT / "data" / "gpt54_exposure_curves.tsv"
DEFAULT_SUMMARY_OUTPUT = REPO_ROOT / "data" / "gpt54_analysis_summary.json"

NORMALIZED_ID_FIELD = "row_id"
LABELS = ("E0", "E1", "E2")
RELEVANCE_TYPES = ("a", "b", "c")
WEIGHTINGS = ("core", "equal")
THRESHOLDS = tuple(range(0, 100, 5))
PRIMARY_CURRENT_COLUMN = "gpt54_exposure"

LABEL_TO_SCORES = {
    "E0": {"alpha": 0.0, "beta": 0.0, "gamma": 0.0},
    "E1": {"alpha": 1.0, "beta": 1.0, "gamma": 1.0},
    "E2": {"alpha": 0.0, "beta": 0.5, "gamma": 1.0},
    "E3": {"alpha": 0.0, "beta": 0.5, "gamma": 1.0},
}

SOURCES = (
    ("human", "human_exposure"),
    ("gpt4", "gpt4_exposure"),
    ("gpt54", PRIMARY_CURRENT_COLUMN),
)


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def parse_number(value: str) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text in {"*", "**", "#", "nan", "NaN", "None"}:
        return None
    text = text.replace(",", "")
    try:
        return float(text)
    except ValueError:
        return None


def format_number(value: float | None, digits: int = 6) -> str:
    if value is None or math.isnan(value):
        return ""
    text = f"{value:.{digits}f}"
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text or "0"


def safe_log(value: float | None) -> float | None:
    if value is None or value <= 0:
        return None
    return math.log(value)


def read_delimited_rows(path: Path, delimiter: str) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
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


def write_delimited_rows(
    path: Path,
    fieldnames: Sequence[str],
    rows: Sequence[Dict[str, str]],
    delimiter: str,
) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter=delimiter)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def label_scores(label: str) -> Dict[str, float]:
    if label not in LABEL_TO_SCORES:
        raise ValueError(f"Unsupported label {label!r}")
    return LABEL_TO_SCORES[label]


def cohen_kappa(labels_a: Sequence[str], labels_b: Sequence[str]) -> float | None:
    if len(labels_a) != len(labels_b):
        raise ValueError("Label sequences must have equal length for kappa.")
    if not labels_a:
        return None
    n = len(labels_a)
    observed = sum(1 for a, b in zip(labels_a, labels_b) if a == b) / n
    counts_a = Counter(labels_a)
    counts_b = Counter(labels_b)
    expected = 0.0
    for label in LABELS:
        expected += (counts_a[label] / n) * (counts_b[label] / n)
    if math.isclose(expected, 1.0):
        return 1.0 if math.isclose(observed, 1.0) else None
    return (observed - expected) / (1.0 - expected)


def confusion_matrix(
    labels_a: Sequence[str],
    labels_b: Sequence[str],
) -> Dict[str, Dict[str, int]]:
    matrix: Dict[str, Dict[str, int]] = {
        label_a: {label_b: 0 for label_b in LABELS} for label_a in LABELS
    }
    for label_a, label_b in zip(labels_a, labels_b):
        matrix[label_a][label_b] += 1
    return matrix


def normalize_rows(
    rows: Sequence[Dict[str, str]],
    current_label_column: str,
    bootstrap_current_from_column: str | None,
) -> List[Dict[str, str]]:
    normalized_rows: List[Dict[str, str]] = []
    for row in rows:
        normalized = dict(row)
        if not normalized.get("human_exposure"):
            normalized["human_exposure"] = normalized.get("human_exposure_agg", "")
        if not normalized.get("gpt4_alt_exposure"):
            normalized["gpt4_alt_exposure"] = normalized.get(
                "gpt4_exposure_alt_rubric", ""
            )

        current_label = normalized.get(current_label_column, "")
        if not current_label and bootstrap_current_from_column:
            current_label = normalized.get(bootstrap_current_from_column, "")

        normalized[PRIMARY_CURRENT_COLUMN] = current_label
        if PRIMARY_CURRENT_COLUMN == current_label_column:
            normalized.setdefault("gpt54_reason", normalized.get("gpt54_reason", ""))
            normalized.setdefault(
                "gpt54_confidence", normalized.get("gpt54_confidence", "")
            )
        elif current_label:
            normalized["gpt54_reason"] = ""
            normalized["gpt54_confidence"] = ""
            normalized["gpt54_model"] = (
                f"bootstrap:{bootstrap_current_from_column}"
                if bootstrap_current_from_column
                else f"alias:{current_label_column}"
            )
        normalized_rows.append(normalized)
    return normalized_rows


def bls_stats_by_occ_code(path: Path) -> Dict[str, Dict[str, float | str | None]]:
    rows = read_delimited_rows(path, ",")
    preferred_order = {"detailed": 0, "broad": 1, "minor": 2, "major": 3, "total": 4}
    selected: Dict[str, Dict[str, str]] = {}
    selected_rank: Dict[str, int] = {}

    for row in rows:
        occ_code = row.get("OCC_CODE", "").strip()
        if not occ_code:
            continue
        rank = preferred_order.get(row.get("O_GROUP", "").strip().lower(), 99)
        if occ_code not in selected or rank < selected_rank[occ_code]:
            selected[occ_code] = row
            selected_rank[occ_code] = rank

    mapping: Dict[str, Dict[str, float | str | None]] = {}
    for occ_code, row in selected.items():
        tot_emp = parse_number(row.get("TOT_EMP", ""))
        a_mean = parse_number(row.get("A_MEAN", ""))
        a_median = parse_number(row.get("A_MEDIAN", ""))
        h_mean = parse_number(row.get("H_MEAN", ""))
        h_median = parse_number(row.get("H_MEDIAN", ""))
        mapping[occ_code] = {
            "OCC_CODE": occ_code,
            "OCC_TITLE": row.get("OCC_TITLE", ""),
            "O_GROUP": row.get("O_GROUP", ""),
            "TOT_EMP": tot_emp,
            "A_MEAN": a_mean,
            "A_MEDIAN": a_median,
            "H_MEAN": h_mean,
            "H_MEDIAN": h_median,
            "log_A_mean": safe_log(a_mean),
            "log_totemp": safe_log(tot_emp),
        }
    return mapping


def task_weight(task_type: str, weighting: str) -> float:
    if weighting == "equal":
        return 1.0
    if weighting == "core":
        return 2.0 if task_type == "Core" else 1.0
    raise ValueError(f"Unsupported weighting {weighting!r}")


def aggregate_scores(
    rows: Sequence[Dict[str, str]],
    bls_stats: Dict[str, Dict[str, float | str | None]],
    weighting: str,
    allow_missing: bool,
) -> List[Dict[str, str]]:
    accumulators: Dict[Tuple[str, str], Dict[str, float]] = defaultdict(
        lambda: {
            "weight_total": 0.0,
            "task_count": 0.0,
            "human_alpha_sum": 0.0,
            "human_beta_sum": 0.0,
            "human_gamma_sum": 0.0,
            "gpt4_alpha_sum": 0.0,
            "gpt4_beta_sum": 0.0,
            "gpt4_gamma_sum": 0.0,
            "gpt54_alpha_sum": 0.0,
            "gpt54_beta_sum": 0.0,
            "gpt54_gamma_sum": 0.0,
        }
    )

    for row in rows:
        current_label = row.get(PRIMARY_CURRENT_COLUMN, "")
        if not current_label:
            if allow_missing:
                continue
            raise ValueError(
                f"Missing {PRIMARY_CURRENT_COLUMN} for row_id {row[NORMALIZED_ID_FIELD]}. "
                "Use --allow-missing to skip unfinished rows."
            )

        key = (row["O*NET-SOC Code"], row["Title"])
        bucket = accumulators[key]
        weight = task_weight(row["Task Type"], weighting)
        bucket["weight_total"] += weight
        bucket["task_count"] += 1.0

        for prefix, column in SOURCES:
            scores = label_scores(row[column])
            bucket[f"{prefix}_alpha_sum"] += scores["alpha"] * weight
            bucket[f"{prefix}_beta_sum"] += scores["beta"] * weight
            bucket[f"{prefix}_gamma_sum"] += scores["gamma"] * weight

    output_rows: List[Dict[str, str]] = []
    for (onet_soc_code, title), bucket in sorted(accumulators.items()):
        occ_code = onet_soc_code[:7]
        bls_row = bls_stats.get(occ_code, {})
        weight_total = bucket["weight_total"] or 1.0

        row = {
            "O*NET-SOC Code": onet_soc_code,
            "Title": title,
            "OCC_CODE": occ_code,
            "weighting": weighting,
            "task_count": str(int(bucket["task_count"])),
            "weight_total": format_number(bucket["weight_total"]),
            "human_rating_alpha": format_number(bucket["human_alpha_sum"] / weight_total),
            "human_rating_beta": format_number(bucket["human_beta_sum"] / weight_total),
            "human_rating_gamma": format_number(bucket["human_gamma_sum"] / weight_total),
            "gpt4_rating_alpha": format_number(bucket["gpt4_alpha_sum"] / weight_total),
            "gpt4_rating_beta": format_number(bucket["gpt4_beta_sum"] / weight_total),
            "gpt4_rating_gamma": format_number(bucket["gpt4_gamma_sum"] / weight_total),
            "gpt54_rating_alpha": format_number(bucket["gpt54_alpha_sum"] / weight_total),
            "gpt54_rating_beta": format_number(bucket["gpt54_beta_sum"] / weight_total),
            "gpt54_rating_gamma": format_number(bucket["gpt54_gamma_sum"] / weight_total),
            "TOT_EMP": format_number(bls_row.get("TOT_EMP")),
            "A_MEAN": format_number(bls_row.get("A_MEAN")),
            "A_MEDIAN": format_number(bls_row.get("A_MEDIAN")),
            "H_MEAN": format_number(bls_row.get("H_MEAN")),
            "H_MEDIAN": format_number(bls_row.get("H_MEDIAN")),
            "log_A_mean": format_number(bls_row.get("log_A_mean")),
            "log_totemp": format_number(bls_row.get("log_totemp")),
        }
        for suffix in ("alpha", "beta", "gamma"):
            row[f"delta_gpt54_vs_gpt4_{suffix}"] = format_number(
                float(row[f"gpt54_rating_{suffix}"]) - float(row[f"gpt4_rating_{suffix}"])
            )
        row["gpt54_human_diff"] = format_number(
            float(row["gpt54_rating_beta"]) - float(row["human_rating_beta"])
        )
        row["gpt4_human_diff"] = format_number(
            float(row["gpt4_rating_beta"]) - float(row["human_rating_beta"])
        )
        output_rows.append(row)
    return output_rows


def build_occ_share_table(
    rows: Sequence[Dict[str, str]],
    bls_stats: Dict[str, Dict[str, float | str | None]],
    allow_missing: bool,
) -> List[Dict[str, str]]:
    grouped: Dict[Tuple[str, str], Dict[str, object]] = {}
    for row in rows:
        current_label = row.get(PRIMARY_CURRENT_COLUMN, "")
        if not current_label:
            if allow_missing:
                continue
            raise ValueError(
                f"Missing {PRIMARY_CURRENT_COLUMN} for row_id {row[NORMALIZED_ID_FIELD]}. "
                "Use --allow-missing to skip unfinished rows."
            )
        key = (row["O*NET-SOC Code"], row["Title"])
        bucket = grouped.setdefault(
            key,
            {
                "counts": {source: Counter() for source, _column in SOURCES},
                "task_count": 0,
            },
        )
        bucket["task_count"] += 1
        for source, column in SOURCES:
            bucket["counts"][source][row[column]] += 1

    share_rows: List[Dict[str, str]] = []
    for (onet_soc_code, title), bucket in sorted(grouped.items()):
        occ_code = onet_soc_code[:7]
        bls_row = bls_stats.get(occ_code, {})
        task_count = bucket["task_count"] or 1
        output = {
            "O*NET-SOC Code": onet_soc_code,
            "Title": title,
            "OCC_CODE": occ_code,
            "task_count": str(task_count),
            "TOT_EMP": format_number(bls_row.get("TOT_EMP")),
            "A_MEAN": format_number(bls_row.get("A_MEAN")),
            "A_MEDIAN": format_number(bls_row.get("A_MEDIAN")),
            "log_A_mean": format_number(bls_row.get("log_A_mean")),
            "log_totemp": format_number(bls_row.get("log_totemp")),
        }
        for source, _column in SOURCES:
            counts: Counter[str] = bucket["counts"][source]
            e0 = counts["E0"] / task_count * 100.0
            e1 = counts["E1"] / task_count * 100.0
            e2 = counts["E2"] / task_count * 100.0
            output[f"{source}_E0_percent"] = format_number(e0)
            output[f"{source}_E1_percent"] = format_number(e1)
            output[f"{source}_E2_percent"] = format_number(e2)
            output[f"{source}_relevance_a"] = format_number(e1)
            output[f"{source}_relevance_b"] = format_number(e1 + 0.5 * e2)
            output[f"{source}_relevance_c"] = format_number(e1 + e2)
        for suffix in RELEVANCE_TYPES:
            output[f"delta_gpt54_vs_gpt4_relevance_{suffix}"] = format_number(
                float(output[f"gpt54_relevance_{suffix}"])
                - float(output[f"gpt4_relevance_{suffix}"])
            )
        share_rows.append(output)
    return share_rows


def exposure_curves(share_rows: Sequence[Dict[str, str]]) -> List[Dict[str, str]]:
    output_rows: List[Dict[str, str]] = []
    total_occupations = len(share_rows) or 1
    employment_total = sum(parse_number(row["TOT_EMP"]) or 0.0 for row in share_rows) or 1.0

    for source, _column in SOURCES:
        for relevance_type in RELEVANCE_TYPES:
            relevance_key = f"{source}_relevance_{relevance_type}"
            for threshold in THRESHOLDS:
                occupation_count = 0
                employee_count = 0.0
                for row in share_rows:
                    relevance = parse_number(row[relevance_key]) or 0.0
                    if relevance > threshold:
                        occupation_count += 1
                        employee_count += parse_number(row["TOT_EMP"]) or 0.0
                output_rows.append(
                    {
                        "source": source,
                        "population": "occupation",
                        "relevance_type": relevance_type,
                        "threshold": str(threshold),
                        "count": str(occupation_count),
                        "percent": format_number(occupation_count / total_occupations * 100.0),
                    }
                )
                output_rows.append(
                    {
                        "source": source,
                        "population": "employee",
                        "relevance_type": relevance_type,
                        "threshold": str(threshold),
                        "count": format_number(employee_count),
                        "percent": format_number(employee_count / employment_total * 100.0),
                    }
                )
    return output_rows


def summary_payload(
    rows: Sequence[Dict[str, str]],
    primary_occ_rows: Sequence[Dict[str, str]],
    share_rows: Sequence[Dict[str, str]],
    curves_rows: Sequence[Dict[str, str]],
    primary_weighting: str,
    current_label_column: str,
    bootstrap_current_from_column: str | None,
) -> Dict[str, object]:
    normalized_rows = [row for row in rows if row.get(PRIMARY_CURRENT_COLUMN)]
    human_labels = [row["human_exposure"] for row in normalized_rows]
    gpt4_labels = [row["gpt4_exposure"] for row in normalized_rows]
    gpt54_labels = [row[PRIMARY_CURRENT_COLUMN] for row in normalized_rows]

    def task_distribution(column: str) -> Dict[str, int]:
        counter = Counter(row[column] for row in normalized_rows)
        return {label: counter.get(label, 0) for label in LABELS}

    occupation_summary: Dict[str, Dict[str, float | None]] = {}
    for source, _column in SOURCES:
        stats: Dict[str, float | None] = {}
        for relevance_type in RELEVANCE_TYPES:
            values = [
                parse_number(row[f"{source}_relevance_{relevance_type}"])
                for row in share_rows
            ]
            numeric_values = [value for value in values if value is not None]
            if numeric_values:
                mean_value = sum(numeric_values) / len(numeric_values)
                variance = (
                    sum((value - mean_value) ** 2 for value in numeric_values)
                    / (len(numeric_values) - 1)
                    if len(numeric_values) > 1
                    else 0.0
                )
                stats[f"relevance_{relevance_type}_mean"] = mean_value
                stats[f"relevance_{relevance_type}_std"] = math.sqrt(variance)
            else:
                stats[f"relevance_{relevance_type}_mean"] = None
                stats[f"relevance_{relevance_type}_std"] = None
        occupation_summary[source] = stats

    curve_lookup: Dict[Tuple[str, str, str, str], Dict[str, str]] = {}
    for row in curves_rows:
        key = (row["source"], row["population"], row["relevance_type"], row["threshold"])
        curve_lookup[key] = row

    threshold_snapshots: Dict[str, Dict[str, Dict[str, Dict[str, str]]]] = {}
    for threshold in ("10", "50"):
        threshold_snapshots[threshold] = {}
        for source, _column in SOURCES:
            threshold_snapshots[threshold][source] = {}
            for population in ("occupation", "employee"):
                threshold_snapshots[threshold][source][population] = {
                    relevance_type: curve_lookup[
                        (source, population, relevance_type, threshold)
                    ]["percent"]
                    for relevance_type in RELEVANCE_TYPES
                }

    ordered_deltas = sorted(
        primary_occ_rows,
        key=lambda row: abs(parse_number(row["delta_gpt54_vs_gpt4_beta"]) or 0.0),
        reverse=True,
    )

    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "primary_weighting": primary_weighting,
        "current_label_column": current_label_column,
        "bootstrap_current_from_column": bootstrap_current_from_column,
        "task_row_count_with_current_labels": len(normalized_rows),
        "occupation_row_count_primary": len(primary_occ_rows),
        "task_label_distributions": {
            "human": task_distribution("human_exposure"),
            "gpt4": task_distribution("gpt4_exposure"),
            "gpt54": task_distribution(PRIMARY_CURRENT_COLUMN),
        },
        "cohen_kappa": {
            "human_vs_gpt4": cohen_kappa(human_labels, gpt4_labels),
            "human_vs_gpt54": cohen_kappa(human_labels, gpt54_labels),
            "gpt4_vs_gpt54": cohen_kappa(gpt4_labels, gpt54_labels),
        },
        "confusion_matrices": {
            "human_vs_gpt4": confusion_matrix(human_labels, gpt4_labels),
            "human_vs_gpt54": confusion_matrix(human_labels, gpt54_labels),
            "gpt4_vs_gpt54": confusion_matrix(gpt4_labels, gpt54_labels),
        },
        "occupation_relevance_summary": occupation_summary,
        "threshold_snapshots": threshold_snapshots,
        "top_beta_deltas_primary_weighting": ordered_deltas[:20],
    }


def write_occ_level_csv(path: Path, rows: Sequence[Dict[str, str]]) -> None:
    fieldnames = [
        "O*NET-SOC Code",
        "Title",
        "OCC_CODE",
        "weighting",
        "gpt54_rating_alpha",
        "gpt54_rating_beta",
        "gpt54_rating_gamma",
        "gpt4_rating_alpha",
        "gpt4_rating_beta",
        "gpt4_rating_gamma",
        "human_rating_alpha",
        "human_rating_beta",
        "human_rating_gamma",
        "delta_gpt54_vs_gpt4_alpha",
        "delta_gpt54_vs_gpt4_beta",
        "delta_gpt54_vs_gpt4_gamma",
        "gpt54_human_diff",
        "gpt4_human_diff",
        "TOT_EMP",
        "A_MEAN",
        "A_MEDIAN",
        "H_MEAN",
        "H_MEDIAN",
        "log_A_mean",
        "log_totemp",
    ]
    write_delimited_rows(path, fieldnames, rows, ",")


def maybe_compare(
    merged_path: Path,
    task_delta_path: Path,
    occ_delta_path: Path,
    compare_summary_path: Path,
    allow_missing: bool,
    skip_compare: bool,
    current_label_column: str,
    bootstrap_current_from_column: str | None,
) -> None:
    if skip_compare:
        print("Skipping GPT-5.4 vs GPT-4 delta export because --skip-compare was set.")
        return
    if compare_results is None:
        print("Skipping GPT-5.4 vs GPT-4 delta export because compare_results is unavailable.")
        return
    if current_label_column != PRIMARY_CURRENT_COLUMN or bootstrap_current_from_column:
        print(
            "Skipping GPT-5.4 vs GPT-4 delta export because the input file is being "
            "bootstrapped or aliased rather than read from persisted gpt54_exposure columns."
        )
        return
    compare_results(
        merged_path=merged_path,
        baseline_label_column="gpt4_exposure",
        task_delta_path=task_delta_path,
        occ_delta_path=occ_delta_path,
        summary_path=compare_summary_path,
        weighting="core",
        allow_missing=allow_missing,
    )
    print(f"Wrote task deltas to {task_delta_path}")
    print(f"Wrote occupation deltas to {occ_delta_path}")
    print(f"Wrote compare summary JSON to {compare_summary_path}")


def run_all(
    merged_path: Path,
    bls_path: Path,
    occ_level_output_path: Path,
    weighted_occ_output_path: Path,
    occ_share_output_path: Path,
    curves_output_path: Path,
    summary_output_path: Path,
    task_delta_path: Path,
    occ_delta_path: Path,
    compare_summary_path: Path,
    primary_weighting: str,
    current_label_column: str,
    bootstrap_current_from_column: str | None,
    allow_missing: bool,
    skip_compare: bool,
) -> None:
    source_rows = read_delimited_rows(merged_path, "\t")
    rows = normalize_rows(
        source_rows,
        current_label_column=current_label_column,
        bootstrap_current_from_column=bootstrap_current_from_column,
    )
    bls_stats = bls_stats_by_occ_code(bls_path)

    primary_occ_rows = aggregate_scores(
        rows,
        bls_stats,
        weighting=primary_weighting,
        allow_missing=allow_missing,
    )
    secondary_weighting = "equal" if primary_weighting == "core" else "core"
    secondary_occ_rows = aggregate_scores(
        rows,
        bls_stats,
        weighting=secondary_weighting,
        allow_missing=allow_missing,
    )
    share_rows = build_occ_share_table(rows, bls_stats, allow_missing=allow_missing)
    curves_rows = exposure_curves(share_rows)
    summary = summary_payload(
        rows,
        primary_occ_rows,
        share_rows,
        curves_rows,
        primary_weighting=primary_weighting,
        current_label_column=current_label_column,
        bootstrap_current_from_column=bootstrap_current_from_column,
    )

    write_occ_level_csv(occ_level_output_path, primary_occ_rows)

    weighted_fieldnames = list(primary_occ_rows[0].keys()) if primary_occ_rows else []
    if not weighted_fieldnames and secondary_occ_rows:
        weighted_fieldnames = list(secondary_occ_rows[0].keys())
    write_delimited_rows(
        weighted_occ_output_path,
        weighted_fieldnames,
        primary_occ_rows + secondary_occ_rows,
        "\t",
    )

    share_fieldnames = list(share_rows[0].keys()) if share_rows else []
    write_delimited_rows(occ_share_output_path, share_fieldnames, share_rows, "\t")

    curves_fieldnames = ["source", "population", "relevance_type", "threshold", "count", "percent"]
    write_delimited_rows(curves_output_path, curves_fieldnames, curves_rows, "\t")

    ensure_parent(summary_output_path)
    summary_output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Wrote primary occupation file to {occ_level_output_path}")
    print(f"Wrote multi-weight occupation file to {weighted_occ_output_path}")
    print(f"Wrote occupation task-share file to {occ_share_output_path}")
    print(f"Wrote exposure curves file to {curves_output_path}")
    print(f"Wrote analysis summary JSON to {summary_output_path}")

    maybe_compare(
        merged_path=merged_path,
        task_delta_path=task_delta_path,
        occ_delta_path=occ_delta_path,
        compare_summary_path=compare_summary_path,
        allow_missing=allow_missing,
        skip_compare=skip_compare,
        current_label_column=current_label_column,
        bootstrap_current_from_column=bootstrap_current_from_column,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Pure-Python rerun analysis replacing the main Jupyter notebook workflow.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--merged-path", type=Path, default=DEFAULT_MERGED_PATH)
    parser.add_argument("--bls-path", type=Path, default=DEFAULT_BLS_PATH)
    parser.add_argument(
        "--occ-level-output-path",
        type=Path,
        default=DEFAULT_OCC_LEVEL_OUTPUT,
    )
    parser.add_argument(
        "--weighted-occ-output-path",
        type=Path,
        default=DEFAULT_WEIGHTED_OCC_OUTPUT,
    )
    parser.add_argument(
        "--occ-share-output-path",
        type=Path,
        default=DEFAULT_OCC_SHARE_OUTPUT,
    )
    parser.add_argument(
        "--curves-output-path",
        type=Path,
        default=DEFAULT_CURVES_OUTPUT,
    )
    parser.add_argument(
        "--summary-output-path",
        type=Path,
        default=DEFAULT_SUMMARY_OUTPUT,
    )
    parser.add_argument(
        "--task-delta-path",
        type=Path,
        default=DEFAULT_TASK_DELTA_PATH,
    )
    parser.add_argument(
        "--occ-delta-path",
        type=Path,
        default=DEFAULT_OCC_DELTA_PATH,
    )
    parser.add_argument(
        "--compare-summary-path",
        type=Path,
        default=DEFAULT_COMPARE_SUMMARY_PATH,
    )
    parser.add_argument(
        "--primary-weighting",
        choices=WEIGHTINGS,
        default="core",
        help="Primary occupation-level aggregation used for the main CSV and summary.",
    )
    parser.add_argument(
        "--current-label-column",
        default=PRIMARY_CURRENT_COLUMN,
        help="Task-level label column to treat as the current model output.",
    )
    parser.add_argument(
        "--bootstrap-current-from-column",
        default=None,
        help="Optional fallback label column for dry runs against legacy files without gpt54_exposure.",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Skip rows without current labels instead of failing.",
    )
    parser.add_argument(
        "--skip-compare",
        action="store_true",
        help="Do not write GPT-5.4 vs GPT-4 delta exports.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        run_all(
            merged_path=args.merged_path,
            bls_path=args.bls_path,
            occ_level_output_path=args.occ_level_output_path,
            weighted_occ_output_path=args.weighted_occ_output_path,
            occ_share_output_path=args.occ_share_output_path,
            curves_output_path=args.curves_output_path,
            summary_output_path=args.summary_output_path,
            task_delta_path=args.task_delta_path,
            occ_delta_path=args.occ_delta_path,
            compare_summary_path=args.compare_summary_path,
            primary_weighting=args.primary_weighting,
            current_label_column=args.current_label_column,
            bootstrap_current_from_column=args.bootstrap_current_from_column,
            allow_missing=args.allow_missing,
            skip_compare=args.skip_compare,
        )
    except Exception as exc:  # pragma: no cover - CLI surface
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
