# GPT-5.4 Exposure Rerun

This repo does not include the original GPT-4 labeling workflow. It only includes the downstream analysis notebooks and the saved task-level / occupation-level outputs. The rerun path added here keeps the original task universe and weighting logic intact, and adds a reproducible `codex exec` classification pipeline beside the notebooks.

## What This Adds

- `code/gpt54_exposure_prompt.txt`
  - The default early-2026 GPT-5.4 exposure rubric and output contract.
  - The default rubric assumes a supervised agentic workflow, not bare chat alone.
- `code/gpt54_exposure_prompt_conservative_v1.txt`
  - The earlier, stricter rubric preserved for comparison runs.
- `code/gpt54_exposure_schema.json`
  - The strict JSON schema used with `codex exec --output-schema`.
- `code/gpt54_rerun.py`
  - A standard-library pipeline with four subcommands:
    - `make-batches`
    - `classify-batch`
    - `classify-all`
    - `merge`
    - `compare`
- `code/gpt54_analysis.py`
  - A standard-library replacement for the main analysis notebook.
  - Produces occupation-level outputs, task-share tables, exposure curves, and summary JSON.

Runtime artifacts are written under `.gpt54_rerun/` and ignored by git.

## Workflow

1. Create shard files from `data/full_onet_data.tsv`.
2. Run `codex exec` once per shard, not once per row.
3. Save each shard response as normalized JSON in `.gpt54_rerun/results/`.
4. Merge the shard outputs back into `data/full_labelset_gpt54.tsv`.

The merged TSV preserves the legacy `full_labelset.tsv` columns when available and appends:

- `gpt54_exposure`
- `gpt54_reason`
- `gpt54_confidence`
- `gpt54_model`
- `gpt54_reasoning_effort`
- `gpt54_prompt_version`
- `gpt54_batch_id`
- `gpt54_alpha`
- `gpt54_beta`
- `gpt54_gamma`

By default, the rerun now uses the agentic rubric version
`gpt54_early2026_agentic_rubric_v2`. If you want to run the archived
conservative rubric instead, pass both:

```bash
python3 code/gpt54_rerun.py classify-all \
  --prompt-template-path code/gpt54_exposure_prompt_conservative_v1.txt \
  --prompt-version gpt54_early2026_conservative_rubric_v1
```

## Recommended Commands

Create 100-row shards:

```bash
python3 code/gpt54_rerun.py make-batches --shard-size 100
```

Smoke-test one shard at medium reasoning:

```bash
python3 code/gpt54_rerun.py classify-all --max-batches 1 --reasoning-effort medium
```

Run the remaining shards:

```bash
python3 code/gpt54_rerun.py classify-all --reasoning-effort medium
```

Merge completed shard outputs into a notebook-friendly TSV:

```bash
python3 code/gpt54_rerun.py merge
```

Generate the occupation-level outputs and summary files without using the notebook:

```bash
python3 code/gpt54_analysis.py
```

Generate machine-readable task and occupation deltas against the legacy GPT-4 labels:

```bash
python3 code/gpt54_rerun.py compare
```

The primary occupation output is core-weighted, which matches the tracked legacy
`data/occ_level.csv` behavior. The analysis script also writes a second TSV with
both `core` and `equal` weightings for comparison.

For a no-cost smoke test against the current repo data, you can bootstrap the
analysis from the legacy GPT-4 labels:

```bash
python3 code/gpt54_analysis.py \
  --merged-path data/full_labelset.tsv \
  --bootstrap-current-from-column gpt4_exposure \
  --skip-compare
```

## Low-Confidence Reruns

The runner is designed to be resume-safe. For a second pass on low-confidence rows:

1. Extract the `row_id` values you want to rerun from `data/full_labelset_gpt54.tsv`.
2. Put one `row_id` per line in a text file.
3. Rebuild shards only for that subset:

```bash
python3 code/gpt54_rerun.py make-batches \
  --row-id-file low_confidence_ids.txt \
  --shard-size 100 \
  --work-dir .gpt54_rerun_low_conf
```

4. Rerun them with higher reasoning:

```bash
python3 code/gpt54_rerun.py classify-all \
  --reasoning-effort high \
  --work-dir .gpt54_rerun_low_conf
```

5. Merge again, letting the low-confidence pass override the original run by `row_id`:

```bash
python3 code/gpt54_rerun.py merge \
  --results-dir .gpt54_rerun/results \
  --results-dir .gpt54_rerun_low_conf/results
```

## Notes

- The current local Codex CLI defaults to very high startup overhead per invocation, so batching is important.
- The pipeline uses `row_id` from the first column of the existing TSVs to keep joins deterministic.
- The current implementation only covers exposure labels (`E0/E1/E2`). It intentionally does not rerun the legacy automation rubric.
- The legacy comparison baseline remains the tracked GPT-4-era files already in the repo, especially `data/full_labelset.tsv`. The rerun pipeline never overwrites that file.
- `gpts_are_gpts_script2.ipynb` is not ported yet because its required input files (`task_ratings_file_7-12.csv` and `DWA_Tasks_Labels.tsv`) are not included in this repository.
- `alpha / beta / gamma` are derived as:
  - `alpha = E1`
  - `beta = E1 + 0.5 * E2`
  - `gamma = E1 + E2`
