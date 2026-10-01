# Version 1 local validation baseline

This pipeline only reads challenge TSVs. It has no test-inference or submission entry point. Run from the original `student_resource` root. It preserves all original data and existing analysis files.

## Environment and reproduction

Use Python 3.12.14, NumPy 2.3.5, and SQLite 3.53.1 with FTS5 and the trigram tokenizer. The machine's bundled runtime is referenced in `reproduce.ps1`; edit only that interpreter path on another machine. No network or pretrained models are used. Original matcher code and from-scratch model artifacts are MIT licensed; parameter counts are saved with each model. NumPy is a numerical software dependency, not a pretrained model.

Exact PowerShell commands from the project root:

```powershell
& ./code/business_entity_resolution/reproduce.ps1 -Stage Smoke
& ./code/business_entity_resolution/reproduce.ps1 -Stage Full -Workers 8
# Refit and verify using completed retrieval artifacts:
& ./code/business_entity_resolution/reproduce.ps1 -Stage Evaluate
```

The full path means **full validation workflow**, not all S1 training rows and never test inference. The corpus searched is every training S2/S3 row. Query entities are the fixed 6,000-entity training sample. Read the root `baseline_execution_plan.md` for estimates. Eight workers use approximately 4–5 GB combined process RAM; reduce workers if less memory is available. The full run is restartable by committed source-index batches and complete 50-query shards. Do not run two orchestrators against the same output directory.

`artifacts/v1/` contains immutable-data SHA256 fingerprints, source-index manifests, environment versions, split JSON/TSV, SQLite indexes, label-free IDF estimates, and per-run outputs. A config mismatch raises an error. New configurations should use a new run directory/version, preserving existing results. Model fitting currently overwrites the same deterministic baseline model names when explicitly rerun; copy a run before changing its model settings.

## Method

Sample the 6,000 lowest seeded BLAKE2 entity hashes from all training S1 rows. Union sampled entities sharing a suffix-normalized name or sorted address-token key. Deterministically balance these groups by country and capped match cardinality across 50% fit, 25% tune, 25% final validation. Labels are used for stratification and supervised fitting/assessment, never to insert targets into candidate sets. Groups are conservative exact-key approximations; semantic clusters and shared ground-truth target checks are reported separately.

Raw Unicode fields are retained. Derived views include NFKC lowercase; punctuation/space normalization preserving Indic marks; Latin accent folding; name/address tokens and 3–5-grams; legal suffix keys; scripts; numeric and postal-like tokens; missingness. Suffix removal only removes a trailing sequence from the documented token list. It never declares a match. French legal suffixes are retained rather than requiring a closed legal-form list.

Each source is indexed separately. Routes are exact folded name; suffix key; rare name token FTS; approximate sparse binary-TF IDF character reranking over trigram-FTS seeds; address token FTS; reordered exact address key; numeric-address component blocking; address character reranking. Each route retains 100 candidates from a 400-record seed budget. Character candidates are **approximate blocked top-k**, not exact global TF-IDF top-k. Address character reranking depends on lexical seeds and cannot recover text with no shared token. IDF is label-free, fitted on a fixed-stride 20k-row target-corpus sample. Unknown grams receive weight 1. Country is an open string agreement feature; every route is country agnostic. No transliteration package was available.

The numeric route was broadened using four fitting-only misses; the interrupted initial candidate run is preserved in `artifacts/v1/preliminary_run_6000`. All revised candidates are generated from the same frozen version-2 retrieval policy. Budgets 10/20/50/100 are assessed by route rank; 100 is the predetermined final scoring set.

Per-shard compressed candidate JSON records exact IDs, original corpus rowids, route ranks, and route scores. NPZ files contain the same candidate order with 59 float32 pair features. Gzipped view caches preserve every derived representation. No candidate is added from ground truth. Labels are joined afterward. ID lists in local validation TSVs are deduplicated; predictions may contain every above-threshold candidate or none.

Models: standardized L2 logistic regression using deterministic Adam; 70 histogram Newton-boosted binary trees, depth 3, 32 bins, learning rate 0.12, leaf L2=5. These are compact original NumPy fallbacks because no practical third-party ML library is installed. All official candidate positives, hard candidate negatives and top route-rank candidates are retained; 20 tail negatives per S1/source are sampled with inverse inclusion weights. Both models receive exactly the same pairs and weights. Parameters and feature order are serialized to JSON, with reload tests. No probability calibration is applied.

Threshold selection maximizes exact macro F0.5 on tuning entities over a broad grid then a refined grid; ties choose the higher threshold. The selected model is also chosen on tuning only. Holdout metrics are produced afterward with frozen choices. Shared thresholds remain the default; source reliability bins are reported, and no French threshold is invented.

## Verification and limitations

Tests cover metric edge cases, ordering, duplicates, TSV parsing, Unicode, entity-group isolation, multi-route deduplication, source validity, multiple/empty sets, models and reloads. `verify.py` checks all scored IDs against full training indexes, unchanged original hashes, finite features, TSV metric roundtrips, and the supplied validator with `--check-ids` on a **training-only** fixture. Files named `test_source*.tsv` inside that fixture are copies of training validation records solely to satisfy the validator's interface. They are not challenge test predictions. A bounded 1,000-row sample per test source is used only for unlabeled schema/Unicode checks; no test retrieval or scoring occurs.

Final candidate recall and model quality are in the root `baseline_validation_report.md`. Retrieval below 99.5% is a limitation and prevents claiming readiness for full inference. Limited cross-script evidence, shared/common names, singleton uncertainty, approximate retrieval, sampled S1 groups, no unseen-country labels, and from-scratch fallback-model limitations must be considered before any future deployment or submission.
