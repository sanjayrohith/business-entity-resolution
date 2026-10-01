# Version 2 local validation

This directory extends the verified V1 system without changing its code, models, reports or original datasets. It has no challenge-test inference entry point. All command paths below are relative to the student_resource project root.

## Environment

Use the same Python 3.12 interpreter as V1. Install the pinned ordinary software dependencies in the isolated workspace directory:

```powershell
$v2Python='python'   # any Python 3.12 interpreter
& $v2Python -m pip install --target artifacts/v2/packages --only-binary=:all: -r code/business_entity_resolution/v2/requirements.txt
```

LightGBM is MIT licensed; XGBoost is Apache-2.0. Both are trained from scratch, with model-size counts saved. AnyAscii is an ISC-licensed local character-transliteration dependency, not a pretrained model or business identity resource. Its output is an additional retrieval/feature view; original Unicode is retained. Sparse-dot-topn is Apache-2.0. Exact installed distribution metadata and license paths are saved in artifacts/v2/environment.json. No network is used by the pipeline itself.

## Reproduction from completed V1

```powershell
& ./code/business_entity_resolution/v2/reproduce.ps1 -Stage Prepare
& ./code/business_entity_resolution/v2/reproduce.ps1 -Stage Pilot
& ./code/business_entity_resolution/v2/reproduce.ps1 -Stage Validation
& ./code/business_entity_resolution/v2/reproduce.ps1 -Stage Benchmark
& ./code/business_entity_resolution/v2/reproduce.ps1 -Stage Verify
```

Prepare audits V1 preservation, creates the new split, profiles V1, builds the initial sampled-vocabulary indexes, builds corrected full-corpus hashed indexes from their cached normalized records, and builds targeted subset indexes. The initial indexes are retained as a documented controlled experiment and a reusable normalized-record cache. Fresh reproduction requires completed V1 artifacts because its IDs are excluded from the new split. Its original README documents how to regenerate those artifacts. Do not run different settings in the same V2 output directory; use a new experiment directory/version.

Pilot uses fitting entities only, measures recall at 10/20/50/100, and compares against the cached original V1 fitting candidates. Validation retrieves 100 per ordinary route, then keeps the union of ranks through 50 plus ranks through 100 for non-Latin or missing-address targets. Targeted routes retain 50 each. This label-free final candidate filter is applied before feature generation; the confirmation candidate TSV records exactly the candidates scored. It searches every training S2/S3 target, generates pair features with four bounded local processes, compares three matchers, selects thresholds on tune, then evaluates confirmation once. Verification recomputes set metrics, checks IDs against the entire training corpus, validates outputs on a training-only fixture and hashes all original data and V1 critical artifacts.

## Split and retrieval

Seed 20260926. 10,000 fresh S1 entities selected by seeded hashes, excluding V1 IDs and their normalized name/address grouping keys. Groups use suffix names, sorted address tokens and shared positive targets. Country/cardinality and hard-case flags balance fit/tune/confirmation. Confirmation labels affect split balance only until the development choices freeze.

Targets are indexed separately by source. Retrieval runs as native sparse top-k batches over 250,000-row partitions, merging top-k globally. Binary TF-IDF uses a 2^20 hashing space and full-corpus document frequencies. Names use words and 3–5-grams; addresses use words and trigrams, with leading-zero digit normalization. Very frequent features are suppressed at 3% of documents for character features and 10% for word features. Hash collisions and this pruning are measured approximations, not semantic equivalence. No dense S1-target matrix is built.

Exact names, trailing-suffix keys and sorted address-token keys use batched indexed SQL joins. These are a few batch operations, not per-candidate record fetches. Numeric/postal information participates in address word/character search and features; its route flag denotes shared numeric components among address-word candidates, rather than a separate strict numeric search. Name-plus-address sparse search provides a joint-evidence route. Two bounded subset searches target non-Latin names (ASCII transliteration 2–4-grams) and missing-address records (name 3–5-grams). Every route is country-agnostic.

## Features, matching and decisions

Preserve V1's 59 feature concepts, replacing the name edit implementation with RapidFuzz normalized Indel similarity and FTS route scores with sparse route scores. Add 12 focused features (token ordering, name truncation/content containment, transliteration, partial/alphabetic addresses, noisy-number similarity, relative name/address evidence) and nine route-hit/rank/score fields for three targeted routes: 80 total. Full feature order and version differences are saved. Exact names never force positives; predictions may be empty or contain multiple links.

Fit retains all retrieved positives, hard negatives, top reciprocal-route ranks and weighted sampled tail negatives. No missed positive is injected. Logistic regression is the sanity baseline; LightGBM and XGBoost are the production-model comparison. Early stopping uses the tuning fold; confirmation remains untouched. Broad then refined threshold search maximizes the exact per-S1 macro F0.5. Source thresholds require >0.002 overall tuning improvement and >0.001 in each deterministic tuning half. No country-specific threshold or calibration fit is used.

## Artifacts and limitations

artifacts/v2/run contains exact source candidate JSONL, feature NPZ shards, model files, threshold grids, frozen selection, local confirmation TSVs, errors, metrics and verification records. Files named test_source*.tsv within training_validator_fixture are copies of training records for the organizer validator interface. They are not challenge test predictions.

Read version2_validation_report.md for verified outcomes and readiness. Runtime extrapolations are estimates from local training-query batches, not a full test run. Differences in country mix, longer French addresses, storage/cache conditions and output volume can affect actual runtime. No French labeled quality claim is possible.
