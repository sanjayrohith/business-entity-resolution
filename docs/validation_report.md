# Version 2 validation report

**READY FOR FIRST TEST INFERENCE**

Completed and verified local validation on 10,000 fresh S1 entities against all 10,320,219 training S2/S3 records. Selected **lightgbm** on tuning entities; evaluated its frozen decisions on 2,500 confirmation entities. Exact macro F0.5: **0.945316**. Candidate recall: **99.0257%**.
No challenge test retrieval/scoring, challenge matching_results.tsv, submission or upload was performed. V1 critical artifacts, code, report and all seven original dataset SHA256 hashes were verified unchanged.

## V1 versus V2

| Measure | V1 verified original holdout | V2 fresh confirmation |
|---|---|---|
| candidate recall | 0.986513 | 0.990257 |
| US candidate recall | 0.992051 | 0.997702 |
| India candidate recall | 0.977995 | 0.978817 |
| cross-script candidate recall | 0.910995 | 0.922581 |
| missing-address candidate recall | 0.925620 | 0.958869 |
| low-name-similarity candidate recall | 0.926782 | 0.935335 |
| confirmation entities | 1499 | 2500 |
| candidate mean | 706.504336 | 542.207600 |
| candidate p99 | 871.020000 | 645.010000 |
| retrieval throughput, S1/s | 2.307 (includes V1 feature/cache work) | 52.024291 |
| estimated full-test retrieval | 8.7 days (includes V1 feature/cache work) | 9.25 hours (retrieval only) |
| estimated retrieval + features | 8.7 days | 18.53 hours |
| model | NumPy histogram boost | lightgbm |
| threshold(s) | .621 | [0.642, 0.642] |
| macro F0.5 | 0.904471 | 0.945316 |
| macro precision | 0.941628 | 0.965808 |
| macro recall | 0.836120 | 0.905370 |
| singleton accuracy | 0.860465 | 0.850746 |
| classifier false negatives | 789 | 727 |
| retrieval false negatives | 70 | 84 |
| false positives | 175 | 211 |
| selected model fit/score/threshold seconds | 261.89 | 61.925869 |
| peak RAM, conservative process bound | 4.801 GiB | 2.432 GiB |
| artifact disk | 8.248 GiB | 23.840 GiB including packages and prototype caches |

**Comparison limits:** V1 and V2 final scores use different entities and sample sizes. Raw FP/FN counts are not directly comparable; use rates and slice recall. V1’s published retrieval timer included features and cache serialization; V2 reports both retrieval-only and retrieval-plus-feature estimates to expose that difference. Runtime estimates exclude future test-index construction, model scoring, final TSV serialization and distribution/hardware variation.
V1 classifier misses were 789/5190 = 15.20% of true links. V2 classifier misses are 727/8622 = 8.43%. V2 cluster-bootstrap 95% interval for F0.5: [0.9395680517971344, 0.9509900957183062].

## Environment and licenses

| Package | Installed version |
|---|---|
| RapidFuzz | 3.14.6 |
| anyascii | 0.3.3 |
| cloudpickle | 3.1.2 |
| joblib | 1.6.0 |
| lightgbm | 4.7.0 |
| narwhals | 2.26.0 |
| numpy | 2.5.3 |
| polars | 1.44.2 |
| polars-runtime-32 | 1.44.2 |
| psutil | 7.2.2 |
| scikit-learn | 1.9.1 |
| scipy | 1.18.1 |
| sparse-dot-topn | 1.2.0 |
| threadpoolctl | 3.7.0 |
| xgboost | 3.4.1 |

Packages were installed under artifacts/v2/packages. Exact pinned dependencies: code/business_entity_resolution/v2/requirements.txt. LightGBM’s bundled LICENSE is MIT; XGBoost metadata and bundled LICENSE declare Apache-2.0. Selected trees were learned only from official fitting labels. Saved parameter counts are conservative counts of numeric model values/nodes, far below 8B. AnyAscii is an ISC-licensed deterministic character mapping used solely as a normal local transliteration library, not a model or business lookup. Full dependency metadata/license paths are in environment.json. No pretrained model or identity data was downloaded.
Python: 3.12.14 (main, Aug 25 2026, 14:01:42) [MSC v.1944 64 bit (AMD64)]. Physical RAM: 15.71 GiB; available at audit: 4.91 GiB. Windows CIM inventory was denied; psutil supplied RAM measurements.

## V1 bottleneck profile

A deterministic 30-fitting-entity, two-source cProfile run measured 113.47 seconds and 21,391 pairs. cProfile adds overhead and inclusive times overlap; these numbers identify hot paths rather than replace the original wall-clock benchmark.
| Component | Measured seconds / finding |
|---|---|
| SQLite execute self time | 79.267; 76,039 calls |
| FTS cumulative | 69.170 |
| target fetch + view cumulative | 29.069; 75,619 gets |
| normalization/view cumulative | 18.173 |
| ranking (sorted) cumulative | 35.098; includes nested fetch/scoring |
| TF-IDF cosine cumulative | 6.185 |
| feature generation cumulative | 6.568 |
| route union/dedup add cumulative | 0.309 |
| compression self time | 0.585; JSON serialization/record cache adds further cost |

Separate unprofiled FTS route measurements, including address-token and numeric-address search, are in v1_route_timing.json. Raw call records and the complete profile are in v1_profile.json/.txt/.prof. Disk I/O timings are process-level wall times including filesystem cache; no physical-device bandwidth claim is made.

| Unprofiled route/stage | Seconds |
|---|---|
| query_normalization | 0.002419 |
| retrieval_total | 31.765841 |
| feature_generation | 0.730011 |
| name_token_fts | 1.770292 |
| name_character_fts | 11.180790 |
| address_token_fts | 11.697544 |
| numeric_address_fts | 0.497133 |

Removed bottlenecks: native sparse batched top-k replaces FTS search; batched corpus fetches replace per-record SQLite access; normalized target fields are reused from versioned caches; RapidFuzz replaces SequenceMatcher edit work. Remaining Python work is bounded candidate feature generation, not loops comparing every query against every target.

## Split and leakage controls

Seed 20260926. Fit 5,001, tune 2,499, confirmation 2,500; 9,708 groups, maximum 5 entities. Excluded all V1 entities and matching V1 sampled suffix-name/reordered-address keys. Grouped new entities by those keys and shared positive target IDs. Balanced country × 0/1/2/3/4/5+ cardinality plus cross-script, missing-address, low-name-similarity and common-name indicators. Exact split IDs and all stratum counts are persisted in split.json/split_ids.tsv.
Confirmation labels were used only to balance the split before modeling, and evaluated only after the model/threshold freeze. All retrieval changes used fitting-only pilots. IDF and indexes use unlabeled full training-target text. Calibration was not fitted. Early stopping and threshold/model selection use tune only. Group checks verify no shared grouping key or positive target crosses folds. Group-order ties and tuning-half assignments use the inherited stable hash seed 2026; pool selection, sampling, tree seeds and bootstrap use 20260926. These are practical sampled clusters, not a proof against every semantic near-duplicate.

## Retrieval architecture and controlled experiments

Separate S2/S3 indexes; 250,000-target partitions; full-corpus hashed binary TF-IDF in 2^20 dimensions. Word and 3–5-gram name routes; word/numeric/postal and trigram address routes; original Unicode plus local transliteration view; leading-zero numeric normalization. Common-feature pruning: character DF >3%, word DF >10%. Native sparse top-k returns bounded results; no dense S1×target similarity array. Exact name, suffix name and reordered-address routes are batch SQL joins. Suffix equality never declares a match.
Initial sampled-vocabulary retrieval omitted rare names/address fragments. Full-corpus hashing removes that vocabulary gap at the cost of possible hash collisions. Targeted routes search non-Latin target names with transliteration 2–4-grams, missing-address targets with name 3–5-grams, and joint name/address evidence. The numeric route feature means numeric overlap within address-word retrieval; it is not a separate strict numeric-only index. Country equality is a feature, not a closed country list or exclusion.
Final policy: ordinary routes retrieve 100, then retain ranks through 50 for every target and through 100 for non-Latin/missing-address targets. Three rescue routes retain 50 each. Filtering precedes scoring; the saved final candidate JSONL and confirmation candidate TSV are the exact model input set. Larger raw route caches are explicitly earlier retrieval artifacts.
| Paired fitting control, same 500 V1 entities | Candidate recall | Mean candidates |
|---|---|---|
| V1 frozen candidates | 0.983372 | 702.852000 |
| V2 final policy | 0.991972 | 544.436000 |

Fresh 500-fitting-entity final-policy pilot recall: 99.0681%. Pilot comparisons and 10/20/50/100 budget results are retained, including failed/weaker prototypes. Candidate recall was checked before matcher training. The 99.5% target remains a working target, not an organizer rule.

## Confirmation candidate quality

| Slice | True links | Retrieved | Recall |
|---|---|---|---|
| all | 8622 | 8538 | 0.990257 |
| US | 5223 | 5211 | 0.997702 |
| S2 | 4183 | 4148 | 0.991633 |
| S3 | 4439 | 4390 | 0.988961 |
| exact_name | 1879 | 1879 | 1.000000 |
| India | 3399 | 3327 | 0.978817 |
| low_name_similarity | 866 | 810 | 0.935335 |
| non_latin_name | 620 | 572 | 0.922581 |
| script_DEVANAGARI | 345 | 317 | 0.918841 |
| cross_script | 620 | 572 | 0.922581 |
| india_cross_script | 620 | 572 | 0.922581 |
| common_name | 1349 | 1312 | 0.972572 |
| script_BENGALI | 63 | 60 | 0.952381 |
| missing_address | 389 | 373 | 0.958869 |
| script_GUJARATI | 29 | 27 | 0.931034 |
| script_MALAYALAM | 30 | 28 | 0.933333 |
| script_TAMIL | 43 | 41 | 0.953488 |
| script_KANNADA | 47 | 41 | 0.872340 |
| script_TELUGU | 48 | 44 | 0.916667 |
| script_GURMUKHI | 13 | 12 | 0.923077 |
| script_ORIYA | 2 | 2 | 1.000000 |

| Candidate statistic | Value |
|---|---|
| mean | 542.2076 |
| p50 | 541.0 |
| p90 | 597.0 |
| p99 | 645.0099999999998 |
| max | 705 |
| pairs | 1355519 |
| reduction_ratio | 0.9999474616187893 |

Perfect-classifier candidate oracle macro F0.5: 0.997103. Every official true link remains in recall denominators; no positives were injected into candidates.

## Matcher, features and hard negatives

80 features: V1’s 59 concepts; 12 focused string/partial-address/transliteration/noisy-number/relative-evidence features; nine route hit/rank/score values for the three rescue routes. Feature order is saved in frozen_selection.json and feature completion manifests. Name edit changes from SequenceMatcher to RapidFuzz normalized Indel; retrieval scores/ranks change to sparse routes, so V1 model weights are not silently reused on changed feature semantics.
Fitting pair sampling: {'available': 2711630, 'positive': 17053, 'hard_negative': 808664, 'selected': 1086908, 'tail': 150003, 'seed': 20260926, 'scheme': 'All retrieved positives, all hard collisions/strong-address/missing-address negatives, top 25 reciprocal route ranks, 15 weighted tail negatives per S1/source'}. All three models use the same candidates, fitting rows and weights. Missing-address, common/exact names, strong-address/co-location and numeric conflicts remain probabilistic evidence. No top-1 or mandatory-link rule is used.
| Model | Tune F0.5 | Threshold S2/S3 | Confirmation F0.5 | Macro P | Macro R | Parameters/upper bound | Fit/score/search seconds |
|---|---|---|---|---|---|---|---|
| logistic | 0.904114 | [0.6950000000000001, 0.6950000000000001] | 0.892453 | 0.932135 | 0.818531 | 241 | 10.609567 |
| lightgbm | 0.947038 | [0.642, 0.642] | 0.945316 | 0.965808 | 0.905370 | 90600 | 61.925869 |
| xgboost | 0.946208 | [0.7070000000000001, 0.7070000000000001] | 0.943446 | 0.967059 | 0.894521 | 202070 | 131.753213 |

Logistic uses standardized L2 regression. LightGBM uses up to 600 trees, 31 leaves, learning rate .04, leaf L2=5; XGBoost up to 600 depth-6 trees, rate .05, L2=5. Both use tuning log-loss early stopping (60 rounds). Full configurations are saved. Model reload predictions match saved tuning scores. Logistic is a sanity comparator; final choice is the better tuning LightGBM/XGBoost model.
Broad threshold search followed by 51-point refinement optimizes exact macro F0.5. Every above-threshold pair is predicted. Source thresholds are adopted only with >.002 overall tuning gain and >.001 in each deterministic tuning half; decisions, calibration bins, all threshold metrics and grids are saved per model. No France-specific threshold.

## Final diagnostics

| Measure | Value |
|---|---|
| n_entities | 2500 |
| macro_precision | 0.9658079365079366 |
| macro_recall | 0.9053695238095238 |
| macro_f0_5 | 0.945315956779393 |
| singleton_accuracy | 0.8507462686567164 |
| singleton_count | 134 |
| zero_match_rate | 0.0524 |
| average_links | 3.2088 |
| false_positive_links | 211 |
| false_negative_links | 811 |

| Links per entity | True count | Predicted count |
|---|---|---|
| 0 | 134 | 131 |
| 1 | 130 | 212 |
| 2 | 440 | 509 |
| 3 | 608 | 616 |
| 4 | 548 | 518 |
| 5+ | 640 | 514 |

| Entity slice | N | F0.5 | Macro precision | Macro recall | Singleton accuracy |
|---|---|---|---|---|---|
| India | 990 | 0.927281 | 0.951701 | 0.884535 | 0.814815 |
| US | 1510 | 0.957140 | 0.975057 | 0.919029 | 0.875000 |
| true_cardinality_0 | 134 | 0.850746 | 0.850746 | 0.850746 | 0.850746 |
| predicted_cardinality_0 | 131 | 0.870229 | 0.870229 | 0.870229 | 1.000000 |
| true_cardinality_1 | 130 | 0.876463 | 0.871795 | 0.915385 | n/a |
| predicted_cardinality_1 | 212 | 0.825847 | 0.910377 | 0.688174 | 0.000000 |
| true_cardinality_2 | 440 | 0.940350 | 0.960606 | 0.914773 | n/a |
| predicted_cardinality_2 | 509 | 0.948169 | 0.981336 | 0.875573 | 0.000000 |
| true_cardinality_3 | 608 | 0.952632 | 0.974589 | 0.910636 | n/a |
| predicted_cardinality_3 | 616 | 0.962055 | 0.979437 | 0.923585 | n/a |
| true_cardinality_4 | 548 | 0.961777 | 0.986010 | 0.907847 | n/a |
| predicted_cardinality_4 | 518 | 0.966762 | 0.974903 | 0.949954 | n/a |
| true_cardinality_5 | 640 | 0.961471 | 0.986932 | 0.901183 | n/a |
| predicted_cardinality_5 | 514 | 0.969229 | 0.972153 | 0.966653 | n/a |
| S2 | 2500 | 0.938144 | 0.948433 | 0.922593 | 0.929204 |
| S3 | 2500 | 0.929844 | 0.944433 | 0.907273 | 0.961806 |

Source entity metrics restrict each S1 set to S2/S3 and apply the same singleton convention. Difficulty metrics below are link metrics, not alternative official scores. Low-name similarity is V1 folded 3–5-gram Jaccard <.2; common-name frequency >=5; cross-script uses differing Unicode letter-script sets with a non-Latin target.
| Link slice | Truth | Predicted | TP | Precision | Recall |
|---|---|---|---|---|---|
| all | 8622 | 8022 | 7811 | 0.973697 | 0.905938 |
| US | 5223 | 4892 | 4800 | 0.981194 | 0.919012 |
| S2 | 4183 | 3946 | 3841 | 0.973391 | 0.918240 |
| S3 | 4439 | 4076 | 3970 | 0.973994 | 0.894346 |
| exact_name | 1879 | 1859 | 1840 | 0.989779 | 0.979244 |
| India | 3399 | 3130 | 3011 | 0.961981 | 0.885849 |
| low_name_similarity | 866 | 678 | 626 | 0.923304 | 0.722864 |
| non_latin_name | 620 | 541 | 504 | 0.931608 | 0.812903 |
| script_DEVANAGARI | 345 | 309 | 283 | 0.915858 | 0.820290 |
| cross_script | 620 | 541 | 504 | 0.931608 | 0.812903 |
| india_cross_script | 620 | 541 | 504 | 0.931608 | 0.812903 |
| common_name | 1349 | 1204 | 1166 | 0.968439 | 0.864344 |
| script_BENGALI | 63 | 54 | 52 | 0.962963 | 0.825397 |
| missing_address | 389 | 210 | 173 | 0.823810 | 0.444730 |
| script_GUJARATI | 29 | 30 | 26 | 0.866667 | 0.896552 |
| script_MALAYALAM | 30 | 24 | 24 | 1.000000 | 0.800000 |
| script_TAMIL | 43 | 33 | 33 | 1.000000 | 0.767442 |
| script_KANNADA | 47 | 41 | 38 | 0.926829 | 0.808511 |
| script_TELUGU | 48 | 39 | 37 | 0.948718 | 0.770833 |
| script_GURMUKHI | 13 | 9 | 9 | 1.000000 | 0.692308 |
| script_ORIYA | 2 | 2 | 2 | 1.000000 | 1.000000 |

| Error | Category (overlapping) | Links |
|---|---|---|
| false_positive | strong_address | 105 |
| false_positive | low_name_similarity | 52 |
| false_positive | suffix_name | 44 |
| false_positive | common_name | 38 |
| false_positive | missing_address | 37 |
| false_positive | other | 24 |
| false_positive | exact_name | 19 |
| false_positive | numeric_conflict | 17 |
| false_negative | classifier_miss | 727 |
| false_negative | low_name_similarity | 240 |
| false_negative | missing_address | 216 |
| false_negative | common_name | 183 |
| false_negative | non_latin_name | 116 |
| false_negative | cross_script | 116 |
| false_negative | india_cross_script | 116 |
| false_negative | retrieval_miss | 84 |
| false_negative | script_DEVANAGARI | 62 |
| false_negative | exact_name | 39 |
| false_negative | script_BENGALI | 11 |
| false_negative | script_TELUGU | 11 |
| false_negative | script_TAMIL | 10 |
| false_negative | script_KANNADA | 9 |
| false_negative | script_MALAYALAM | 6 |
| false_negative | script_GURMUKHI | 4 |
| false_negative | script_GUJARATI | 3 |

False-merge and missed-link records with IDs, scores and text are saved in each model’s errors JSON. Categories overlap. Strong-address errors can reflect co-located businesses; cross-script/partial names and missing addresses remain difficult. These are error descriptions, not allegations of incorrect labels.

Focused findings: 105 false-positive links have strong address similarity; weak names with partial/shared addresses can also be scored overconfidently (e.g. names differing by a single character at the same address). Missing addresses account for 216 missed links, of which 200 were retrieved and rejected. Cross-script link recall improved substantially, but candidate retrieval still misses 48 of 620 cross-script links. These confirmation findings are for future experiments, not changes to the frozen model. Record-level error examples are omitted from this public report because they quote competition data.

## Runtime, memory, disk and cache locations

| Stage | Seconds | Peak process RAM GiB |
|---|---|---|
| S2 retrieval | 242.200782 | 0.747875 |
| S3 retrieval | 213.368310 | 0.741066 |
| Four-worker feature generation | 368.403517 | 1.230175 |
| All model training/selection | 221.130684 | 1.790447 |
| Verification | 153.745914 | 0.459259 |

Retrieval throughput: 52.02 S1/s across both complete target corpora, measured on 10,000 entities. Linear full-test retrieval estimate: 9.25 hours; retrieval plus feature generation: 18.53 hours. This is an extrapolation, not test inference. Fixed index-load overhead makes pilot-size extrapolation pessimistic; longer French addresses and larger output volumes may increase cost. Peak RAM is a conservative sum of worker peaks, not a simultaneously sampled machine peak.
Original sequential-source validation retrieval: 455.57 seconds; four-worker features: 368.40 seconds. Parallel replay benchmark: {'seconds': 192.2179019000032, 'entities': 10000, 'source_processes': 2, 'native_threads_each': 6, 'candidate_content_identical': {'2': True, '3': True}, 'peak_ram_bound': 1597460480, 'estimated_test_hours': 9.250721461928865}; feature replay: {'seconds': 192.86501750000025, 'workers': 8, 'identical_feature_pairs': 5421015, 'peak_ram_bound': 2611032064, 'estimated_test_hours': 9.28186469109779}. Replay candidates and every feature value must match the original frozen run before its throughput is used. Parallel retrieval uses two source processes with six native threads each; feature replay uses eight local workers. Benchmark costs and caches are additional work in this experiment.
The parallel replay benefits from previously accessed filesystem caches. Its speedup includes both concurrency and cache effects; a cold-start full run may be slower.
Disk 23.840 GiB under artifacts/v2, including installed packages, preserved prototypes and binary/weighted sparse caches. index_s2/index_s3 hold normalized raw/derived Parquet and prototype matrices; index2_s2/index2_s3 hold full-corpus DF/vectorizer/sparse matrices; targeted_s2/targeted_s3 hold ID maps and rescue indexes; run holds exact candidates, features, models, thresholds, metrics and verification. Index manifests contain measured build times and fingerprints. One partition at a time bounds retrieval memory.

## Reproduction and verification

Run from the project root in PowerShell; install pinned dependencies as described in code/business_entity_resolution/v2/README.md:
```powershell
& ./code/business_entity_resolution/v2/reproduce.ps1 -Stage Prepare
& ./code/business_entity_resolution/v2/reproduce.ps1 -Stage Pilot
& ./code/business_entity_resolution/v2/reproduce.ps1 -Stage Validation
& ./code/business_entity_resolution/v2/reproduce.ps1 -Stage Benchmark
& ./code/business_entity_resolution/v2/reproduce.ps1 -Stage Verify
```
Prepare uses the completed V1 artifacts as immutable inputs; its README provides their reproduction. Use a separate version directory for changed settings. Re-running validation refits the same deterministic configuration and is reproduction, not authorization to tune on confirmation.
| Check | Result |
|---|---|
| v1_unchanged | True |
| all_original_dataset_hashes_unchanged | True |
| finite_lineage_verified_pairs | 5421015 |
| logistic_validator | PASS with --check-ids, training-only fixture |
| lightgbm_validator | PASS with --check-ids, training-only fixture |
| xgboost_validator | PASS with --check-ids, training-only fixture |
| metric_and_candidate_subset | True |
| tests | 25 passed |
| selected_model_license | MIT |
| parameter_limit | True |
| synthetic_unicode_open_country | True |

The metric is exact per-S1 macro F0.5, including true-empty/predicted-empty=1 and true-empty/nonempty=0. Supplementary macro P/R use both-empty=1 and other undefined denominators=0, as in V1. Original metric/TSV/Unicode tests and V2 sparse-product/merge/group/model tests pass. Organizer validator passes with --check-ids on training-only fixtures; prediction-subset checks are independent. Synthetic French Unicode and previously unseen country labels remain valid. No French labeled quality is claimed.

## Readiness, limitations and next experiments

**READY FOR FIRST TEST INFERENCE**. Operational working gates used here: candidate recall at least the V1 level (.9865), F0.5 >.9145, retrieval estimate <12 hours, and retrieval-plus-feature estimate <24 hours. These are explicitly local feasibility checks, not official challenge constraints. Observed values: recall 0.990257, F0.5 0.945316, retrieval 9.25 h, retrieval+features 18.53 h.
Remaining limits: 99.5% candidate recall is not guaranteed; difficult slices are smaller and less stable; character hashing/pruning approximates similarity; no French supervision; index/cache build and full-scale output costs are additional. Readiness is a local engineering assessment and does not authorize test inference.
Next three experiments: (1) enlarge grouped fitting data and focus matcher/calibration work on missing-address/common-name ambiguity and overconfident partial-address merges; (2) vectorize remaining feature work and reduce sparse posting traversal, benchmarking scoring/output and France-like address lengths; (3) improve residual cross-script/numeric/partial-address retrieval using fitting-only evidence. Use a fresh confirmation set for any new selections.
Stopped after verified local validation and this report. No challenge test inference or submission activity.
