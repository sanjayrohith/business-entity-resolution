# Recovery experiment log

Started 2026-09-27 05:07 UTC. User reported first public leaderboard F0.5 **0.917593**, about 13 hours remaining, and four submissions left. The public score is recorded only as an outcome; it was not used as a training label or feature. All experiments use supplied training/test files and existing pinned dependencies.

## Inputs and validation

Frozen first submission: `submissions/deadline_v1/`, LightGBM threshold 0.642, fast R10 route profile. Its 77,700,922 test candidate pairs and 5,403,921 predictions are preserved. The test run retained candidate IDs and final decisions but deleted per-pair scores and features.

Fresh recovery sample: 10,000 training S1, excluding V1/V2 sampled IDs and sampled suffix-name/address grouping keys. Grouped by suffix-name or reordered address; 6,000 development and 4,000 confirmation entities. Training S2/S3 indexes are hard-linked read-only under `distributed/recovery/train_index/` to reuse identical bytes. The old V2 10,000-S1 sample is used for diagnostics only.

Fresh R10 baseline: development F0.5 **0.934372**, precision **0.965190**, recall **0.869695**, candidate recall **0.968978**; confirmation F0.5 **0.936306**, precision **0.966472**, recall **0.873340**, candidate recall **0.973170**. The old 2,500-S1 confirmation R10 candidate oracle is **0.987013**; R10 cannot plausibly bridge the top-100 gap by selection alone.

## Candidate model families

| Experiment | Inputs | Tune F0.5 | Fresh confirmation F0.5 | Decision |
|---|---|---:|---:|---|
| R10 base | Frozen LightGBM score >=0.642 | 0.933812 | 0.936306 | Reference |
| Score-context meta | Base score, 41 frozen pair features, S1 score ranks/gaps/counts/source context | 0.938583 | **0.943751** | Selected for Submission #2 |
| Route-free cheap | Reconstructable frozen text features plus base decision bit | 0.937511 | 0.939963 | Rejected: smaller gain, measured ~9h single-machine four-worker test runtime and tight RAM |
| Compact normalized text | Fast name/address/numeric text features plus base decision bit | 0.935727 | 0.937443 | Rejected: gain too small |

The score-context model used 4,045 fresh development S1 for fitting and 1,955 for threshold/early stopping. Its threshold **0.7** was selected on that development tune subset before fresh confirmation evaluation. On 4,000 fresh confirmation S1 it yields macro precision **0.965374** and recall **0.898631**. Paired bootstrap 95% interval for the F0.5 improvement over R10 base: **[+0.004308,+0.010595]**. US improvement +0.008446; India +0.005992. True singleton F0.5 improves +0.059361; the true-one-match slice declines by 0.025601. France has no supplied labels, so these figures do not estimate French performance.

The score-aware meta model cannot be applied directly to the first test output because base pair scores/features were deleted. A full R10 replay is authorized by the user and justified by the paired fresh-validation gain. It stores frozen base scores and meta feature columns by checkpoint, checks that each candidate shard is byte-for-byte identical to the first submission, and writes only under `submissions/recovery_02/` and `distributed/recovery/full_r10/`. It does not modify the first submitted output or model.

## Structural findings

See `leaderboard_recovery_structural_audit.md`. All 7,638,365 training target links have unique S1 ownership. The first test prediction graph has 48,356 conflicted target IDs and 90,284 excess ownership links. Target exclusivity is not yet applied because an ownership-ranking policy must be validated on enriched ambiguous training cases; a random 10,000-S1 sample has essentially no competing target predictions.

No retrieval expansion to R20/R30/R50 has been launched. Such a run requires a measured gain and enough remaining time for validation and manual upload.

## Repeated-name ownership stress test

A second train-only sample of 10,004 S1 entities was grouped by repeated raw business name (1,267 groups), with 6,003 development and 4,001 held-out confirmation entities. Exact R10 retrieval on the complete training S2/S3 set gave candidate recall 0.921883/0.929611 and frozen baseline F0.5 0.912573/0.916192. Applying the already-fitted score-context meta model gave 0.914737/0.920060. There were 58/25 target IDs selected for multiple sampled S1 entities in the two folds. Enforcing one owner per selected target, chosen by highest meta probability with a 0.05 lead requirement, increased F0.5 to 0.917258/0.922066. The absolute confirmation gain over unrestrained meta is +0.002006, with paired bootstrap 95% interval [+0.001021,+0.003032]; training ground truth target ownership is 100% unique. This supports a distinct ownership-constrained Submission #3 after Submission #2 is validated. The sample includes only selected S1 entities, so this is limited evidence for full-test conflict resolution.

## Targeted rescue timing and oracle

Using the existing frozen transliteration and missing-address rescue indexes against all 10,000 fresh training queries took 137.6 seconds for both sources. The two targeted routes recovered 184 additional true links on 6,000 development S1 and 135 on 4,000 confirmation S1, raising total R10 candidate recall from 0.968978 to 0.977828 and from 0.973170 to 0.982933 respectively. The missing-address route provided almost all of the gain; transliteration added only 0/7 development and 3/5 confirmation links for S2/S3. These are candidate-oracle gains, not measured F0.5 gains. A naive full-test extrapolation is at least 6.6 hours just for these routes against the smaller training corpus, excluding feature generation and validation. Do not launch a full test targeted-retrieval pass under this deadline without a measured selective route and scoring plan.

Cheap query-selection signals do not concentrate the rescue cases: on held-out confirmation, selecting the 37.5% of S1 with fewer than three frozen R10 predicted links captures only 43.0% of rescued links; selecting 62.5% with fewer than four captures 67.4%. Selecting S1 whose maximum frozen score is below 0.99 captures 12.6% of rescues at 17.2% of queries. Rescued links often occur beside already strong matches, so a selective test pass would sacrifice much of the small measured recall gain. No full-test targeted pass is justified.

The recovery #2 replay runs in a persistent background session with per-10,000-S1 verified checkpoints. A separate finalizer waits for its atomic merge, performs an independent full-row scan and 18 organizer-validator chunks, and writes the PASS manifest. After that manifest appears, a second background watcher starts the ownership alternative, which will be saved and validated independently as recovery #3. No upload is automated.

An ownership scoring smoke test on the first ten recovery checkpoints found 818 conflicted target IDs and recomputed all owner meta scores from the retained candidate features. Restricted-S1 context scores were bit-identical to full-batch context on a separate checkpoint smoke. The preselected 0.05 margin assigned 347 of those 818 conflict targets to a clear owner and would leave the rest unassigned. The full #3 decision remains subject to complete output validation.

## Final cardinality check

After both recovery submissions passed validation, a small adaptive threshold policy was tuned on the original development tune groups. It used only the existing meta predicted count (<=1 versus >1), selected thresholds 0.50/0.75, and moved tune F0.5 from 0.938583 to 0.938894. It **reduced** held-out confirmation F0.5 from 0.943751 to 0.942589 (paired delta -0.001162, 95% interval [-0.003302,+0.001113]). Rejected. No Submission #4 was created because this is neither an improvement nor a materially distinct validated strategy.

## Final recovery #4 cached-score search

After public results #1/#2/#3 of 0.917593/0.921733/0.930288 were reported, the complete post-ownership decoder was retuned using only retained scores. A 119-point global grid covered meta thresholds 0.40–0.76 and ownership margins 0–0.10 on both the fresh grouped sample and repeated-name ambiguity sample. Mutual top-1/top-3 rules strongly reduced F0.5 and were rejected. A 25-point S2/S3 threshold grid also failed held-out robustness. A narrow 95-point refinement covered threshold 0.620–0.710 and margins 0.08–0.20, plus 30 conservative exact-name rescue rules.

Recovery #3 baseline (threshold 0.70, margin 0.05) remains fresh confirmation F0.5 0.943751 and ambiguity confirmation 0.922066. The best global confirmation-consistent alternative, threshold 0.665 and margin 0.12, changed fresh confirmation by only +0.000362 and ambiguity confirmation by +0.000727. The best exact-name rescue changed confirmations by +0.000431/+0.001055 but regressed fresh development by -0.000604. The development-selected threshold 0.64/margin 0.10 was flat on fresh confirmation (-0.000009) and +0.001131 on ambiguity confirmation. None reaches the required approximately 0.0015–0.002 robust gain across both independent slices. No recovery #4 output was generated, preserving the remaining leaderboard slot. Machine-readable results: `recovery04_grid.json` and `recovery04_refine.json`.

## 2026-09-27 targeted retrieval recovery decision

- Hard set: 346,509 / 1,732,544 test S1 (20.0000%); includes all 259,452 France S1 plus ranked uncertainty cases.
- Retrieval benchmark: targeted transliteration + missing-name routes. R20 recovered 68/73 validation-hard links recoverable by R50 while adding 158,071 pairs for 2,000 hard validation S1. R20 test benchmark was 69.141 seconds for 5,000 hard S1; projected retrieval was 79.86 minutes for the full hard set.
- Exact validation: existing recovery_03 pair features and scores were frozen; only 158,071 new R20 pairs were featured/scored; recovery_03 threshold 0.7 and ownership margin 0.05 were applied globally.
- Development macro F0.5: 0.948646571 -> 0.948693385 (+0.000046814).
- Confirmation macro F0.5: 0.943751401 -> 0.943683171 (-0.000068230).
- Hard development F0.5: 0.874167989 -> 0.874400318 (+0.000232329).
- Hard confirmation F0.5: 0.875911872 -> 0.875566840 (-0.000345032).
- Confirmation candidate recall: baseline R10 0.973170380 overall; augmented 0.975195256 (+0.002024876). The decoder converted this into only 5 additional true positives and 8 additional predicted links on confirmation, reducing F0.5.
- Decision: STOP. The required robust >=0.002 F0.5 gain was not demonstrated. No full test targeted retrieval was started and no recovery_04 submission was created. recovery_03 remains the upload candidate.
