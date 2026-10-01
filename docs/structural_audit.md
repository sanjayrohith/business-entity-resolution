# Leaderboard recovery structural audit

Time: 2026-09-27 UTC. All measurements use only supplied competition data. The first submission and frozen V2 artifacts were left unchanged.

## Training ground truth

All **7,638,365** positive S2/S3 target IDs occur under exactly one S1. **Zero** occur under multiple S1 entities. This was measured by streaming every training ground-truth row; it supports target-side exclusivity as a structural constraint, subject to validation of the assignment policy.

Training S1 cardinality (5 means 5+):

| Country/source | 0 | 1 | 2 | 3 | 4 | 5+ |
|---|---:|---:|---:|---:|---:|---:|
| US combined | 73,896 | 71,689 | 225,285 | 318,876 | 290,446 | 343,441 |
| India combined | 49,351 | 47,468 | 149,927 | 211,965 | 193,669 | 230,808 |
| US S2 | 172,744 | 474,144 | 391,281 | 199,890 | 71,172 | 14,402 |
| US S3 | 159,723 | 430,004 | 400,693 | 223,397 | 86,903 | 22,913 |
| India S2 | 115,001 | 314,964 | 261,498 | 134,067 | 47,906 | 9,752 |
| India S3 | 106,553 | 286,413 | 267,682 | 149,046 | 58,213 | 15,281 |

On 34,620 true target records linked to a fresh 10,000-S1 training sample, purity among multi-record normalized clusters was: exact name **96.23%** (162 impure clusters), suffix name **96.23%** (220 impure), reordered address key **99.88%** (6 impure), exact name plus address **100%** (897 sampled clusters). This is sample-based purity, not a full-corpus guarantee. Exact or suffix name and address keys are therefore evidence, not hard ownership rules.

Among 4,568 true links missed by the R10 pair decision in that sample, 3,620 had a >=0.9-scoring true link in the other source; 258 also had strong target-to-target name similarity (>=0.8 and at least 0.15 stronger than S1-to-weak-target). This identifies a measurable cross-source rescue opportunity, not yet a validated prediction rule.

## First submission and R10 ceiling

The submitted R10 output contains 5,403,921 predicted links and 5,313,637 distinct target IDs. **48,356** target IDs have multiple predicted S1 owners, giving 90,284 excess ownership links. The completed inference retained candidate IDs and final decisions, but discarded pair feature vectors, pair scores, and retrieval-route metadata. The frozen first submission remains at `submissions/deadline_v1/`.

Replaying exact R10 route membership on retained V2 validation artifacts gives candidate recall **0.966017** and candidate-oracle macro F0.5 **0.987013** for 2,500 prior confirmation S1. On a fresh grouped 10,000-S1 training sample, 6,000 development S1 have candidate recall **0.968978** and baseline macro F0.5 **0.934372**; 4,000 confirmation S1 have candidate recall **0.973170** and baseline macro F0.5 **0.936306**. The fresh sample excludes V1/V2 sampled IDs and sampled suffix-name/address keys; 9,781 groups were assigned intact, largest group size 3.

The 10,000-S1 replay has almost no target ownership conflicts, because conflicts are uncommon at this sample fraction. It cannot reliably estimate the effect of exclusivity on the full test graph. The public leaderboard F0.5 of **0.917593** is an experiment result, not a label or feature.

## Experiment implications

- Target ownership is structurally unique in training, but assigning a conflicted target requires a validated ranking or margin rule.
- R10 retrieval is below the 0.99 candidate-recall gate. Its oracle F0.5 is close to the stated top-100 boundary, so a substantially stronger result will likely require better retrieval as well as selection.
- The first test run cannot directly support a pair-score meta model because its scores and features were deleted. A score-context meta model improved fresh confirmation F0.5 from 0.936306 to 0.943751, so an exact R10 replay is producing Submission #2 while checking every candidate checkpoint against the first run. A separate repeated-name stress sample supports a target-ownership alternative for Submission #3.
- Existing targeted rescue routes raise fresh candidate recall from 0.973170 to 0.982933, but their measured runtime and the lack of a selective query trigger make a full extra test pass unsafe under this deadline.
- No France labels exist in training; local validation does not measure French performance directly.

Machine-readable counts: `distributed/recovery/structural_counts.json`, `cluster_audit.json`, `prediction_graph.json`, `cached_r10_report.json`, and `fresh_r10/assessment.json`.
