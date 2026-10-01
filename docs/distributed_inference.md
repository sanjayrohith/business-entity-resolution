# Amazon ML Challenge 2026 distributed inference

The 160 shards partition **only official test Source 1**. Every laptop requires the full official test Source 2 and Source 3 files and the same complete test indexes. The adapter calls the unchanged frozen V2 retrieval and 80-feature functions, scores the frozen LightGBM model, and uses the frozen 0.642 threshold. Full official inference has not been started. The shared test indexes are built and passed a same-host relocation check. The 100-S1 equivalence test passed bitwise for candidates, features, scores, decisions and final IDs. A separate five-S1 worker smoke test passed atomic shard finalization and checksum verification, then removed its temporary result.

## Status and preparation

Read `distributed/distributed_audit.md` and `distributed/shard_verification.json`. Before worker launch, both official test target indexes must have a verified `distributed/shared_test_indexes/portable_manifest.json`. The build commands, if an index is missing, are:

```powershell
& .\.venv\Scripts\python.exe distributed\test_indexes.py build --source 2
& .\.venv\Scripts\python.exe distributed\test_indexes.py build --source 3
& .\.venv\Scripts\python.exe distributed\test_indexes.py verify
& .\.venv\Scripts\python.exe distributed\test_indexes.py archive-intermediates
```

The builder refuses a mismatched completed normalized cache. An index is portable only after a relocated-path smoke test and matching SHA256 checks. Copy the complete `distributed/shared_test_indexes/` directory. The archive command moves unneeded binary build intermediates to `distributed/index_build_intermediates/`, which is not copied to workers.

The worker tree is in `distributed_worker_package/`; `migration_manifest.json` gives hashes and exact files. Each laptop receives the **contents** of `distributed_worker_package/` as its project root, plus `dataset/test/` and `distributed/shared_test_indexes/` in those exact relative locations. Keep the index manifest and all listed files together. Copy size is about **17.61 GiB per laptop**: 181 MiB worker tree, 16.32 GiB indexes, and 1.12 GiB dataset. Reserve at least **30 GiB free** before copying so the local `.venv`, temporary stage files, and result shards fit. Do not copy `.venv` between laptops. Python 3.12 is required. Initial internet access is needed if the exact pinned V2 wheels are not already in pip's local cache.

Use the same copy list on Laptop 1, Laptop 2, and Laptop 3:

1. Copy every file and folder **inside** `distributed_worker_package/` into an empty project folder.
2. Copy the complete `dataset/test/` into that project's `dataset/test/`.
3. Copy the complete `distributed/shared_test_indexes/` into that project's `distributed/shared_test_indexes/`.

## Worker commands

Open PowerShell in the copied project root. The scripts bootstrap a project-local `.venv`, verify every dependency version, fingerprint the model/indexes/data, detect CPU/RAM/disk/GPU, and process one shard at a time. GPU presence is reported but the frozen CPU sparse pipeline is used. About 2 GiB of free RAM is reserved when choosing native thread counts.

Laptop 1:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\run_worker.ps1 -WorkerName Laptop1 -WorkerIndex 0 -WorkerCount 3
```

Laptop 2:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\run_worker.ps1 -WorkerName Laptop2 -WorkerIndex 1 -WorkerCount 3
```

Laptop 3:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\run_worker.ps1 -WorkerName Laptop3 -WorkerIndex 2 -WorkerCount 3
```

Only after the mandatory equivalence and benchmark reports show PASS should those full commands be used. For a controlled single shard, use `-ShardIds '0000' -MaxShards 1`. A faster laptop can take any unfinished IDs shown by status, such as `-ShardIds '0004,0007'`; results already marked `VERIFIED_COMPLETE` are skipped. Never run the same unfinished shard on two machines at once.

The final 500-S1 benchmark on this 16 GB laptop measured 52.90 s retrieval, 27.00 s features with four child processes, 4.71 s scoring, 0.24 s merge serialization, 5.89 S1/s and 3,179 candidate pairs/s, with a conservative 1.41 GB process-sum peak. Stage-based planning estimates are **54 hours on one similar laptop; 29 hours on two; 20.5 hours on three; 16.3 hours on four**. For three comparable laptops, plan a **17–31 hour** wall-clock range after copying/bootstrap. These are estimates, not measured full-shard speeds. Slower laptops or reduced feature-worker counts may take longer. Give more unfinished shards to faster laptops based on observed S1/hour.

Copy each laptop's `distributed/results/shard_XXXX/` directories to the merger laptop under its `distributed/results/`. Copy only completed shard directories; do not copy `.working/`.

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\status_distributed.ps1
powershell -NoProfile -ExecutionPolicy Bypass -File .\merge_distributed_results.ps1
```

The merger requires all 160 verified shards, checks hashes, full S1 coverage, valid S2/S3 IDs and candidate membership using streaming rows, then runs `utils/validate_submission.py --check-ids` on the matching file. The organizer helper materializes the entire candidate mapping and may exceed 16 GiB on the official test, so the merger gives it an absent candidate path and performs stricter independent candidate checks. A PASS is required before calling the output complete.
