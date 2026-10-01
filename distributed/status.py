import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / 'distributed'
with open(DIST / 'shard_manifest.tsv', encoding='utf-8', newline='') as f:
    shards = list(csv.DictReader(f, delimiter='\t'))
complete = []
for s in shards:
    path = DIST / f"results/shard_{s['shard_id']}/shard_manifest.json"
    if path.exists():
        m = json.loads(path.read_text(encoding='utf-8'))
        if m.get('status') == 'VERIFIED_COMPLETE' and m.get('source1_fingerprint') == s['source1_fingerprint']:
            complete.append(m)
done = {m['shard_id'] for m in complete}
seconds = sum(m['end_to_end_runtime'] for m in complete)
finished_s1 = sum(m['source1_count'] for m in complete)
remaining_s1 = sum(int(s['source1_count']) for s in shards if s['shard_id'] not in done)
print(json.dumps({'total_shards': len(shards), 'verified_complete_shards': len(done),
                  'unfinished_shards': [s['shard_id'] for s in shards if s['shard_id'] not in done],
                  'source1_complete': finished_s1, 'source1_remaining': remaining_s1,
                  'candidate_pairs_generated': sum(m['candidate_pair_count'] for m in complete),
                  'estimated_remaining_worker_hours': remaining_s1 / (finished_s1 / seconds) / 3600 if seconds and finished_s1 else None}, indent=2))
