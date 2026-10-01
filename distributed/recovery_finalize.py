"""Independent and organizer validation of recovery Submission #2."""
import json
import sys
import time
from datetime import datetime,timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'distributed'))
import finalize_deadline as final
from worker import file_hash,preflight

RESULT=ROOT/'submissions/recovery_02'
final.MATCHING=RESULT/'matching_results.tsv'
final.CANDIDATE=RESULT/'candidate_pairs.tsv'
final.WORK=RESULT/'validation_work'


def main():
    start=time.monotonic()
    if not final.MATCHING.exists() or not final.CANDIDATE.exists():
        raise ValueError('Recovery files not yet merged')
    candidate_hash=file_hash(final.CANDIDATE)
    first_hash=file_hash(ROOT/'submissions/deadline_v1/candidate_pairs.tsv')
    if candidate_hash!=first_hash:
        raise ValueError('Recovery candidate file differs from first R10 candidate set')
    summary=final.independent_scan()
    print('INDEPENDENT_PASS',json.dumps(summary),flush=True)
    logs=final.organizer_chunks()
    report=json.loads((ROOT/'distributed/recovery/meta_r10_result.json').read_text(encoding='utf-8'))
    checkpoints=list((ROOT/'distributed/recovery/full_r10/checkpoints').glob('batch_*/complete.json'))
    if len(checkpoints)!=174:raise ValueError(f'Expected 174 checkpoints, got {len(checkpoints)}')
    total_seconds=sum(json.loads(path.read_text(encoding='utf-8'))['elapsed_seconds'] for path in checkpoints)
    manifest={'status':'PASS','profile':'V2_RECOVERY_META_R10','model':'LightGBM score-context meta model',
              'threshold':.7,'base_model_threshold':.642,'base_model_sha256':file_hash(ROOT/'artifacts/v2/run/lightgbm.txt'),
              'meta_model_sha256':file_hash(ROOT/'distributed/recovery/meta_r10_lightgbm.txt'),
              'fresh_confirmation':report['confirmation_meta'],
              'fresh_baseline':report['confirmation_baseline'],
              'candidate_recall':json.loads((ROOT/'distributed/recovery/fresh_r10/assessment.json').read_text(encoding='utf-8'))['confirmation']['candidate_recall'],
              'summary':summary,'candidate_set_unchanged_from_submission_1':True,
              'organizer_validator':{'mode':'18 exhaustive chunks using full official test S2/S3 and --check-ids',
                                     'chunks':logs},
              'inference_checkpoint_seconds_sum':total_seconds,
              'finalization_seconds':time.monotonic()-start,
              'created_utc':datetime.now(timezone.utc).isoformat(),
              'sha256':{'matching_results.tsv':file_hash(final.MATCHING),
                        'candidate_pairs.tsv':candidate_hash},
              'preflight':preflight(verify_files=False)}
    (RESULT/'submission_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print('SUBMISSION_2_READY',json.dumps({'matching_sha256':manifest['sha256']['matching_results.tsv'],
          'fresh_confirmation':report['confirmation_meta'],'candidate_recall':manifest['candidate_recall']}),flush=True)


if __name__=='__main__':main()
