"""Streaming final checks, chunked organizer validation, and frozen submission."""
import csv
import hashlib
import importlib.metadata as metadata
import json
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
from itertools import zip_longest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'artifacts/v2/packages'))
sys.path.insert(0, str(ROOT / 'distributed'))
from worker import preflight, file_hash

OUT = ROOT / 'output'
MATCHING = OUT / 'matching_results.tsv'
CANDIDATE = OUT / 'candidate_pairs.tsv'
SOURCE1 = ROOT / 'dataset/test/test_source1.tsv'
TEST = ROOT / 'dataset/test'
WORK = ROOT / 'distributed/deadline/validation_work'
FROZEN = ROOT / 'submissions/deadline_v1'
EXPECTED = 1_732_544


def parse_row(line, name):
    parts = line.rstrip('\r\n').split('\t')
    if len(parts) != 2:
        raise ValueError(f'Malformed {name} row')
    return parts[0], parts[1].split(',') if parts[1] else []


def independent_scan():
    targets = set()
    for src in (2, 3):
        with open(TEST / f'test_source{src}.tsv', encoding='utf-8') as f:
            if f.readline().rstrip('\r\n') != 'entity_id\tbusiness_name\tbusiness_address\tcountry':
                raise ValueError('Official target header mismatch')
            targets.update(line.split('\t', 1)[0] for line in f)
    seen = set()
    country = defaultdict(Counter)
    distributions = defaultdict(Counter)
    n = pairs = links = 0
    with open(SOURCE1, encoding='utf-8') as sf, open(MATCHING, encoding='utf-8') as mf, open(CANDIDATE, encoding='utf-8') as cf:
        if sf.readline().rstrip('\r\n') != 'entity_id\tbusiness_name\tbusiness_address\tcountry':
            raise ValueError('Official S1 header mismatch')
        if mf.readline().rstrip('\r\n') != 'source1_entity_id\tmatched_entity_ids':
            raise ValueError('Matching header mismatch')
        if cf.readline().rstrip('\r\n') != 'source1_entity_id\tcandidate_entity_ids':
            raise ValueError('Candidate header mismatch')
        for sline, mline, cline in zip_longest(sf, mf, cf):
            if sline is None or mline is None or cline is None:
                raise ValueError('Output row count mismatch')
            source = sline.rstrip('\r\n').split('\t')
            if len(source) != 4:
                raise ValueError('Official S1 row schema mismatch')
            sid, matched = parse_row(mline, 'matching')
            cid, candidates = parse_row(cline, 'candidate')
            if sid != source[0] or cid != sid or sid in seen:
                raise ValueError('S1 output alignment or duplicate mismatch')
            if len(matched) != len(set(matched)) or len(candidates) != len(set(candidates)):
                raise ValueError('Duplicate target inside list')
            if not set(matched) <= set(candidates):
                raise ValueError('Prediction outside scored candidates')
            if any(target not in targets for target in candidates):
                raise ValueError('Candidate ID absent from official test S2/S3')
            seen.add(sid)
            n += 1
            pairs += len(candidates)
            links += len(matched)
            label = source[3]
            country[label]['entities'] += 1
            country[label]['predicted_links'] += len(matched)
            country[label]['candidate_pairs'] += len(candidates)
            distributions[label][str(min(5, len(matched)))] += 1
    if n != EXPECTED or len(seen) != EXPECTED:
        raise ValueError(f'Official S1 coverage mismatch: {n}')
    del targets, seen
    return {'source1_rows': n, 'candidate_pairs': pairs, 'predicted_links': links,
            'average_predicted_links': links / n,
            'countries': {k: {**v, 'average_predicted_links': v['predicted_links'] / v['entities'],
                              'prediction_cardinality_0_1_2_3_4_5plus': distributions[k]}
                          for k, v in country.items()}}


def organizer_chunks(chunk_size=100000):
    WORK.mkdir(parents=True, exist_ok=True)
    test_dir = WORK / 'test'
    test_dir.mkdir(exist_ok=True)
    for src in (2, 3):
        destination = test_dir / f'test_source{src}.tsv'
        source = TEST / f'test_source{src}.tsv'
        if not destination.exists():
            shutil.copy2(source, destination)
        if destination.stat().st_size != source.stat().st_size:
            raise ValueError('Validator target copy size mismatch')
    logs = []
    with open(SOURCE1, encoding='utf-8') as sf, open(MATCHING, encoding='utf-8') as mf, open(CANDIDATE, encoding='utf-8') as cf:
        s_header, m_header, c_header = sf.readline(), mf.readline(), cf.readline()
        chunk = 0
        while True:
            s_path = test_dir / 'test_source1.tsv'
            m_path = WORK / 'matching_results.tsv'
            c_path = WORK / 'candidate_pairs.tsv'
            count = 0
            with open(s_path, 'w', encoding='utf-8', newline='') as so, open(m_path, 'w', encoding='utf-8', newline='') as mo, open(c_path, 'w', encoding='utf-8', newline='') as co:
                so.write(s_header); mo.write(m_header); co.write(c_header)
                for _ in range(chunk_size):
                    sline, mline, cline = sf.readline(), mf.readline(), cf.readline()
                    if not sline:
                        if mline or cline:
                            raise ValueError('Validator chunk row mismatch')
                        break
                    if not mline or not cline:
                        raise ValueError('Validator chunk output missing row')
                    so.write(sline); mo.write(mline); co.write(cline)
                    count += 1
            if not count:
                break
            command = [sys.executable, str(ROOT / 'utils/validate_submission.py'),
                       '--matching', str(m_path), '--candidate', str(c_path),
                       '--test-dir', str(test_dir), '--check-ids']
            result = subprocess.run(command, text=True, capture_output=True,
                                    encoding='cp1252' if sys.platform == 'win32' else 'utf-8',
                                    errors='replace')
            log = WORK / f'organizer_chunk_{chunk:03d}.log'
            log.write_text((result.stdout or '') + '\n' + (result.stderr or ''), encoding='utf-8')
            if result.returncode or 'PASS — no blocking issues found' not in (result.stdout or ''):
                raise ValueError(f'Organizer validator failed on chunk {chunk}: {log}')
            logs.append({'chunk': chunk, 'rows': count, 'log': str(log.relative_to(ROOT)), 'status': 'PASS'})
            print('ORGANIZER_PASS', chunk, count, flush=True)
            chunk += 1
    if sum(item['rows'] for item in logs) != EXPECTED:
        raise ValueError('Chunked organizer validator coverage incomplete')
    return logs


def main():
    started = time.perf_counter()
    pre = preflight(verify_files=False)
    summary = independent_scan()
    print('INDEPENDENT_PASS', json.dumps(summary), flush=True)
    logs = organizer_chunks()
    FROZEN.mkdir(parents=True, exist_ok=True)
    shutil.copy2(MATCHING, FROZEN / 'matching_results.tsv')
    shutil.copy2(CANDIDATE, FROZEN / 'candidate_pairs.tsv')
    for path in (MATCHING, CANDIDATE):
        if file_hash(path) != file_hash(FROZEN / path.name):
            raise ValueError('Frozen submission copy checksum mismatch')
    requirements = (ROOT / 'code/business_entity_resolution/v2/requirements.txt').read_text().splitlines()
    packages = {line.split('==')[0]: metadata.version(line.split('==')[0]) for line in requirements if line.strip()}
    complete = list((ROOT / 'distributed/deadline/checkpoints').glob('batch_*/complete.json'))
    compute_seconds = sum(json.loads(path.read_text())['elapsed_seconds'] for path in complete)
    manifest = {'status': 'PASS', 'profile': 'V2_FAST_DEADLINE_R10', 'model': 'LightGBM',
                'local_confirmation_macro_f0_5': 0.9338127274081095,
                'threshold': 0.642, 'routes': ['name_char', 'address_word', 'joint_name_address',
                                               'exact_name', 'suffix_name', 'address_key'],
                'max_route_rank': 10, 'preflight': pre, 'summary': summary,
                'organizer_validator': {'mode': '18 exhaustive chunks with full official target files and --check-ids',
                                        'chunks': logs},
                'runtime': {'inference_checkpoint_seconds_sum': compute_seconds,
                            'finalization_seconds': time.perf_counter() - started},
                'package_versions': packages,
                'sha256': {'matching_results.tsv': file_hash(MATCHING),
                           'candidate_pairs.tsv': file_hash(CANDIDATE)}}
    path = FROZEN / 'submission_manifest.json'
    path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    (ROOT / 'submission_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print('SUBMISSION_READY', json.dumps({'summary': summary, 'sha256': manifest['sha256'],
                                          'manifest': str(path)}), flush=True)


if __name__ == '__main__':
    main()
