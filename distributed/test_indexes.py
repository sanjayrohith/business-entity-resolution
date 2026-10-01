"""Build official test indexes with the frozen V2 functions and test-only paths."""
import argparse
import csv
import hashlib
import json
import sqlite3
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / 'distributed'
INDEX = DIST / 'shared_test_indexes'
V2CODE = ROOT / 'code/business_entity_resolution/v2'
sys.path.insert(0, str(V2CODE))


def configure():
    import common
    import sparse_retrieval
    import hashed_index
    import targeted
    common.V2 = INDEX
    common.V1 = INDEX
    sparse_retrieval.V2 = INDEX
    sparse_retrieval.V1 = INDEX
    sparse_retrieval.connection = common.connection
    hashed_index.V2 = INDEX
    targeted.V2 = INDEX
    return common, sparse_retrieval, hashed_index, targeted


def build_db(src, common):
    from core import VERSION, norm, fold, suffix, rows
    source = ROOT / f'dataset/test/test_source{src}.tsv'
    db = INDEX / f's{src}.sqlite'
    manifest = INDEX / f's{src}.json'
    fingerprint = {'size': source.stat().st_size, 'sha256': common.digest(source), 'normalization': VERSION, 'limit': 0}
    if manifest.exists():
        old = common.read(manifest)
        if old['fingerprint'] != fingerprint:
            raise ValueError(f'Test S{src} SQLite fingerprint mismatch')
        if old.get('complete'):
            return
    start = time.perf_counter()
    con = sqlite3.connect(db)
    con.executescript('PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL; PRAGMA cache_size=-65536; PRAGMA temp_store=FILE;')
    con.execute('CREATE TABLE IF NOT EXISTS r(eid TEXT, business_name TEXT,business_address TEXT,country TEXT,n TEXT,a TEXT,s TEXT,ak TEXT)')
    done = con.execute('SELECT count(*) FROM r').fetchone()[0]
    batch = []
    for i, row in enumerate(rows(source), 1):
        if i <= done:
            continue
        n, a = fold(norm(row['business_name'])), fold(norm(row['business_address']))
        batch.append((i, row['entity_id'], row['business_name'], row['business_address'], row['country'], n, a, suffix(n), ' '.join(sorted(set(a.split())))))
        if len(batch) == 25000:
            with con:
                con.executemany('INSERT INTO r(rowid,eid,business_name,business_address,country,n,a,s,ak) VALUES(?,?,?,?,?,?,?,?,?)', batch)
            batch.clear()
            print('normalized', src, i, round(time.perf_counter()-start, 1), flush=True)
    if batch:
        with con:
            con.executemany('INSERT INTO r(rowid,eid,business_name,business_address,country,n,a,s,ak) VALUES(?,?,?,?,?,?,?,?,?)', batch)
    for field in ('eid', 'n', 's', 'ak'):
        con.execute(f'CREATE INDEX IF NOT EXISTS idx_{field} ON r({field})')
        con.commit()
    count = con.execute('SELECT count(*) FROM r').fetchone()[0]
    con.execute('PRAGMA wal_checkpoint(TRUNCATE)')
    con.close()
    common.save(manifest, {'fingerprint': fingerprint, 'complete': True, 'rows': count,
                           'seconds': time.perf_counter()-start, 'bytes': db.stat().st_size})


def build(src):
    INDEX.mkdir(parents=True, exist_ok=True)
    common, sparse, hashed, targeted = configure()
    portable_path = INDEX / 'portable_manifest.json'
    if portable_path.exists():
        portable = common.read(portable_path)
        source = ROOT / f'dataset/test/test_source{src}.tsv'
        expected = portable['sources'][str(src)]['dataset_fingerprint']
        if source.stat().st_size != expected['size'] or common.digest(source) != expected['sha256']:
            raise ValueError('Existing portable index fingerprint mismatch; refusing rebuild')
        return
    build_db(src, common)
    sparse.build(src)
    hashed_manifest = INDEX / f'index2_s{src}/manifest.json'
    if hashed_manifest.exists():
        existing = common.read(hashed_manifest)
        if existing['config']['source_fingerprint'] != common.read(INDEX / f's{src}.json')['fingerprint']:
            raise ValueError('Existing hashed index fingerprint mismatch; refusing rebuild')
    hashed.build(src)
    targeted.build(src)


def verify():
    common, _, _, _ = configure()
    from core import VERSION
    package = {'creation_version': 'distributed-test-v1; frozen V2 index builders',
               'normalization_version': VERSION, 'sources': {}, 'files': {}}
    for src in (2, 3):
        source = ROOT / f'dataset/test/test_source{src}.tsv'
        fm = common.read(INDEX / f's{src}.json')
        base = common.read(INDEX / f'index_s{src}/manifest.json')
        hashed = common.read(INDEX / f'index2_s{src}/manifest.json')
        target = common.read(INDEX / f'targeted_s{src}/complete.json')
        expected = {'size': source.stat().st_size, 'sha256': common.digest(source), 'normalization': VERSION, 'limit': 0}
        if not fm.get('complete') or fm['fingerprint'] != expected or base['config']['source_fingerprint'] != expected or hashed['config']['source_fingerprint'] != expected:
            raise ValueError(f'Test S{src} index fingerprint mismatch')
        if not target.get('nonlatin_records') or not target.get('missing_records'):
            raise ValueError('Targeted index incomplete')
        package['sources'][str(src)] = {'dataset_fingerprint': expected,
                                       'retrieval_config': hashed['config'],
                                       'normalization_config': base['config'],
                                       'rows': fm['rows']}
    encode = lambda x: hashlib.sha256(json.dumps(x, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    package['retrieval_config_fingerprint'] = encode([package['sources'][str(s)]['retrieval_config'] for s in (2, 3)])
    package['normalization_config_fingerprint'] = encode([package['sources'][str(s)]['normalization_config'] for s in (2, 3)])
    package['transliteration_config_fingerprint'] = encode({'library': 'anyascii', 'version': '0.3.3', 'routes': ['translit_char', 'missing_name_char', 'joint_name_address']})
    for path in INDEX.rglob('*'):
        if path.is_file() and path.name != 'portable_manifest.json' and 'binary' not in path.parts and not path.name.endswith(('.tmp', '-wal', '-shm')):
            rel = path.relative_to(INDEX).as_posix()
            package['files'][rel] = {'bytes': path.stat().st_size, 'sha256': common.digest(path)}
    existing = INDEX / 'portable_manifest.json'
    if existing.exists() and common.read(existing) != package:
        raise ValueError('Existing portable index manifest differs; refusing replacement')
    common.save(existing, package)
    print('Portable index manifest:', len(package['files']), 'files', flush=True)


def archive_intermediates():
    if not (INDEX / 'portable_manifest.json').exists():
        raise ValueError('Verify indexes before archiving build intermediates')
    archive = DIST / 'index_build_intermediates'
    archive.mkdir(parents=True, exist_ok=True)
    for src in (2, 3):
        original = INDEX / f'index2_s{src}/binary'
        destination = archive / f'index2_s{src}_binary'
        if original.exists():
            if destination.exists() or not original.resolve().is_relative_to(ROOT.resolve()) or not destination.resolve().is_relative_to(ROOT.resolve()):
                raise ValueError('Unsafe or conflicting intermediate move')
            original.replace(destination)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['build', 'verify', 'archive-intermediates'])
    p.add_argument('--source', type=int, choices=[2, 3])
    a = p.parse_args()
    if a.action == 'build':
        if not a.source:
            p.error('--source is required for build')
        build(a.source)
    elif a.action == 'verify':
        verify()
    else:
        archive_intermediates()
