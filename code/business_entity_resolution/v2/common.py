"""Isolated V2 runtime and immutable V1 reuse."""
import os,sys,json,time,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
V2=ROOT/'artifacts/v2';V1=ROOT/'artifacts/v1'
sys.path.insert(0,str(V2/'packages'))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','4')
from core import *
from prepare import save,memory
import numpy as np
import psutil
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def connection(src):
    import sqlite3
    return sqlite3.connect(f'file:{(V1/f"s{src}.sqlite").as_posix()}?mode=ro',uri=True)
def audit():
    import importlib.metadata as md, platform
    pkgs={d.metadata['Name']:d.version for d in md.distributions(path=[str(V2/'packages')])}
    licenses={}
    for d in md.distributions(path=[str(V2/'packages')]):
        licenses[d.metadata['Name']]={'version':d.version,'license':d.metadata.get('License-Expression') or d.metadata.get('License'),
          'license_files':[str(p) for p in d.files or [] if 'license' in str(p).lower() or 'copying' in str(p).lower()]}
    save(V2/'environment.json',{'packages':pkgs,'python':sys.version,'platform':platform.platform(),'ram':dict(psutil.virtual_memory()._asdict()),'disk':dict(psutil.disk_usage(str(ROOT))._asdict()),'licenses':licenses})
    (Path(__file__).parent/'requirements.txt').write_text('\n'.join(f'{k}=={v}' for k,v in sorted(pkgs.items()))+'\n')
    inventory=[{'path':str(p.relative_to(ROOT)),'bytes':p.stat().st_size} for p in V1.rglob('*') if p.is_file()]
    save(V2/'v1_inventory.json',inventory)
    manifests=read(V1/'run_6000/artifact_manifest.json')
    for name,v in manifests.items():assert digest(V1/'run_6000'/name)==v['sha256'],name
    save(V2/'v1_preservation.json',{'critical_artifacts':manifests,'code':read(V1/'run_6000/code_fingerprints.json'),'report_sha256':digest(ROOT/'baseline_validation_report.md')})
    print(json.dumps({'packages':pkgs,'available_gib':psutil.virtual_memory().available/2**30,'v1_files':len(inventory)}),flush=True)
if __name__=='__main__':audit()
