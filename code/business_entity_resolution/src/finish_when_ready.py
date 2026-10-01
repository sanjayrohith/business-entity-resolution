"""Finish this same local experiment after its already-started retrieval workers exit."""
import subprocess
import sys
import time
import json
from pathlib import Path
from prepare import CACHE

if __name__=='__main__':
    p=CACHE/'run_6000/retrieval_walltime.json'
    while not p.exists():time.sleep(10)
    result=json.loads(p.read_text(encoding='utf-8'))
    if any(result['exit_codes']):raise SystemExit('Retrieval failed; no model stages started')
    src=Path(__file__).parent
    commands=[['experiment.py','diagnose'],['experiment.py','fit'],['experiment.py','final'],['verify.py'],['report.py']]
    for cmd in commands:
        args=[sys.executable,str(src/cmd[0]),*cmd[1:],'--n','6000']
        print('RUN',args,flush=True)
        subprocess.run(args,check=True)
