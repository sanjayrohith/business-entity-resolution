param([ValidateSet('Smoke','Full','Evaluate')][string]$Stage='Smoke',[int]$Workers=8,[string]$Python='python')
$ErrorActionPreference='Stop'
# Python 3.12 with SQLite FTS5 + trigram tokenizer. Pass -Python <path> to use a specific interpreter.
$baselinePython=$Python
$baselineRoot=(Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
Set-Location -LiteralPath $baselineRoot
$env:OPENBLAS_NUM_THREADS='1'
$env:PYTHONIOENCODING='utf-8'
function Run-Python { & $baselinePython @args; if ($LASTEXITCODE -ne 0) { throw "Python failed with exit $LASTEXITCODE" } }
Run-Python -m unittest discover -s code/business_entity_resolution/src -v
if ($Stage -eq 'Smoke') {
    Run-Python code/business_entity_resolution/src/prepare.py split --n 6000
    Run-Python code/business_entity_resolution/src/prepare.py index --source 2 --limit 100000
    Run-Python code/business_entity_resolution/src/retrieval.py --n 6000 --smoke-queries 10 --smoke-index 100000
    exit
}
if ($Stage -eq 'Full') {
    Run-Python code/business_entity_resolution/src/prepare.py audit
    Run-Python code/business_entity_resolution/src/prepare.py split --n 6000
    Run-Python code/business_entity_resolution/src/prepare.py index --source 2
    Run-Python code/business_entity_resolution/src/prepare.py index --source 3
    Run-Python code/business_entity_resolution/src/run_retrieval.py --n 6000 --workers $Workers
}
Run-Python code/business_entity_resolution/src/experiment.py diagnose --n 6000
Run-Python code/business_entity_resolution/src/experiment.py fit --n 6000
Run-Python code/business_entity_resolution/src/experiment.py final --n 6000
Run-Python code/business_entity_resolution/src/verify.py --n 6000
Run-Python code/business_entity_resolution/src/report.py --n 6000
