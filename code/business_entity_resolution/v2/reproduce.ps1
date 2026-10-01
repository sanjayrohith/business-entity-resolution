param([ValidateSet('Prepare','Pilot','Validation','Benchmark','Verify')][string]$Stage='Verify',[string]$Python='python')
$ErrorActionPreference='Stop'
# Python 3.12. Pass -Python <path> to use a specific interpreter.
$v2Python=$Python
$v2Root=(Resolve-Path (Join-Path $PSScriptRoot '../../..')).Path
Set-Location -LiteralPath $v2Root
$env:OPENBLAS_NUM_THREADS='1'
$env:PYTHONIOENCODING='utf-8'
function Run-V2 { & $v2Python @args; if ($LASTEXITCODE -ne 0) { throw "V2 command failed: $args" } }
if ($Stage -eq 'Prepare') {
    Run-V2 code/business_entity_resolution/v2/common.py
    Run-V2 code/business_entity_resolution/v2/split.py
    Run-V2 code/business_entity_resolution/v2/profile_v1.py
    Run-V2 code/business_entity_resolution/v2/profile_routes.py
    foreach ($source in 2,3) {
        Run-V2 code/business_entity_resolution/v2/sparse_retrieval.py build --source $source
        Run-V2 code/business_entity_resolution/v2/hashed_index.py --source $source
        Run-V2 code/business_entity_resolution/v2/targeted.py --source $source
    }
}
if ($Stage -eq 'Pilot') {
    Run-V2 code/business_entity_resolution/v2/compare_v1.py
    Run-V2 code/business_entity_resolution/v2/pilot.py --n 500 --revision _frozen --policy tiered
    Run-V2 code/business_entity_resolution/v2/pilot.py --old --n 500 --revision _frozen --policy tiered
}
if ($Stage -eq 'Validation') {
    foreach ($source in 2,3) { Run-V2 code/business_entity_resolution/v2/sparse_retrieval.py retrieve --source $source --k 100 --policy tiered }
    Run-V2 code/business_entity_resolution/v2/run_features.py
    Run-V2 code/business_entity_resolution/v2/evaluate.py train
    Run-V2 code/business_entity_resolution/v2/evaluate.py confirm
}
if ($Stage -eq 'Verify') {
    Run-V2 code/business_entity_resolution/v2/verify_v2.py
    Run-V2 code/business_entity_resolution/v2/report_v2.py
}
if ($Stage -eq 'Benchmark') {
    Run-V2 code/business_entity_resolution/v2/benchmark_parallel.py
    Run-V2 code/business_entity_resolution/v2/benchmark_features.py
}
