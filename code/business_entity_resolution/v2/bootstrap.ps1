param([switch]$Install)
$ErrorActionPreference = 'Stop'
$root = (Resolve-Path (Join-Path $PSScriptRoot '../../..')).Path
$venv = Join-Path $root '.venv'
$exe = Join-Path $venv 'Scripts/python.exe'
$req = Join-Path $PSScriptRoot 'requirements.txt'
if (-not (Test-Path -LiteralPath $exe)) {
    $candidates = @()
    foreach ($name in @('python3.12','python')) {
        $cmd = Get-Command $name -ErrorAction SilentlyContinue
        if ($cmd) { $candidates += $cmd.Source }
    }
    $py = Get-Command py -ErrorAction SilentlyContinue
    if ($py) { $candidates += 'py -3.12' }
    $created = $false
    foreach ($candidate in $candidates) {
        try {
            if ($candidate -eq 'py -3.12') { & py -3.12 -c 'import sys; assert sys.version_info[:2] == (3,12)' 2>$null; if ($LASTEXITCODE -eq 0) { & py -3.12 -m venv $venv; $created = $LASTEXITCODE -eq 0 } }
            else { & $candidate -c 'import sys; assert sys.version_info[:2] == (3,12)' 2>$null; if ($LASTEXITCODE -eq 0) { & $candidate -m venv $venv; $created = $LASTEXITCODE -eq 0 } }
            if ($created) { break }
        } catch { }
    }
    if (-not $created) { throw 'Python 3.12 was not found. Install Python 3.12, then rerun bootstrap.' }
}
if ($Install) {
    & $exe -m pip install --disable-pip-version-check --only-binary=:all: -r $req
    if ($LASTEXITCODE -ne 0) { throw 'Pinned dependency installation failed.' }
}
& $exe -c 'import importlib.metadata as m, pathlib, sys; assert sys.version_info[:2] == (3,12); req=pathlib.Path(sys.argv[1]).read_text().splitlines(); [(lambda n,v: (lambda x: x == v or (_ for _ in ()).throw(RuntimeError(f"{n}: {x} != {v}")))(m.version(n)))(*line.split("==")) for line in req if line.strip()]; import rapidfuzz,anyascii,joblib,lightgbm,numpy,polars,psutil,sklearn,scipy,sparse_dot_topn,threadpoolctl,xgboost; print(sys.executable)' $req
if ($LASTEXITCODE -ne 0) { throw 'Pinned V2 import/version verification failed. Run bootstrap.ps1 -Install.' }
Write-Output $exe
