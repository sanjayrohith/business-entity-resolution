$ErrorActionPreference='Stop'
$root=$PSScriptRoot
$python=Join-Path $root '.venv/Scripts/python.exe'
if (-not (Test-Path $python)) { throw 'Run bootstrap.ps1 -Install first' }
& $python (Join-Path $root 'distributed/status.py')
if ($LASTEXITCODE -ne 0) { throw 'Status failed' }
