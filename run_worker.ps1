param([Parameter(Mandatory)][string]$WorkerName,[string]$ShardIds,[int]$WorkerIndex=-1,[int]$WorkerCount=0,[int]$MaxShards=0)
$ErrorActionPreference='Stop'
$root=$PSScriptRoot
$bootstrap=Join-Path $root 'code/business_entity_resolution/v2/bootstrap.ps1'
$python=(& $bootstrap -Install | Select-Object -Last 1)
if ($LASTEXITCODE -ne 0) { throw 'Bootstrap failed' }
$workerArguments=@('distributed/worker.py','--worker-name',$WorkerName)
if ($ShardIds) { $workerArguments+=@('--shard-ids',$ShardIds) }
elseif ($WorkerIndex -ge 0 -and $WorkerCount -gt 0) { $workerArguments+=@('--worker-index',"$WorkerIndex",'--worker-count',"$WorkerCount") }
else { throw 'Supply -ShardIds or -WorkerIndex and -WorkerCount' }
if ($MaxShards -gt 0) { $workerArguments+=@('--max-shards',"$MaxShards") }
Push-Location $root
try { & $python @workerArguments; if ($LASTEXITCODE -ne 0) { throw 'Worker failed' } }
finally { Pop-Location }
