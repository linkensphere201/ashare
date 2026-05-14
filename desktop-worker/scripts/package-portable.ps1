$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$workerRoot = Resolve-Path (Join-Path $scriptDir "..")
$targetExe = Join-Path $workerRoot "dist\Stock Picker Desktop Worker 0.1.0.exe"

Set-Location $workerRoot

function Test-FileLocked {
  param([string]$Path)
  if (-not (Test-Path -LiteralPath $Path)) {
    return $false
  }
  try {
    $stream = [System.IO.File]::Open($Path, [System.IO.FileMode]::Open, [System.IO.FileAccess]::ReadWrite, [System.IO.FileShare]::None)
    $stream.Close()
    return $false
  } catch {
    return $true
  }
}

function Stop-WorkerProcesses {
  $candidateNames = @("Stock Picker Desktop Worker", "stock-picker-runtime")
  $processes = Get-Process | Where-Object {
    $candidateNames -contains $_.ProcessName
  }

  foreach ($process in $processes) {
    Write-Host "Stopping process $($process.ProcessName) pid=$($process.Id)"
    Stop-Process -Id $process.Id -Force
  }
}

Stop-WorkerProcesses

if (Test-FileLocked -Path $targetExe) {
  Write-Host "Target exe is locked, waiting for release: $targetExe"
  $deadline = (Get-Date).AddSeconds(30)
  while ((Get-Date) -lt $deadline -and (Test-FileLocked -Path $targetExe)) {
    Start-Sleep -Seconds 1
  }
  if (Test-FileLocked -Path $targetExe) {
    throw "Target exe is still locked after stopping worker processes: $targetExe"
  }
}

npm.cmd run package:runtime
npx.cmd electron-builder --win portable
