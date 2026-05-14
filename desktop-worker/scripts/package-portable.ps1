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

function New-AppIcon {
  $iconRoot = Join-Path $workerRoot "resources\rabbit-v2-modern"
  $buildRoot = Join-Path $workerRoot "build"
  $iconPath = Join-Path $buildRoot "icon.ico"
  $source512 = Join-Path $iconRoot "512.png"
  $temp256 = Join-Path $buildRoot "icon-256.png"

  if (-not (Test-Path -LiteralPath $source512)) {
    throw "Missing app icon source: $source512"
  }

  New-Item -ItemType Directory -Force -Path $buildRoot | Out-Null
  Add-Type -AssemblyName System.Drawing
  $image = [System.Drawing.Image]::FromFile($source512)
  try {
    $bitmap = New-Object System.Drawing.Bitmap 256, 256
    try {
      $graphics = [System.Drawing.Graphics]::FromImage($bitmap)
      try {
        $graphics.InterpolationMode = [System.Drawing.Drawing2D.InterpolationMode]::HighQualityBicubic
        $graphics.SmoothingMode = [System.Drawing.Drawing2D.SmoothingMode]::HighQuality
        $graphics.PixelOffsetMode = [System.Drawing.Drawing2D.PixelOffsetMode]::HighQuality
        $graphics.DrawImage($image, 0, 0, 256, 256)
        $bitmap.Save($temp256, [System.Drawing.Imaging.ImageFormat]::Png)
      } finally {
        $graphics.Dispose()
      }
    } finally {
      $bitmap.Dispose()
    }
  } finally {
    $image.Dispose()
  }

  $pngs = @("32.png", "180.png", "192.png") | ForEach-Object {
    Join-Path $iconRoot $_
  }
  $pngs += $temp256
  foreach ($png in $pngs) {
    if (-not (Test-Path -LiteralPath $png)) {
      throw "Missing app icon frame: $png"
    }
  }

  $entries = @()
  foreach ($png in $pngs) {
    $bytes = [System.IO.File]::ReadAllBytes($png)
    $name = [System.IO.Path]::GetFileNameWithoutExtension($png)
    $size = if ($name -eq "icon-256") { 256 } else { [int]$name }
    $entries += [pscustomobject]@{ Size = $size; Bytes = $bytes }
  }

  $out = New-Object System.IO.MemoryStream
  $writer = New-Object System.IO.BinaryWriter($out)
  $writer.Write([UInt16]0)
  $writer.Write([UInt16]1)
  $writer.Write([UInt16]$entries.Count)
  $offset = 6 + (16 * $entries.Count)
  foreach ($entry in $entries) {
    $sizeByte = if ($entry.Size -ge 256) { 0 } else { $entry.Size }
    $writer.Write([byte]$sizeByte)
    $writer.Write([byte]$sizeByte)
    $writer.Write([byte]0)
    $writer.Write([byte]0)
    $writer.Write([UInt16]1)
    $writer.Write([UInt16]32)
    $writer.Write([UInt32]$entry.Bytes.Length)
    $writer.Write([UInt32]$offset)
    $offset += $entry.Bytes.Length
  }
  foreach ($entry in $entries) {
    $writer.Write($entry.Bytes)
  }
  $writer.Flush()
  [System.IO.File]::WriteAllBytes($iconPath, $out.ToArray())
  Write-Host "Generated app icon: $iconPath"
}

Stop-WorkerProcesses
New-AppIcon

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
