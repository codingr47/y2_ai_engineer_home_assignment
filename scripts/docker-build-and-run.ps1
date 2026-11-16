<#
.SYNOPSIS
Build and run the API, applying .env and mounting the local E5 encoder when enabled.
.EXAMPLE
.\scripts\docker-build-and-run.ps1
.EXAMPLE
.\scripts\docker-build-and-run.ps1 -Port 8001 -Detach
#>
[CmdletBinding()]
param(
    [ValidateRange(1, 65535)][int] $Port = 8000,
    [string] $Image = 'yad2-api:local',
    [string] $Name = 'yad2-api',
    [string] $EnvFile = '.env',
    [string] $EncoderPath,
    [switch] $Detach,
    [switch] $DryRun
)

$ErrorActionPreference = 'Stop'
$projectRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))

function Get-ProjectPath([string] $Value) {
    if ([System.IO.Path]::IsPathRooted($Value)) { return [System.IO.Path]::GetFullPath($Value) }
    return [System.IO.Path]::GetFullPath((Join-Path $projectRoot $Value))
}

$envPath = Get-ProjectPath $EnvFile
if (-not (Test-Path -LiteralPath $envPath -PathType Leaf)) {
    throw "Environment file not found: $envPath. Copy .env.example to .env first."
}

# Read only the settings needed for mounting; Docker consumes the complete env file.
# Do not execute .env contents or print secrets.
$extractorEnabled = 'false'
$configuredEncoder = 'models/multilingual-e5-small'
foreach ($line in Get-Content -LiteralPath $envPath -Encoding UTF8) {
    if ($line -match '^\s*EXTRACTOR_ENABLED=(.*)$') { $extractorEnabled = $Matches[1].Trim() }
    if ($line -match '^\s*E5_MODEL_PATH=(.*)$') { $configuredEncoder = $Matches[1].Trim() }
}
if ($extractorEnabled.ToLowerInvariant() -notin @('true', 'false', '1', '0')) {
    throw 'EXTRACTOR_ENABLED must be an unquoted true, false, 1, or 0 in the Docker env file.'
}

$runArgs = @('run', '--rm', '--name', $Name, '-p', "${Port}:8000", '--env-file', $envPath)
if ($Detach) { $runArgs += '-d' }
if ($extractorEnabled.ToLowerInvariant() -in @('true', '1')) {
    $candidates = @()
    if ($EncoderPath) {
        $candidates += Get-ProjectPath $EncoderPath
    } else {
        $candidates += Get-ProjectPath $configuredEncoder
        $candidates += Join-Path $projectRoot 'models/multilingual-e5-small'
        $hubCache = Join-Path $env:USERPROFILE '.cache/huggingface/hub'
        if ($env:HF_HOME) { $hubCache = Join-Path $env:HF_HOME 'hub' }
        if ($env:HF_HUB_CACHE) { $hubCache = $env:HF_HUB_CACHE }
        $candidates += Join-Path $hubCache 'models--intfloat--multilingual-e5-small/snapshots/614241f622f53c4eeff9890bdc4f31cfecc418b3'
    }
    $encoderDirectory = $candidates | Where-Object {
        (Test-Path -LiteralPath (Join-Path $_ 'config.json') -PathType Leaf) -and
        (Test-Path -LiteralPath (Join-Path $_ 'modules.json') -PathType Leaf) -and
        (Test-Path -LiteralPath (Join-Path $_ 'model.safetensors') -PathType Leaf)
    } | Select-Object -First 1
    if (-not $encoderDirectory) {
        throw 'E5 encoder not found. Provision it as described in docs/optional-parser.md, or pass -EncoderPath <directory>.'
    }
    $encoderDirectory = (Resolve-Path -LiteralPath $encoderDirectory).Path
    $mountSource = $encoderDirectory
    $containerEncoder = '/models/e5'
    $snapshotParent = Split-Path -Parent $encoderDirectory
    $repositoryDirectory = Split-Path -Parent $snapshotParent
    # HF snapshots may link to ../../blobs. Mount the repository to preserve those links.
    if ((Split-Path -Leaf $snapshotParent) -eq 'snapshots' -and
        (Test-Path -LiteralPath (Join-Path $repositoryDirectory 'blobs') -PathType Container)) {
        $mountSource = $repositoryDirectory
        $containerEncoder = '/models/e5/snapshots/' + (Split-Path -Leaf $encoderDirectory)
    }
    if ($mountSource.Contains(',')) { throw 'Docker mount paths must not contain commas.' }
    Write-Host "E5 encoder: $encoderDirectory"
    $runArgs += @('--mount', "type=bind,source=$mountSource,target=/models/e5,readonly", '-e', "E5_MODEL_PATH=$containerEncoder")
}
$runArgs += $Image
$buildArgs = @('build', '-t', $Image, $projectRoot)

if ($DryRun) {
    foreach ($commandArgs in @($buildArgs, $runArgs)) {
        $display = ($commandArgs | ForEach-Object { "'" + $_.Replace("'", "''") + "'" }) -join ' '
        Write-Host "docker $display"
    }
    return
}

if (-not (Get-Command docker -ErrorAction SilentlyContinue)) { throw 'Docker CLI not found. Install and start Docker first.' }
& docker info --format '{{.ServerVersion}}' *> $null
if ($LASTEXITCODE -ne 0) { throw 'Docker daemon is unavailable. Start Docker Desktop or the Docker service first.' }

& docker @buildArgs
if ($LASTEXITCODE -ne 0) { throw 'Docker build failed; the container was not started.' }
Write-Host "API: http://localhost:$Port/docs"
Write-Host "Container: $Name. Stop it with: docker stop $Name"
& docker @runArgs
if ($LASTEXITCODE -ne 0) {
    throw "Docker run failed. Check the error above; if '$Name' already exists, stop it or choose a different -Name."
}
