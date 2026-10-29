# Training stand environment check (Windows host + Docker/WSL2).
# Compatible with Windows PowerShell 5.1. Read-only: changes nothing on the system.
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File env_check.ps1 > env_report.txt
#   powershell -ExecutionPolicy Bypass -File env_check.ps1 -DockerGpuTest > env_report.txt
#
# -DockerGpuTest pulls the nvidia/cuda base image (~100-200 MB) and runs nvidia-smi
# inside a container to confirm the GPU is visible from Docker.

param(
    [switch]$DockerGpuTest,
    [string]$CudaImage = "nvidia/cuda:12.8.1-base-ubuntu22.04"
)

$ErrorActionPreference = "Continue"
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

function Section($name) { Write-Output ""; Write-Output "== $name" }

function Run($exe, [string[]]$arguments) {
    if (-not (Get-Command $exe -ErrorAction SilentlyContinue)) {
        Write-Output "$exe : NOT FOUND"
        return
    }
    try { & $exe @arguments 2>&1 | ForEach-Object { "$_" } }
    catch { Write-Output "$exe failed: $($_.Exception.Message)" }
}

# wsl.exe writes UTF-16 to a pipe; decode it as such and drop empty lines
function RunWsl([string[]]$arguments) {
    $saved = [Console]::OutputEncoding
    Remove-Item Env:WSL_UTF8 -ErrorAction SilentlyContinue
    try {
        [Console]::OutputEncoding = [Text.Encoding]::Unicode
        $out = & wsl @arguments 2>&1 | ForEach-Object { "$_".Replace("`0", "") } | Where-Object { $_.Trim() }
        return @{ Code = $LASTEXITCODE; Lines = $out }
    }
    finally { [Console]::OutputEncoding = $saved }
}

Section "OS"
$os = Get-CimInstance Win32_OperatingSystem
Write-Output "$($os.Caption) | version $($os.Version) | build $($os.BuildNumber) | $($os.OSArchitecture)"
if ([int]$os.BuildNumber -lt 19044) {
    Write-Output "WARNING: GPU in WSL2/Docker needs Windows 10 build 19044+ or Windows 11"
}

Section "Machine / virtualization"
$cs = Get-CimInstance Win32_ComputerSystem
Write-Output "Manufacturer: $($cs.Manufacturer) | Model: $($cs.Model)"
Write-Output "HypervisorPresent: $($cs.HypervisorPresent)"
$cpu = Get-CimInstance Win32_Processor | Select-Object -First 1
Write-Output "VirtualizationFirmwareEnabled: $($cpu.VirtualizationFirmwareEnabled)"
Write-Output "(If this is a VM, WSL2/Docker require nested virtualization enabled by the hypervisor)"

Section "CPU / RAM"
Write-Output "CPU: $($cpu.Name.Trim())"
Write-Output "Cores: $($cpu.NumberOfCores) | Logical: $($cs.NumberOfLogicalProcessors)"
Write-Output ("RAM total: {0:N1} GB | free: {1:N1} GB" -f ($cs.TotalPhysicalMemory / 1GB), ($os.FreePhysicalMemory * 1KB / 1GB))

Section "Disks"
Get-CimInstance Win32_LogicalDisk -Filter "DriveType=3" | ForEach-Object {
    "{0} total {1:N0} GB | free {2:N0} GB" -f $_.DeviceID, ($_.Size / 1GB), ($_.FreeSpace / 1GB)
}

Section "GPU (host)"
Run "nvidia-smi" @("--query-gpu=name,compute_cap,memory.total,memory.used,driver_version,power.limit", "--format=csv")
if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    nvidia-smi | Select-String "CUDA Version" | ForEach-Object { "$_".Trim() }
    # Section headers in nvidia-smi -q are followed by a "Current : ..." line
    $q = nvidia-smi -q
    for ($i = 0; $i -lt $q.Count; $i++) {
        if ($q[$i] -match '^\s*(MIG Mode|Driver Model)\s*$') {
            Write-Output ("{0}: {1}" -f $Matches[1], ($q[$i + 1] -replace '^.*:\s*', ''))
        }
        elseif ($q[$i] -match '^\s*(Virtualization Mode|Host VGPU Mode)\s*:\s*(.*)$') {
            Write-Output ("{0}: {1}" -f $Matches[1], $Matches[2])
        }
    }
}
else {
    Get-CimInstance Win32_VideoController | ForEach-Object { "$($_.Name) | driver $($_.DriverVersion)" }
}

Section "WSL"
if (Get-Command wsl -ErrorAction SilentlyContinue) {
    $v = RunWsl @("--version")
    if ($v.Code -eq 0) { $v.Lines }
    else { Write-Output "wsl --version unsupported: inbox WSL; Docker Desktop needs 'wsl --update'" }
    (RunWsl @("--status")).Lines
    Write-Output "--- distributions (GPU works only with VERSION 2)"
    (RunWsl @("-l", "-v")).Lines
}
else {
    Write-Output "wsl : NOT FOUND"
}
$wslConfig = Join-Path $env:USERPROFILE ".wslconfig"
if (Test-Path $wslConfig) {
    Write-Output "--- $wslConfig (limits RAM/CPU available to containers)"
    Get-Content $wslConfig
}
else {
    Write-Output ".wslconfig not found: WSL2 gets 50% of host RAM by default"
}

Section "Docker"
Run "docker" @("version", "--format", "client {{.Client.Version}} | server {{.Server.Version}} {{.Server.Os}}/{{.Server.Arch}}")
Run "docker" @("info", "--format", "OSType={{.OSType}} | Runtimes={{json .Runtimes}} | CPUs={{.NCPU}} | Mem={{.MemTotal}} | Root={{.DockerRootDir}}")
Run "docker" @("context", "ls")
if (Get-Command docker -ErrorAction SilentlyContinue) {
    docker info *> $null
    if ($LASTEXITCODE -ne 0) { Write-Output "WARNING: Docker engine is not running (start Docker Desktop and re-run)" }
}

if ($DockerGpuTest) {
    Section "Docker GPU test ($CudaImage)"
    Run "docker" @("run", "--rm", "--gpus", "all", $CudaImage, "nvidia-smi")
}
else {
    Section "Docker GPU test"
    Write-Output "skipped (run with -DockerGpuTest)"
}

Section "Tools on host (optional; normally inside the container)"
foreach ($t in @("python", "git", "iverilog", "yosys")) {
    $c = Get-Command $t -ErrorAction SilentlyContinue
    if ($c) { Write-Output "$t : $($c.Source)" } else { Write-Output "$t : not found" }
}

Section "Network"
foreach ($url in @("https://huggingface.co", "https://pypi.org", "https://registry-1.docker.io/v2/", "https://download.pytorch.org")) {
    try {
        $r = Invoke-WebRequest -Uri $url -Method Head -UseBasicParsing -TimeoutSec 10
        Write-Output "$url : $($r.StatusCode)"
    }
    catch {
        $resp = $_.Exception.Response
        # registry-1.docker.io answers 401 without a token: that still means it is reachable
        if ($resp) { Write-Output "$url : HTTP $([int]$resp.StatusCode) (reachable)" }
        else { Write-Output "$url : UNREACHABLE ($($_.Exception.Message))" }
    }
}
$target = [Uri]"https://huggingface.co"
$proxy = [System.Net.WebRequest]::GetSystemWebProxy().GetProxy($target)
if ($proxy -eq $target) { Write-Output "System proxy: none" } else { Write-Output "System proxy: $proxy" }
