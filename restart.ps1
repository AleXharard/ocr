param(
    [switch]$Background
)

$ErrorActionPreference = "SilentlyContinue"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

$exeOne = Join-Path $root "dist\KeyAuto.exe"
$exeFolder = Join-Path $root "dist\KeyAuto\KeyAuto.exe"
$exe = if (Test-Path $exeOne) { $exeOne } elseif (Test-Path $exeFolder) { $exeFolder } else { $null }

$pidFile = Join-Path $root ".keyauto.pid"
if (Test-Path $pidFile) {
    $oldPid = Get-Content $pidFile -Raw
    if ($oldPid -match '^\d+$') {
        Stop-Process -Id ([int]$oldPid) -Force -ErrorAction SilentlyContinue
    }
    Remove-Item $pidFile -Force -ErrorAction SilentlyContinue
}

Get-Process -Name "KeyAuto" -ErrorAction SilentlyContinue |
    ForEach-Object { Stop-Process -Id $_.Id -Force -ErrorAction SilentlyContinue }

Get-CimInstance Win32_Process -Filter "Name='python.exe' OR Name='py.exe'" |
    Where-Object { $_.CommandLine -match 'main\.py' -and $_.CommandLine -match [regex]::Escape($root) } |
    ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }

Start-Sleep -Milliseconds 400

if ($exe) {
    $args = @{
        FilePath         = $exe
        WorkingDirectory = Split-Path $exe
    }
    if ($Background) {
        $args.WindowStyle = "Hidden"
    }
    Start-Process @args | Out-Null
    exit 0
}

$py = if (Get-Command py -ErrorAction SilentlyContinue) { "py -3" } else { "python" }
$cmd = "Set-Location '$root'; & $py main.py"

if ($Background) {
    Start-Process powershell -ArgumentList @(
        "-NoProfile", "-ExecutionPolicy", "Bypass", "-WindowStyle", "Hidden", "-Command", $cmd
    ) | Out-Null
} else {
    Invoke-Expression $cmd
}
