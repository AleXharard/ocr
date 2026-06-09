# Repornire Key Auto: opreste instanta veche, porneste una noua
param([switch]$Background)

$ProjectDir = $PSScriptRoot
$MainPy = Join-Path $ProjectDir "main.py"
$PidFile = Join-Path $ProjectDir ".keyauto.pid"
$CurrentPid = $PID

function Stop-KeyAutoProcess {
    param([int]$ProcessId)
    if ($ProcessId -le 0 -or $ProcessId -eq $CurrentPid) { return }
    $proc = Get-Process -Id $ProcessId -ErrorAction SilentlyContinue
    if ($proc) {
        Write-Host "Oprire PID $ProcessId ($($proc.ProcessName))"
        Stop-Process -Id $ProcessId -Force -ErrorAction SilentlyContinue
    }
}

# 1) PID salvat de instanta anterioara
if (Test-Path $PidFile) {
    $savedPid = (Get-Content $PidFile -ErrorAction SilentlyContinue | Select-Object -First 1).Trim()
    if ($savedPid -match '^\d+$') {
        Stop-KeyAutoProcess -ProcessId ([int]$savedPid)
    }
    Remove-Item $PidFile -Force -ErrorAction SilentlyContinue
}

# 2) Cauta procese python/py care ruleaza main.py
$patterns = @(
    "*tot nou*main.py*",
    "*main.py*"
)

Get-CimInstance Win32_Process -ErrorAction SilentlyContinue | Where-Object {
    $_.ProcessId -ne $CurrentPid -and $_.CommandLine -and (
        ($_.Name -match '^(python|pythonw|py)(\.exe)?$') -and (
            $_.CommandLine -like "*$MainPy*" -or
            ($_.CommandLine -match 'main\.py' -and $_.CommandLine -like "*tot nou*")
        )
    )
} | ForEach-Object {
    Stop-KeyAutoProcess -ProcessId $_.ProcessId
}

Start-Sleep -Milliseconds 1000

Set-Location $ProjectDir
Write-Host "Pornire Key Auto..."

if ($Background) {
    Start-Process -FilePath "py" -ArgumentList "-3", "main.py" -WorkingDirectory $ProjectDir -WindowStyle Normal
} else {
    & py -3 main.py
}
