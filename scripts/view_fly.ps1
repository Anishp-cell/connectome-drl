# Launch Interactive 3D MuJoCo Viewer with hardware acceleration
$env:GALLIUM_DRIVER = "d3d12"
$env:MESA_D3D12_DEFAULT_ADAPTER_NAME = "NVIDIA"

$resetFound = $false
$filteredArgs = @()
foreach ($a in $args) {
    if ($a -eq "--reset" -or $a -eq "-r") {
        $resetFound = $true
    } else {
        $filteredArgs += $a
    }
}

if ($resetFound) {
    Write-Host "[!] Clearing WSLg graphics cache and resetting display server..." -ForegroundColor Yellow
    wsl --shutdown
    Start-Sleep -Seconds 1
}

Write-Host "Launching 3D MuJoCo Fruit Fly Viewer on NVIDIA GPU..." -ForegroundColor Cyan
Write-Host "Tip: If the window is off-screen, click the Tux icon and press Win + Up Arrow." -ForegroundColor DarkGray
Write-Host "Tip: If [WARN: COPY MODE] persists, run with --reset to clear the cache.`n" -ForegroundColor DarkGray

$cmdArgs = "$filteredArgs"
wsl -d Ubuntu bash -ic "cd /mnt/d/python/biomechanical_drl && /home/anish/miniconda3/envs/connectome-rl/bin/python scripts/view_fly_interactive.py $cmdArgs"


