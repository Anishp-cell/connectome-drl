# Launch Interactive 3D MuJoCo Viewer with hardware acceleration
$env:GALLIUM_DRIVER = "d3d12"
$env:MESA_D3D12_DEFAULT_ADAPTER_NAME = "NVIDIA"

Write-Host "Launching 3D MuJoCo Fruit Fly Viewer on NVIDIA GPU..." -ForegroundColor Cyan
$cmdArgs = "$args"
wsl -d Ubuntu bash -ic "cd /mnt/d/python/biomechanical_drl && /home/anish/miniconda3/envs/connectome-rl/bin/python scripts/view_fly_interactive.py $cmdArgs"

