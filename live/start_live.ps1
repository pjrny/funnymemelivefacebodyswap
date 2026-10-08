# start_live.ps1 - one-click Avatar1 live: webcam -> Runpod GPU -> preview window + OBS Virtual Camera
# Run from anywhere:  powershell -ExecutionPolicy Bypass -File "<FaceBodySwapStream>\live\start_live.ps1"
#   -NoTunnel      use the Runpod HTTPS proxy instead of the faster SSH tunnel
#   -Video <file>  drive from a video/image file instead of the webcam (testing)
#   -Cam <n>       webcam index (default 0)      -SnapDir <dir>  save a sample frame every 3 s
#   any other args go straight to capture_webcam.py, e.g.  start_live.ps1 -- --fps 15 --no-vcam
param([switch]$NoTunnel, [string]$Video = "", [int]$Cam = 0, [string]$SnapDir = "",
      [Parameter(ValueFromRemainingArguments = $true)]$Rest)
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$cfg = Get-Content "$here\live_config.json" | ConvertFrom-Json
$py = "$here\.venv\Scripts\python.exe"
$key = "$env:USERPROFILE\.ssh\runpod_avatar_laptop"
$tunnel = $null
$url = "$($cfg.url)/ws?k=$($cfg.token)"
if (-not $NoTunnel -and (Test-Path $key) -and $cfg.ssh_host) {
    $tunnel = Start-Process ssh -ArgumentList @('-i', $key, '-p', "$($cfg.ssh_port)", '-o', 'StrictHostKeyChecking=accept-new',
        '-o', 'ServerAliveInterval=15', '-o', 'ExitOnForwardFailure=yes', '-N', '-L', '8765:127.0.0.1:8080', "root@$($cfg.ssh_host)") -WindowStyle Hidden -PassThru
    Start-Sleep 3
    if ($tunnel.HasExited) { Write-Host "SSH tunnel failed; using Runpod proxy"; $tunnel = $null }
    else { $url = "ws://127.0.0.1:8765/ws?k=$($cfg.token)"; Write-Host "SSH tunnel up (pid $($tunnel.Id))" }
}
$a = @("$here\capture_webcam.py", "--url", $url, "--cam", "$Cam")
if ($Video) { $a += @("--video", $Video) }
if ($SnapDir) { $a += @("--snap-dir", $SnapDir) }
if ($Rest) { $a += $Rest }
try { & $py @a } finally { if ($tunnel -and -not $tunnel.HasExited) { Stop-Process -Id $tunnel.Id } }
