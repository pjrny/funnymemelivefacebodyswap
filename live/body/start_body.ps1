# start_body.ps1 - one-click Avatar1 live FULL BODY: webcam -> pose on this laptop -> skeleton to Runpod GPU ->
# rendered Avatar1 -> preview window 'Avatar1 body' + OBS Virtual Camera + http://127.0.0.1:8766/
# Run:  powershell -ExecutionPolicy Bypass -File "<FaceBodySwapStream>\live\start_body.ps1"
#   -NoTunnel  use the Runpod HTTPS proxy instead of the SSH tunnel    -Cam <n> webcam index
#   -SnapDir <dir>  save rendered output + skeleton every 3 s (no camera frames)
#   any other args go to capture_body.py, e.g.  start_body.ps1 -- --fps 10 --no-vcam
param([switch]$NoTunnel, [int]$Cam = 0, [string]$SnapDir = "", [string]$Video = "",
      [Parameter(ValueFromRemainingArguments = $true)]$Rest)
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$cfg = Get-Content "$here\live_config_body.json" | ConvertFrom-Json
$py = "$here\.venv\Scripts\python.exe"
$key = "$env:USERPROFILE\.ssh\runpod_avatar_laptop"
# stop any face/body live session still holding the OBS Virtual Camera
Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -match 'capture_webcam.py|capture_body.py' } |
    ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
$tunnel = $null
$url = "$($cfg.url)/ws?k=$($cfg.token)"
if (-not $NoTunnel -and (Test-Path $key) -and $cfg.ssh_host) {
    $tunnel = Start-Process ssh -ArgumentList @('-i', $key, '-p', "$($cfg.ssh_port)", '-o', 'StrictHostKeyChecking=accept-new',
        '-o', 'ServerAliveInterval=15', '-o', 'ExitOnForwardFailure=yes', '-N', '-L', '8765:127.0.0.1:8080', "root@$($cfg.ssh_host)") -WindowStyle Hidden -PassThru
    Start-Sleep 3
    if ($tunnel.HasExited) { Write-Host "SSH tunnel failed; using Runpod proxy"; $tunnel = $null }
    else { $url = "ws://127.0.0.1:8765/ws?k=$($cfg.token)"; Write-Host "SSH tunnel up (pid $($tunnel.Id))" }
}
$a = @("$here\capture_body.py", "--url", $url, "--cam", "$Cam")
if ($SnapDir) { $a += @("--snap-dir", $SnapDir) }
if ($Video) { $a += @("--video", $Video) }
if ($Rest) { $a += $Rest }
try { & $py @a } finally { if ($tunnel -and -not $tunnel.HasExited) { Stop-Process -Id $tunnel.Id } }
