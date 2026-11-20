# Stop the Kintsugi-27B server and the layer started from THIS folder. Servers started from other folders are left alone.
#   powershell -ExecutionPolicy Bypass -File stop.ps1
# If the layer is saving a chat right now (it saves a long chat after each answer), the stop waits for that save, up to
# 30 s, so the chat can be restored after the next start. With no layer (layer off, or no Python) it stops at once.
$Root = $PSScriptRoot
$Bin = (Join-Path $Root 'bin').ToLower()
$layer = Join-Path $Root 'layer\bonsai_layer.py'
$layers = @(Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" -ErrorAction SilentlyContinue |
    Where-Object { $_.CommandLine -and $_.CommandLine -like "*$layer*" })

# ---- wait for a chat save in flight: the layer answers GET /layer/slot-cache with "busy" (a chat request with its
#      save is running) and "saving" (a background save after an answer is still being written); a layer of an
#      older build has no "saving" field (then only busy counts)
foreach ($lp in $layers) {
    $port = if ($lp.CommandLine -match '--port\s+(\d+)') { $Matches[1] } else { '8080' }
    $hostName = if ($lp.CommandLine -match '--host\s+(\S+)') { $Matches[1] } else { '127.0.0.1' }
    if ($hostName -eq '0.0.0.0' -or $hostName -eq '::') { $hostName = '127.0.0.1' }
    $url = "http://$($hostName):$port/layer/slot-cache"
    $told = $false
    $until = (Get-Date).AddSeconds(30)
    while ((Get-Date) -lt $until) {
        $busy = $false
        try {
            $r = Invoke-RestMethod -Uri $url -TimeoutSec 2 -ErrorAction Stop
            $busy = [bool]$r.busy -or [bool]$r.saving
        } catch { break }                      # the layer does not answer: stop at once, as before
        if (-not $busy) { break }
        if (-not $told) { Write-Output "waiting for a chat save to finish..."; $told = $true }
        Start-Sleep -Milliseconds 500
    }
}

$n = 0
foreach ($p in (Get-Process llama-server -ErrorAction SilentlyContinue)) {
    $path = ''
    try { $path = $p.Path } catch {}
    if (-not $path -or -not $path.ToLower().StartsWith($Bin)) { continue }
    Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue
    try { $p.WaitForExit(15000) | Out-Null } catch {}
    $n++
}
foreach ($p in $layers) { Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue; $n++ }
Write-Output "stopped $n"
