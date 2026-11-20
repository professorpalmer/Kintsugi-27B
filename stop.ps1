# Stop the Kintsugi-27B server and the layer started from THIS folder. Servers started from other folders are left alone.
#   powershell -ExecutionPolicy Bypass -File stop.ps1
$Root = $PSScriptRoot
$Bin = (Join-Path $Root 'bin').ToLower()
$n = 0
foreach ($p in (Get-Process llama-server -ErrorAction SilentlyContinue)) {
    $path = ''
    try { $path = $p.Path } catch {}
    if (-not $path -or -not $path.ToLower().StartsWith($Bin)) { continue }
    Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue
    try { $p.WaitForExit(15000) | Out-Null } catch {}
    $n++
}
$layer = Join-Path $Root 'layer\bonsai_layer.py'
foreach ($p in (Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" -ErrorAction SilentlyContinue)) {
    if ($p.CommandLine -and $p.CommandLine -like "*$layer*") { Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue; $n++ }
}
Write-Output "stopped $n"
