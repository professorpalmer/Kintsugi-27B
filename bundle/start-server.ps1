# Kintsugi-27B on one 12 GB NVIDIA card (RTX 20 / 30 / 40 / 50, H100): 262,144-token window, tiered K/V cache, MTP + lookup drafting.
#
#   powershell -ExecutionPolicy Bypass -File start-server.ps1 [-Port 8080] [-Effort medium] [-Wait] [-DryRun]
#
#   -Effort   the default reasoning effort: medium (default) | xhigh | low (the same as KINTSUGI_EFFORT; -Effort wins)
#   -Wait     wait until the server answers, send one warm-up request, print the VRAM in use, then return
#   -DryRun   print the settings and the server command line, start nothing
#
# The model file goes in models\Kintsugi-27B-12G.gguf (see README.md). It listens on 127.0.0.1 only (this computer),
# unless you set KINTSUGI_HOST. Stop it with: powershell -ExecutionPolicy Bypass -File stop.ps1
#
# Settings (environment variables; each one also works with the prefix ESCHA_ in place of KINTSUGI_):
#   KINTSUGI_MODEL         model file (default models\Kintsugi-27B-12G.gguf)
#   KINTSUGI_CTX           window in tokens (default 262144; with the tiered cache it is rounded up to a multiple of 32,768)
#   KINTSUGI_KV            K/V cache type: q4_0 (default) | q8_0 | q3_0
#   KINTSUGI_MARGIN        MiB kept free under the card's limit (default 250 with no display on the card, 600 with one)
#   KINTSUGI_CELLS         K/V cells in VRAM (skips the arithmetic below)
#   KINTSUGI_DRAFT         auto (default: DFlash2 draft 7 when the drafter file is there, else MTP 3 + lookup) | dflash |
#                          mtp | off; a number = the MTP draft size (0 = no drafting)
#   KINTSUGI_DRAFTER       DFlash2 drafter file (default models\dflash-Qwen3.8-27B-Q4_0.gguf; an optional download)
#   KINTSUGI_VISION        auto (default: images on when the projector file is there; the projector on the CPU on a card
#                          under 16 GB, on the GPU from 16 GB) | gpu | cpu | off. cpu: no VRAM cost, an image reads in
#                          5-25 s; gpu: ~1.3 GB of VRAM (fewer K/V cells in VRAM), an image reads in 1-3 s
#   KINTSUGI_MMPROJ        vision projector file (default models\mmproj-Qwen3.8-27B-Q8_0.gguf; an optional download)
#   KINTSUGI_IMAGE_MAX_TOKENS  tokens for one image at most (default 2048; a larger image is scaled down to fit)
#   KINTSUGI_THINK         medium (default) | off
#   KINTSUGI_EFFORT        the default reasoning effort: medium (default) | xhigh | low. A request can set its own
#                          ("reasoning_effort"); a word that is not xhigh, medium or low gets this default. xhigh thinks
#                          longer and is slower (see README.md)
#   KINTSUGI_READ          fast (default: int8 read of long prompts) | exact
#   KINTSUGI_HEAD2         two-stage output head: exact candidates for each row (default 512; 0 = off)
#   KINTSUGI_HOST          address clients connect to (default 127.0.0.1 = this computer only; 0.0.0.0 = your network:
#                          then set KINTSUGI_KEY_FILE too, or anyone on the network can use your GPU)
#   KINTSUGI_KEY_FILE      API key file; with it, clients must send "Authorization: Bearer <key>"
#   KINTSUGI_LAYER         1 (default) = saved conversations through the layer (needs Python); 0 = plain server
#   KINTSUGI_SLOT_SAVE_PATH  folder of the saved conversations (default slots\ in this folder)
#   KINTSUGI_SLOT_CACHE_GB   size limit of that folder in GB (default 20; the oldest files are removed past it)
#   KINTSUGI_CACHE_RAM     MiB of system RAM for the server's prompt cache (default 4096; 0 = off)
#   KINTSUGI_GPU_COUNTERS  1 (default) = count the VRAM that other programs hold (Windows GPU memory counters); 0 = nvidia-smi only
#   KINTSUGI_CPU           auto (default: the CPU library that this CPU can run, checked at the first start) | avx2 |
#                          avx (the AVX + F16C library, for CPUs without AVX2, for example Intel Ivy Bridge)
#   KINTSUGI_UB            micro-batch for prompt reads (default 1024; 512 saves about 146 MiB and reads 8-13 % slower)
#   KINTSUGI_ALIAS         the model name that /v1/models lists and replies carry (default kintsugi; an --alias in
#                          KINTSUGI_EXTRA wins)
#   KINTSUGI_EXTRA         more llama-server arguments
#   KINTSUGI_CHAT_TEMPLATE chat template file (default templates\kintsugi.jinja: the model's template, except that a
#                          system message after the first turn, as Claude Code sends, is accepted); "model" = the
#                          template inside the model file; any other value = the path of your own template
param([int]$Port = 8080, [string]$Effort = '', [switch]$Wait, [switch]$DryRun)
$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
function Pick($name, $default) {
    foreach ($n in @("KINTSUGI_$name", "ESCHA_$name")) {
        $v = [Environment]::GetEnvironmentVariable($n); if ($v) { return $v }
    }
    $default
}
# A driver too old for CUDA 13 (or no CUDA device) does not stop the engine: it runs on the CPU, very slowly, and still
# answers. Returns the stop message when the server log shows that, else $null.
function Get-NoGpuMessage($LogPath) {
    if (-not (Test-Path $LogPath)) { return $null }
    if (Select-String -Path $LogPath -SimpleMatch -Pattern 'failed to initialize CUDA', 'no usable GPU found' -Quiet) {
        return "the engine could not use the GPU (see $LogPath): update the NVIDIA driver to 580 or newer"
    }
    $null
}
# The server stopped during the load: stop the layer too (no layer with no server behind it), print the last lines of the
# server log, exit 1. A server that stopped at once with an empty log never ran (a Windows loader or CPU fault): then
# the message points to the Windows event log and names the exit code.
function Test-NotStarted {
    if (-not ($p -and $p.HasExited)) { return $false }
    $len = 0; foreach ($f in $log, ($log + '.out')) { if (Test-Path $f) { $len += (Get-Item $f).Length } }
    $len -eq 0 -and $p.ExitTime -and (($p.ExitTime - $p.StartTime).TotalSeconds -lt 15)
}
function Stop-LoadFailed {
    if ($lp -and -not $lp.HasExited) { Stop-Process -Id $lp.Id -Force -ErrorAction SilentlyContinue; try { $lp.WaitForExit(10000) | Out-Null } catch { } }
    if (Test-NotStarted) {
        $code = try { [int]$p.ExitCode } catch { $null }
        $hex = if ($null -ne $code) { '0x{0:X8}' -f $code } else { 'unknown' }
        Write-Host "the process did not start (log 0 bytes, it stopped at once, exit code $hex): see the Windows event log"
        Write-Host "  (Event Viewer > Windows Logs > Application, source 'Application Error'; or in PowerShell:"
        Write-Host "   Get-WinEvent -LogName Application -MaxEvents 30 | Where-Object { `$_.Message -match 'llama-server' } | Format-List TimeCreated, Message)"
        if ($hex -eq '0xC000001D') {
            Write-Host "  0xC000001D = illegal instruction: this CPU lacks an instruction of the CPU library in bin\. $(if ((Get-CpuLibrary) -eq 'avx') { 'The AVX library is in use, so the CPU lacks AVX or F16C.' } else { 'Start again with KINTSUGI_CPU=avx (the AVX + F16C library).' })"
        } elseif ($hex -eq '0xC0000135') {
            Write-Host "  0xC0000135 = a DLL was not found: install the Microsoft Visual C++ Redistributable 2015-2022 (x64), or unzip the bundle again."
        }
        exit 1
    }
    Write-Host "server stopped during load: see $log"
    Get-Content -LiteralPath $log -Tail 3 -ErrorAction SilentlyContinue | ForEach-Object { Write-Host "  $_" }
    exit 1
}
function Stop-NoGpu($Message) {
    if ($p) { Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue }
    if ($lp) { Stop-Process -Id $lp.Id -Force -ErrorAction SilentlyContinue }
    Write-Host $Message
    exit 1
}

# The size of a file, also when the path is a symbolic link to it (Get-Item gives the link's own length then).
function FileLength($path) {
    $it = Get-Item -LiteralPath $path
    for ($n = 0; $n -lt 8 -and $it.LinkType -eq 'SymbolicLink'; $n++) {
        $t = @($it.Target)[0]
        if (-not [IO.Path]::IsPathRooted($t)) { $t = Join-Path (Split-Path -Parent $it.FullName) $t }
        $it = Get-Item -LiteralPath $t
    }
    $it.Length
}
$Model = Pick 'MODEL' (Join-Path $Root 'models\Kintsugi-27B-12G.gguf')
$Bin = Join-Path $Root 'bin'
$Ctx = [int](Pick 'CTX' 262144)
$Kv = Pick 'KV' 'q4_0'
$Think = Pick 'THINK' 'medium'
# Reasoning effort: the template accepts xhigh, medium and low (it stops with an error on any other word).
if (-not $Effort) { $Effort = Pick 'EFFORT' 'medium' }
$Effort = $Effort.Trim().ToLower()
if ($Effort -notin @('medium', 'xhigh', 'low')) { throw "KINTSUGI_EFFORT must be medium, xhigh or low (got '$Effort')" }
if (-not (Test-Path $Model)) {
    Write-Host "Kintsugi-27B: model file not found: $Model"
    Write-Host "  Download Kintsugi-27B-12G.gguf from https://huggingface.co/CaryPalmer/Kintsugi-27B"
    Write-Host "  and put it in $(Join-Path $Root 'models'). SHA-256 0510bae82445dc53dbc7d0d842d8b7e146b8eafce0837e39b157d411e48c8a74"
    Write-Host "  (check it with: Get-FileHash models\Kintsugi-27B-12G.gguf). Or set KINTSUGI_MODEL to the file's path."
    exit 1
}
if (-not (Test-Path (Join-Path $Bin 'llama-server.exe'))) { throw "llama-server.exe not found in $Bin" }
if (-not (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) { throw "nvidia-smi not found: install the NVIDIA driver" }

# ---- NVIDIA driver: the CUDA 13 runtime in bin\ needs driver 580 or newer ---------------------------------------------
# KINTSUGI_TEST_DRIVER=<version> replaces the value that nvidia-smi gives (tests).
$Driver = Pick 'TEST_DRIVER' ''
if (-not $Driver) {
    try { $Driver = "$((& nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>$null) | Select-Object -First 1)".Trim() } catch { $Driver = '' }
}
if ($Driver -match '^(\d+)\.') {
    if ([int]$Matches[1] -lt 580) {
        Write-Host "NVIDIA driver $Driver is too old: Kintsugi needs driver 580 or newer (CUDA 13). Update the driver, then start again."
        exit 1
    }
} else {
    Write-Host "WARN   could not read the NVIDIA driver version from nvidia-smi: the driver check is skipped"
}

# ---- Card generation: bin\ has native code for compute capability 7.5 / 8.6 / 8.9 (RTX 20 / 30 / 40), 9.0 (H100 / H200)
# and 12.0 (RTX 50), and portable code (compute_90 PTX) that newer cards compile at the first start. CUDA loads only the
# code of your card, so the code for the other cards costs no VRAM or speed. KINTSUGI_TEST_CC=<x.y> replaces the value
# that nvidia-smi gives (tests).
$Cc = Pick 'TEST_CC' ''
if (-not $Cc) {
    try { $Cc = "$((& nvidia-smi --query-gpu=compute_cap --format=csv,noheader -i 0 2>$null) | Select-Object -First 1)".Trim() } catch { $Cc = '' }
}
if ($Cc -match '^(\d+)\.(\d+)$') {
    $CcMajor = [int]$Matches[1]; $CcMinor = [int]$Matches[2]; $CcN = 10 * $CcMajor + $CcMinor
    $Supported = 'RTX 20 / 30 / 40 / 50 cards with 12 GB or more, and H100 / H200'
    if ($CcN -lt 75) {
        Write-Host "This NVIDIA card (compute capability $Cc) is older than RTX 20 (Turing, 7.5): Kintsugi-27B has no code for it. Supported: $Supported."
        exit 1
    } elseif (($CcMajor -eq 7 -and $CcN -ne 75) -or ($CcMajor -eq 8 -and $CcMinor -lt 6)) {
        Write-Host "This NVIDIA card (compute capability $Cc, for example a V100 / A100 / A30) has no code in this bundle. Supported: $Supported."
        exit 1
    } elseif ($CcN -ge 90 -and $CcN -ne 90 -and $CcN -ne 120) {
        Write-Host "note   this card (compute capability $Cc) runs the engine's portable code: the driver compiles it at the first start (several minutes, once, then cached). We have not tested this card."
        # the driver's compile cache must hold all of it, or every start compiles again: 4 GiB (the largest value)
        if (-not $env:CUDA_CACHE_MAXSIZE) { $env:CUDA_CACHE_MAXSIZE = '4294967296' }
    }
} else {
    Write-Host "WARN   could not read the card's compute capability from nvidia-smi: the card check is skipped"
}

# ---- CPU (issue #21): bin\ggml-cpu.dll is built for AVX2 + FMA + BMI2 (Intel Haswell 2013 / AMD Zen 2017 or newer);
# ---- bin\ggml-cpu-avx.dll for AVX + F16C (Intel Ivy Bridge 2012 or newer). At the first start (and when the CPU is a
# ---- different one) the launcher runs `llama-server.exe --version` once: when it stops with 0xC000001D (illegal
# ---- instruction), the launcher puts the AVX library in place (two renames in bin\) and checks again. The result is in
# ---- bin\cpu-library.txt. KINTSUGI_CPU=auto (default) | avx2 | avx sets the library without the check.
# The GPU does nearly all of the work, so the speed is about the same with either library.
# KINTSUGI_TEST_CPU_SDE=<path of Intel SDE's sde.exe> runs the check under `sde -ivb` (an emulated Ivy Bridge CPU; tests).
$CpuName = "$((Get-CimInstance Win32_Processor -ErrorAction SilentlyContinue | Select-Object -First 1).Name)".Trim() -replace '\s+', ' '
$CpuWant = (Pick 'CPU' 'auto').Trim().ToLower()
if ($CpuWant -notin @('auto', 'avx2', 'avx')) { throw "KINTSUGI_CPU must be auto, avx2 or avx (got '$CpuWant')" }
$CpuMark = Join-Path $Bin 'cpu-library.txt'
# which build bin\ggml-cpu.dll is now: the other build is next to it under its own name
function Get-CpuLibrary { if (Test-Path (Join-Path $Bin 'ggml-cpu-avx2.dll')) { 'avx' } else { 'avx2' } }
function Set-CpuLibrary([string]$To) {
    $now = Get-CpuLibrary
    if ($To -eq $now) { return $true }
    $cur = Join-Path $Bin 'ggml-cpu.dll'; $keep = Join-Path $Bin "ggml-cpu-$now.dll"; $new = Join-Path $Bin "ggml-cpu-$To.dll"
    if (-not (Test-Path $new)) { Write-Host "the CPU library $new is missing (unzip the bundle again)"; exit 1 }
    if (@(Get-Process llama-server -ErrorAction SilentlyContinue | Where-Object { $_.Path -like "$Bin\*" }).Count) {
        Write-Host "a llama-server from this folder is running: stop it first (stop.ps1), then start again"; exit 1
    }
    try { Move-Item -LiteralPath $cur -Destination $keep -ErrorAction Stop } catch { return $false }
    try { Move-Item -LiteralPath $new -Destination $cur -ErrorAction Stop } catch {
        Move-Item -LiteralPath $keep -Destination $cur -ErrorAction SilentlyContinue; return $false
    }
    $true
}
function Test-CpuLibrary {
    # 'ok' | 'illegal' (the CPU lacks an instruction of bin\ggml-cpu.dll) | 'other <exit code>'
    $exe = Join-Path $Bin 'llama-server.exe'; $sde = Pick 'TEST_CPU_SDE' ''
    $o = Join-Path $env:TEMP "kintsugi-cpu-check-$PID.out"; $e = Join-Path $env:TEMP "kintsugi-cpu-check-$PID.err"
    if ($sde) { $f = $sde; $al = @('-ivb', '--', ('"' + $exe + '"'), '--version') } else { $f = $exe; $al = @('--version') }
    $pr = Start-Process -FilePath $f -ArgumentList $al -WorkingDirectory $Bin -WindowStyle Hidden -PassThru -RedirectStandardOutput $o -RedirectStandardError $e
    $null = $pr.Handle   # Windows PowerShell 5.1 keeps the exit code only when the handle was read
    if (-not $pr.WaitForExit(120000)) { Stop-Process -Id $pr.Id -Force -ErrorAction SilentlyContinue; return 'other timeout' }
    $pr.WaitForExit()
    $code = [int]$pr.ExitCode
    $text = (Get-Content -Raw -LiteralPath $e -ErrorAction SilentlyContinue) + (Get-Content -Raw -LiteralPath $o -ErrorAction SilentlyContinue)
    Remove-Item -LiteralPath $o, $e -ErrorAction SilentlyContinue
    if ($code -eq -1073741795 -or ($sde -and $text -match 'not valid for specified chip')) { return 'illegal' }   # 0xC000001D
    if ($code -eq 0) { return 'ok' }
    'other 0x{0:X8}' -f $code
}
$NoWrite = "the launcher can not rename files in $Bin (no write rights there?). Copy bin\ggml-cpu-avx.dll over bin\ggml-cpu.dll by hand (keep the old file as bin\ggml-cpu-avx2.dll), or move the Kintsugi folder to a place where you can write, then start again."
if ($CpuWant -ne 'auto') {
    if (-not $DryRun -and -not (Set-CpuLibrary $CpuWant)) { Write-Host $NoWrite; exit 1 }
    $CpuUse = $CpuWant
} else {
    $mark = if (Test-Path $CpuMark) { @(Get-Content -LiteralPath $CpuMark -ErrorAction SilentlyContinue) } else { @() }
    $CpuUse = Get-CpuLibrary
    if (-not $DryRun -and -not ($mark.Count -ge 2 -and $mark[1] -eq $CpuName -and $mark[0] -eq $CpuUse)) {
        # first start on this CPU: the AVX2 library first, the AVX library when the CPU lacks an AVX2-build instruction
        if (-not (Set-CpuLibrary 'avx2')) { Write-Host $NoWrite; exit 1 }
        $r = Test-CpuLibrary
        if ($r -eq 'illegal') {
            if (-not (Set-CpuLibrary 'avx')) { Write-Host $NoWrite; exit 1 }
            $r = Test-CpuLibrary
            if ($r -eq 'illegal') {
                $null = Set-CpuLibrary 'avx2'
                Write-Host "This CPU ($CpuName) lacks AVX or F16C: the engine needs at least AVX + F16C (Intel Ivy Bridge 2012 / AMD 2012 or newer). Kintsugi-27B can not run on it."
                exit 1
            }
        }
        $CpuUse = Get-CpuLibrary
        # only a check that ran to its end is kept (another result: check again at the next start)
        if ($r -eq 'ok') { try { Set-Content -LiteralPath $CpuMark -Value @($CpuUse, $CpuName) -Encoding ASCII -ErrorAction Stop } catch { } }
    }
}
if ($CpuUse -eq 'avx') {
    Write-Host "cpu    ${CpuName}: $(if ($CpuWant -eq 'avx') { 'KINTSUGI_CPU=avx' } else { 'no AVX2 / FMA / BMI2' }), so the engine uses its AVX + F16C CPU library (the AVX2 one is kept as bin\ggml-cpu-avx2.dll). The GPU does nearly all of the work; the speed is about the same."
}

# ---- VRAM line: how many K/V cells fit in VRAM; the rest of the window lives in system RAM -----------------------------
$HalfBytes = @{ 'q8_0' = 17408; 'q4_0' = 9216; 'q3_0' = 7168 }   # bytes of K (or of V) per cell, all attention layers
if (-not $HalfBytes.ContainsKey($Kv)) { throw "KINTSUGI_KV must be one of: $($HalfBytes.Keys -join ', ')" }
$Cell = 2 * $HalfBytes[$Kv]
$q = (& nvidia-smi --query-gpu=memory.total,memory.used,display_active --format=csv,noheader,nounits | Select-Object -First 1).Split(',')
$TotalMiB = [int]$q[0]; $UsedMiB = [int]$q[1]; $Headless = $q[2].Trim() -ne 'Enabled'
$SmiUsedMiB = $UsedMiB
if ((Pick 'TEST_DISPLAY' '') -eq '1') { $Headless = $false }   # test hook: plan as with a display on the card
# ---- VRAM of other programs (issue #17). Windows moves memory of an idle program out of VRAM and back when it needs
# ---- it; nvidia-smi shows only what is in VRAM now. The Windows GPU memory counters ("Total Committed" of each process
# ---- on this card) show all of it. With the display on this card the plan counts the larger figure: the desktop uses
# ---- its memory again whenever the screen changes. With no display on the card: a note only (an idle program's memory
# ---- stays out of VRAM; rig test 2026-11-24). KINTSUGI_GPU_COUNTERS=0 skips this.
$OthersMiB = $null
if ((Pick 'GPU_COUNTERS' '1') -ne '0') {
    $t = Pick 'TEST_OTHERS_MIB' ''
    if ($t) { $OthersMiB = [int]$t } else {
        try {
            Add-Type -ErrorAction Stop -TypeDefinition @'
using System;
using System.Runtime.InteropServices;
public static class KtGpu {
    [StructLayout(LayoutKind.Sequential)] public struct LUID { public uint Low; public int High; }
    [StructLayout(LayoutKind.Sequential)] public struct ADAPTERINFO { public uint hAdapter; public LUID Luid; public uint NumOfSources; public int Precise; }
    [StructLayout(LayoutKind.Sequential)] public struct ENUM2 { public uint NumAdapters; public IntPtr pAdapters; }
    [StructLayout(LayoutKind.Sequential)] public struct QUERY { public uint hAdapter; public int Type; public IntPtr pData; public uint Size; }
    [StructLayout(LayoutKind.Sequential)] public struct DEVIDS { public uint PhysIndex; public uint Vendor; public uint Device; public uint SubVendor; public uint SubSystem; public uint Revision; public uint Bus; }
    [StructLayout(LayoutKind.Sequential)] public struct CLOSE { public uint hAdapter; }
    [DllImport("gdi32.dll")] static extern int D3DKMTEnumAdapters2(ref ENUM2 e);
    [DllImport("gdi32.dll")] static extern int D3DKMTQueryAdapterInfo(ref QUERY q);
    [DllImport("gdi32.dll")] static extern int D3DKMTCloseAdapter(ref CLOSE c);
    public static string[] Luids(uint vendor) {
        var res = new System.Collections.Generic.List<string>();
        var e = new ENUM2();
        if (D3DKMTEnumAdapters2(ref e) != 0 || e.NumAdapters == 0) { return res.ToArray(); }
        int sz = Marshal.SizeOf(typeof(ADAPTERINFO));
        e.pAdapters = Marshal.AllocHGlobal(sz * (int) e.NumAdapters);
        try {
            if (D3DKMTEnumAdapters2(ref e) != 0) { return res.ToArray(); }
            for (int i = 0; i < e.NumAdapters; i++) {
                var a = (ADAPTERINFO) Marshal.PtrToStructure(new IntPtr(e.pAdapters.ToInt64() + i * sz), typeof(ADAPTERINFO));
                IntPtr p = Marshal.AllocHGlobal(Marshal.SizeOf(typeof(DEVIDS)));
                Marshal.StructureToPtr(new DEVIDS(), p, false);
                var q = new QUERY { hAdapter = a.hAdapter, Type = 31, pData = p, Size = (uint) Marshal.SizeOf(typeof(DEVIDS)) };
                if (D3DKMTQueryAdapterInfo(ref q) == 0 && ((DEVIDS) Marshal.PtrToStructure(p, typeof(DEVIDS))).Vendor == vendor) {
                    res.Add(string.Format("luid_0x{0:x8}_0x{1:x8}", a.Luid.High, a.Luid.Low));
                }
                Marshal.FreeHGlobal(p);
                var c = new CLOSE { hAdapter = a.hAdapter };
                D3DKMTCloseAdapter(ref c);
            }
        } finally { Marshal.FreeHGlobal(e.pAdapters); }
        return res.ToArray();
    }
}
'@
            $luids = @([KtGpu]::Luids(0x10DE))
            if ($luids.Count -eq 1) {
                $s = (Get-Counter '\GPU Process Memory(*)\Total Committed' -ErrorAction Stop).CounterSamples | Where-Object { $_.InstanceName -like "*$($luids[0])*" }
                $OthersMiB = [int](($s | Measure-Object CookedValue -Sum).Sum / 1MB)
            }
        } catch { $OthersMiB = $null }
    }
    if ($null -ne $OthersMiB -and $OthersMiB -gt $UsedMiB + 64) {
        if (-not $Headless) {
            Write-Host "note   other programs hold $OthersMiB MiB of this card (Windows GPU memory counters; nvidia-smi shows $UsedMiB MiB in VRAM now): with the display on this card the plan counts $OthersMiB MiB (KINTSUGI_GPU_COUNTERS=0 skips this)"
            $UsedMiB = $OthersMiB
        } else {
            Write-Host "note   other programs hold $OthersMiB MiB of this card (Windows GPU memory counters; nvidia-smi shows $UsedMiB MiB in VRAM now). If they use it again, the server can spill to system RAM and get slow (a 'spill' line in the server log): close them for full speed"
        }
    }
}
if ($TotalMiB -lt 11500) { Write-Host "WARN   this card has $TotalMiB MiB of VRAM. Kintsugi-27B needs a 12 GB card (the model alone takes about 9.5 GB): it may not load, or most of the context will be in system RAM and slow." }
# Margin under the card's limit, on top of the measured growth (a display on the card moves by itself, so more margin).
$Margin = [int](Pick 'MARGIN' $(if ($Headless) { 250 } else { 600 }))
$MarginWhy = if (Pick 'MARGIN' '') { 'KINTSUGI_MARGIN' } elseif ($Headless) { 'no display on this card' } else { 'display on this card: nvidia-smi display_active; KINTSUGI_MARGIN=250 if that is wrong' }
# Fixed footprint at load without the K/V cells (measured on an RTX 4070 with the 10,546,686,688-byte file); a model file
# of another size moves it by the difference.
$HeadMiB = [int][math]::Round(((FileLength $Model) - 10546686688) / 1MB)
# Two-stage output head (default 512 candidates): the exact Q8_0 head goes to system RAM, a small first stage stays in
# VRAM (606 MiB saved at load). The text is the same as with the full head on our tests.
# ---- DFlash2 -> MTP, once: when DFlash2 was chosen automatically (KINTSUGI_DRAFT=auto) and the server stops during the
# ---- load or the warm-up request fails, the launcher keeps that log, stops the server (and the layer) and starts again
# ---- with MTP + lookup drafting. KINTSUGI_DRAFT=dflash fails instead. Only with -Wait (the warm-up needs it).
$Retried = $false; $SetDflashUb = $false
function Test-MtpRetry($why) {
    if (-not ($script:Mode -eq 'dflash' -and $script:DraftSet -eq 'auto' -and -not $script:Retried)) { return $false }
    $failLog = $script:log -replace '\.log$', '.dflash-failed.log'
    Copy-Item -LiteralPath $script:log -Destination $failLog -Force -ErrorAction SilentlyContinue
    Write-Host "note   DFlash2 start failed ($why): starting again once with MTP + lookup drafting (KINTSUGI_DRAFT=mtp skips DFlash2)."
    Write-Host "       Please report it with your card, driver and the log $failLog at https://github.com/professorpalmer/Kintsugi-27B/issues"
    @(Select-String -Path $script:log -Pattern 'out of memory|cudaMalloc|failed to allocate|CUDA error|error|failed|abort' -ErrorAction SilentlyContinue) |
        Select-Object -Last 6 | ForEach-Object { Write-Host "  log  $($_.Line.Trim())" }
    foreach ($x in @($script:lp, $script:p)) {
        if ($x -and -not $x.HasExited) { Stop-Process -Id $x.Id -Force -ErrorAction SilentlyContinue; try { $x.WaitForExit(30000) | Out-Null } catch { } }
    }
    # the driver frees the memory of a stopped process a moment later; the cell count of the retry reads the free VRAM
    for ($i = 0; $i -lt 20; $i++) {
        $u = [int]((& nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits) | Select-Object -First 1)
        if ($u -le $script:SmiUsedMiB + 64) { break }
        Start-Sleep 1
    }
    if ($script:SetDflashUb) { Remove-Item Env:LLAMA_DFLASH_DRAFT_UBATCH -ErrorAction SilentlyContinue }
    $script:Retried = $true
    $true
}
:launch while ($true) {
    # ---- Drafting: DFlash2 (draft 7) when its drafter file is there, else the model's own MTP block (draft 3) + lookup ------
    # The DFlash2 drafter is an optional download (models\dflash-Qwen3.8-27B-Q4_0.gguf, see README.md). It takes about 0.8 GB
    # of VRAM, so fewer K/V cells fit in VRAM; measured on an RTX 4070 with these settings: about +25 to +35 % decode at
    # 4k-64k tokens, about +23 % at 117k, the same speed at 245k.
    $Drafter = Pick 'DRAFTER' (Join-Path $Root 'models\dflash-Qwen3.8-27B-Q4_0.gguf')
    $DraftSet = (Pick 'DRAFT' 'auto').ToLower()
    if ($Retried) { $DraftSet = 'mtp' }   # the one retry after a failed DFlash2 start (below)
    $DraftNote = ''
    if ($DraftSet -match '^\d+$') { $Mode = if ([int]$DraftSet -gt 0) { 'mtp' } else { 'off' }; $MtpDraft = [int]$DraftSet }
    else {
        $MtpDraft = 3
        switch ($DraftSet) {
            'auto'   { $Mode = if (Test-Path $Drafter) { 'dflash' } else { 'mtp' }
                       if ($Mode -eq 'mtp') { $DraftNote = "note   DFlash2 drafter not found ($Drafter): MTP + lookup drafting. For about 25-35 % faster decode up to ~64k tokens (about 23 % at 117k), download dflash-Qwen3.8-27B-Q4_0.gguf from https://huggingface.co/CaryPalmer/Kintsugi-27B into models\" } }
            'dflash' { if (-not (Test-Path $Drafter)) { throw "DFlash2 drafter not found: $Drafter (set KINTSUGI_DRAFTER, or KINTSUGI_DRAFT=auto)" }; $Mode = 'dflash' }
            'mtp'    { $Mode = 'mtp' }
            'off'    { $Mode = 'off' }
            default  { throw "KINTSUGI_DRAFT must be auto, dflash, mtp, off or a number" }
        }
    }
    $Head2 = [int](Pick 'HEAD2' 512)
    $Head2MiB = if ($Head2 -gt 0) { 606 } else { 0 }
    # With DFlash2 (draft micro-batch 64): 10,238 MiB at load for the 1,094,346,080-byte drafter (rig, 2026-11-19); another
    # drafter file moves it by the difference.
    $DrafterMiB = if ($Mode -eq 'dflash') { 765 + [int][math]::Round(((FileLength $Drafter) - 1094346080) / 1MB) } else { 0 }
    # ---- Vision: the Qwen3.8 projector (mmproj) reads images; an optional download -------------------------------------
    # auto: on a card under 16 GB the projector runs on the CPU (it keeps every K/V cell in VRAM; on a 12 GB card the GPU
    # projector would leave about 61,000 of 144,000), on 16 GB or more on the GPU (faster image reads).
    # On the GPU it takes VRAM for its weights and for the work of one image at the cap (measured on an RTX 4070,
    # docs: KINTSUGI-VISION V3), so fewer K/V cells fit in VRAM; the window stays the same. On the CPU (KINTSUGI_VISION=cpu)
    # it takes almost no VRAM and reads an image more slowly.
    $Mmproj = Pick 'MMPROJ' (Join-Path $Root 'models\mmproj-Qwen3.8-27B-Q8_0.gguf')
    $VisSet = (Pick 'VISION' 'auto').ToLower()
    switch ($VisSet) {
        'auto' { $Vision = if (-not (Test-Path $Mmproj)) { 'off' } elseif ($TotalMiB -ge 15500) { 'gpu' } else { 'cpu' } }
        'gpu'  { if (-not (Test-Path $Mmproj)) { throw "vision projector not found: $Mmproj (set KINTSUGI_MMPROJ, or KINTSUGI_VISION=auto)" }; $Vision = 'gpu' }
        'cpu'  { if (-not (Test-Path $Mmproj)) { throw "vision projector not found: $Mmproj (set KINTSUGI_MMPROJ, or KINTSUGI_VISION=auto)" }; $Vision = 'cpu' }
        'off'  { $Vision = 'off' }
        default { throw "KINTSUGI_VISION must be auto, gpu, cpu or off" }
    }
    # DFlash2 does not work with images yet (an image late in a long conversation fails in the drafter, KINTSUGI-VISION T3):
    # with vision on, drafting is MTP + lookup.
    if ($Vision -ne 'off' -and $Mode -eq 'dflash') {
        if ($DraftSet -eq 'dflash') { throw "KINTSUGI_DRAFT=dflash does not work with vision on: set KINTSUGI_VISION=off for DFlash2, or KINTSUGI_DRAFT=auto" }
        $Mode = 'mtp'; $DrafterMiB = 0
        $DraftNote = "note   vision on: MTP + lookup drafting (the DFlash2 drafter does not work with images yet; KINTSUGI_VISION=off uses it)"
    }
    # ---- More than one slot (-np N or --parallel N in KINTSUGI_EXTRA): the VRAM of the recurrent state ---------------
    # One slot (the default) keeps one copy of the recurrent state (about 150 MiB) and the fast rollback. With N slots
    # and drafting, the engine keeps 1 + D copies for each slot (D = the draft size: 7 for DFlash2, 3 for MTP), so that
    # it can roll back rejected draft tokens: about 150 MiB x (1 + D) x N. (Up to v1.2.1 a rollback with 3 or more slots
    # could read the copies of another slot, and an answer broke off into unrelated text: found by the NetSphere lab,
    # 2026-11-27. v1.2.2 fixes it; LLAMA_RS_MULTI_SEQ_GUARD=1 turns drafting off with more than one slot, as a safety net.)
    $Slots = 1
    $SlotArgs = @((Pick 'EXTRA' '').Split(' ') | Where-Object { $_ })
    for ($i = 0; $i + 1 -lt $SlotArgs.Count; $i++) {
        if ($SlotArgs[$i] -in @('-np', '--parallel') -and $SlotArgs[$i + 1] -match '^\d+$') { $Slots = [int]$SlotArgs[$i + 1] }
    }
    $RsMiB = 0; $RsNote = ''
    if ($Slots -gt 1) {
        if ($env:LLAMA_RS_MULTI_SEQ_GUARD -and $env:LLAMA_RS_MULTI_SEQ_GUARD -ne '0' -and $Mode -ne 'off') {
            $Mode = 'off'; $DrafterMiB = 0
            $DraftNote = "note   $Slots slots (-np $Slots) and LLAMA_RS_MULTI_SEQ_GUARD: drafting is off."
        }
        $RsD = if ($Mode -eq 'dflash') { 7 } elseif ($Mode -eq 'mtp') { $MtpDraft } else { 0 }
        $RsMiB = 150 * (1 + $RsD) * $Slots - 150
        $RsNote = "note   $Slots slots (-np $Slots): the recurrent state takes about $(150 * (1 + $RsD) * $Slots) MiB of VRAM ($(1 + $RsD) x 150 MiB for each slot)."
    }
    $ImgMax = [int](Pick 'IMAGE_MAX_TOKENS' 2048)
    # Measured for the 629,247,008-byte Q8_0 projector at a 2048-token cap; another file moves it by the difference.
    $VisionMiB = if ($Vision -eq 'gpu') { 852 + [int][math]::Round(((FileLength $Mmproj) - 629247008) / 1MB) } elseif ($Vision -eq 'cpu') { 0 } else { 0 }
    $Fixed = 9473 + $HeadMiB - $Head2MiB + $DrafterMiB + $VisionMiB + $RsMiB
    # The first request adds about 222 MiB once (cuBLAS, kernel code at first use); the warm-up request pays it at start.
    # Requests of many new lengths then add up to about 210 MiB (CUDA graphs); the margin covers that. With DFlash2, 128 MiB
    # more: Linux needs about 50 MiB more for it than Windows, and the rest keeps the peak 64 MiB or more under the limit.
    $Growth = 222 + $(if ($Mode -eq 'dflash') { 128 } else { 0 }) + $(if ($Vision -eq 'gpu') { 512 } else { 0 })
    # The driver keeps part of the card for itself ("Reserved" in nvidia-smi -q; 272 MiB on an RTX 4070 with Windows, 416 MiB
    # on one with Linux). Past the rest, Windows moves memory to system RAM without an error and everything slows down (the
    # "spill" below). Ceiling = the smaller of 97.7 % of the card and the card less the reserved part less 8 MiB.
    # If nvidia-smi does not show the reserved part, 512 MiB is assumed. KINTSUGI_CEILING_MIB overrides.
    $ReservedMiB = $null; $InFb = $false
    foreach ($l in @(& nvidia-smi -q -d MEMORY -i 0 2>$null)) {
        if ($l -match 'FB Memory Usage') { $InFb = $true }
        elseif ($InFb -and $l -match '^\s*Reserved\s*:\s*(\d+)\s*MiB') { $ReservedMiB = [int]$Matches[1]; break }
        elseif ($InFb -and $l -match 'Memory Usage') { break }
    }
    if ($null -eq $ReservedMiB) {
        $ReservedMiB = 512
        Write-Host "note   nvidia-smi does not show the card's reserved memory: 512 MiB assumed (KINTSUGI_CEILING_MIB overrides)"
    }
    $Ceiling = [int](Pick 'CEILING_MIB' ([math]::Min([math]::Floor($TotalMiB * 0.977), $TotalMiB - $ReservedMiB - 8)))
    # Staging in VRAM = one layer's K and V of the cells in system RAM = (window - cells) x cell / 16.
    function CellsFor($fixed) {
        $budget = ([math]::Min($TotalMiB, $Ceiling) - $UsedMiB - $Margin - $fixed - $Growth) * 1MB - $Ctx * $Cell / 16
        [math]::Min($Ctx, [math]::Max(3328, [math]::Floor($budget / ($Cell * 15 / 16) / 256) * 256))
    }
    $Cells = [int](Pick 'CELLS' (CellsFor $Fixed))
    if ($Cells -gt $Ctx) { $Cells = $Ctx }
    # Window guard: with the tiered cache (cells in VRAM < window) the window is a multiple of 32,768 tokens. Engines before
    # the fix wrote wrong V rows past the VRAM line on prompt reads otherwise; this engine has the fix, the guard stays.
    if ($Cells -lt $Ctx -and ($Ctx % 32768) -ne 0) {
        $CtxOld = $Ctx
        $Ctx = [int]([math]::Ceiling($Ctx / 32768) * 32768)
        if ($CtxOld -le 262144 -and $Ctx -gt 262144) { $Ctx = 262144 }
        Write-Host "note   window $CtxOld rounded up to $Ctx (a multiple of 32,768 with the tiered K/V cache)"
        $Cells = [int](Pick 'CELLS' (CellsFor $Fixed)); if ($Cells -gt $Ctx) { $Cells = $Ctx }
    }
    if ($Cells -le 3328) { Write-Host "note   only $Cells K/V cells fit in VRAM: decode reads most of the K/V cache from system RAM (close other programs that use the card)" }
    # ---- Hand-set cells: the launcher's own count for this card and drafting mode is the safe maximum. Past it, the first
    # ---- prompt read can fail with "out of memory" (RTX 3080 Ti with a display, DFlash2, 83,712 cells set by hand: about
    # ---- 700 MiB too many). The launcher keeps the value that you set and tells you which setting to lower.
    $OwnCells = [int](CellsFor $Fixed)
    $OwnMtp = $OwnCells
    if ($Mode -eq 'dflash') { $g0 = $Growth; $Growth = $Growth - 128; $OwnMtp = [int](CellsFor ($Fixed - $DrafterMiB)); $Growth = $g0 }
    $LowerLine = if ($Mode -eq 'dflash') { "lower KINTSUGI_CELLS to $OwnCells, or set KINTSUGI_DRAFT=mtp (MTP + lookup: at most $OwnMtp cells)" } else { "lower KINTSUGI_CELLS to $OwnCells" }
    if ((Pick 'CELLS' '') -and $Cells -gt $OwnCells) {
        $ModeTxt = if ($Mode -eq 'dflash') { 'DFlash2 drafting' } elseif ($Mode -eq 'mtp') { 'MTP + lookup drafting' } else { 'no drafting' }
        Write-Host "WARN   KINTSUGI_CELLS=$Cells is more than this card can hold with $ModeTxt`: the safe maximum is $OwnCells cells (free $($TotalMiB - $UsedMiB) MiB, ceiling $Ceiling, margin $Margin). A prompt read can fail with `"out of memory`"."
        Write-Host "       To fix it: $LowerLine. The launcher keeps your value."
    }

    # ---- System memory (issue #17). The K/V cells in system RAM are pinned RAM (never in the page file), the two-stage
    # ---- head takes 1,288 MiB pinned, and the server about 3,700 MiB (DFlash2) / 5,000 MiB (MTP) of other commit
    # ---- (measured, RTX 4070, 2026-11-24). A 16 GB PC with 1.1 GB of RAM free failed at 4.5 GB of host cells and started
    # ---- at 1.1 GB (#17). The launcher takes the largest window (a multiple of 32,768) whose host part fits: pinned
    # ---- <= free RAM + 2,048 MiB (Windows trims other programs to make room), and commit <= free commit (RAM + page
    # ---- file). With KINTSUGI_DRAFT=auto it takes MTP + lookup when that keeps a larger window. KINTSUGI_MEMCHECK=0 skips.
    $CacheRam = [int](Pick 'CACHE_RAM' 4096)
    $Head2Pin = if ($Head2 -gt 0) { 1288 } else { 0 }
    $CommitFree = $null; $RamFree = $null
    try { $os = Get-CimInstance Win32_OperatingSystem -ErrorAction Stop; $CommitFree = [int]($os.FreeVirtualMemory / 1KB); $RamFree = [int]($os.FreePhysicalMemory / 1KB) } catch { }
    $t = Pick 'TEST_COMMIT_FREE_MIB' ''; if ($t) { $CommitFree = [int]$t }
    $t = Pick 'TEST_RAM_FREE_MIB' ''; if ($t) { $RamFree = [int]$t }
    function CellsAt($fixed, $growth, $ctx) {
        $budget = ([math]::Min($TotalMiB, $Ceiling) - $UsedMiB - $Margin - $fixed - $growth) * 1MB - $ctx * $Cell / 16
        [int][math]::Min($ctx, [math]::Max(3328, [math]::Floor($budget / ($Cell * 15 / 16) / 256) * 256))
    }
    function HostMiB($ctx, $cells) { [int](($ctx - $cells) * $Cell / 1MB) }
    function Fits($ctx, $cells, $mode) {
        $h = (HostMiB $ctx $cells) + $Head2Pin
        $pinOk = ($null -eq $RamFree) -or ($h -le $RamFree + 2048)
        $comOk = ($null -eq $CommitFree) -or ($h + $(if ($mode -eq 'dflash') { 3700 } else { 5000 }) + 512 -le $CommitFree)
        $pinOk -and $comOk
    }
    function FitWindow($fixed, $growth, $mode) {
        for ($c = $Ctx; $c -ge 32768; $c -= 32768) { $n = CellsAt $fixed $growth $c; if (Fits $c $n $mode) { return @($c, $n) } }
        return @(32768, (CellsAt $fixed $growth 32768))
    }
    if ((Pick 'MEMCHECK' '1') -ne '0' -and $null -ne $RamFree -and -not (Fits $Ctx $Cells $Mode)) {
        $Why = "the K/V cells in system RAM need $(HostMiB $Ctx $Cells) MiB of pinned RAM$(if ($Head2Pin) { " (and the two-stage head $Head2Pin MiB)" }); $RamFree MiB of RAM is free, $CommitFree MiB of commit (RAM + page file)"
        if ((Pick 'CTX' '') -or (Pick 'CELLS' '')) {
            Write-Host "WARN   system memory: $Why. The load can fail with out of memory. KINTSUGI_CTX / KINTSUGI_CELLS are yours, so the launcher keeps them; to fix it: close other programs, or lower KINTSUGI_CTX."
        } else {
            $Ctx0 = $Ctx
            $Fit = FitWindow $Fixed $Growth $Mode
            if ($Mode -eq 'dflash' -and $DraftSet -eq 'auto') {
                $FitM = FitWindow ($Fixed - $DrafterMiB) ($Growth - 128) 'mtp'
                if ($FitM[0] -gt $Fit[0]) {
                    Write-Host "memory $Why. MTP + lookup drafting keeps a window of $($FitM[0]) tokens, DFlash2 only $($Fit[0]): MTP + lookup."
                    $Mode = 'mtp'; $Fixed = $Fixed - $DrafterMiB; $Growth = $Growth - 128; $DrafterMiB = 0; $Fit = $FitM
                }
            }
            $Ctx = [int]$Fit[0]; $Cells = [int]$Fit[1]
            if ($Ctx -lt $Ctx0) { Write-Host "memory window $Ctx0 -> $Ctx tokens ($Cells cells in VRAM), so the K/V cells in system RAM fit$(if ($Mode -ne 'mtp' -or $DraftSet -ne 'auto') { ": $Why" })." }
            Write-Host "       For the full window: close other programs (RAM), or make the page file larger. KINTSUGI_MEMCHECK=0 keeps the full window (the load can then fail with out of memory)."
        }
    }
    # ---- DFlash2 with few cells in VRAM (#17: 3,328 on a 12 GB card with a display): MTP + lookup keeps more of the window
    if ($Mode -eq 'dflash') {
        $MtpCells = CellsAt ($Fixed - $DrafterMiB) ($Growth - 128) $Ctx
        if ($Cells -lt 32768 -and $MtpCells -ge $Cells + 8192) {
            Write-Host "note   DFlash2 leaves $Cells K/V cells in VRAM (its drafter takes about $($DrafterMiB + 128) MiB); MTP + lookup (KINTSUGI_DRAFT=mtp) would keep $MtpCells. DFlash2 drafts up to 7 tokens a step; with MTP more of a long chat is read from VRAM."
        }
    }

    # ---- Drafting: the model's own MTP block (draft 3) + lookup drafting out of the prompt -----------------------------------
    $Draft = if ($Mode -eq 'mtp') { $MtpDraft } elseif ($Mode -eq 'dflash') { 7 } else { 0 }
    $Spec = @()
    if ($Mode -eq 'dflash') {
        $Spec = @('--spec-type', 'draft-dflash', '-md', ('"' + $Drafter + '"'), '--spec-draft-n-max', '7', '-ctkd', 'q4_0', '-ctvd', 'q4_0')
    } elseif ($Draft -gt 0) {
        $Spec = @('--spec-type', 'ngram-mod,draft-mtp', '--spec-draft-n-max', "$Draft", '--spec-draft-n-max-tail', "$([math]::Max(4, $Draft))",
                  '-ctkd', 'q8_0', '-ctvd', 'q8_0', '--spec-lookup-n-max', '32', '--spec-draft-window', '16384')
    }

    # ---- Engine switches -------------------------------------------------------------------------------------------------
    $env:GGML_CUDA_BATCH_INVARIANT = '1'          # a token's output does not depend on the batch it runs in
    if ($Mode -eq 'dflash' -and -not $env:LLAMA_DFLASH_DRAFT_UBATCH) { $env:LLAMA_DFLASH_DRAFT_UBATCH = '64'; $SetDflashUb = $true }
    if ($Draft -gt 0) { $env:LLAMA_RS_REPLAY = '1'; $env:GGML_CUDA_SHARED_POOL = '1'; if (-not $env:LLAMA_MTP_DRAFT_UBATCH) { $env:LLAMA_MTP_DRAFT_UBATCH = '256' } }
    $env:LLAMA_ARG_CHECKPOINT_EVERY_NT = '8192'
    if (-not $env:GGML_CUDA_POOL_RESERVE_MIB) { $env:GGML_CUDA_POOL_RESERVE_MIB = '96' }
    # Spill guard: the engine prints a "spill" line when the server's dedicated GPU memory goes past this figure.
    if (-not $env:GGML_CUDA_DEDICATED_CEILING_MIB) { $env:GGML_CUDA_DEDICATED_CEILING_MIB = "$Ceiling" }
    if ($Head2 -gt 0) { $env:LLAMA_HEAD2 = "$Head2"; if (-not $env:LLAMA_HEAD2_TYPE) { $env:LLAMA_HEAD2_TYPE = 'q4_0' } }
    # Read path for long prompts: fast = int8 matrix products, about 2x the exact read; exact = the fp16 path.
    $Read = Pick 'READ' 'fast'
    if (-not $env:GGML_ESCHA_PREFILL) { $env:GGML_ESCHA_PREFILL = if ($Read -eq 'exact') { 'recon' } else { 'i8' } }
    if (-not $env:GGML_ESCHA_I8_MIN_ROWS) { $env:GGML_ESCHA_I8_MIN_ROWS = '256' }
    # The default effort of a request; a request can set its own with "reasoning_effort" (below: --reasoning-effort-allow).
    $env:LLAMA_ARG_CHAT_TEMPLATE_KWARGS = if ($Think -eq 'off') { "{`"reasoning_effort`":`"$Effort`",`"enable_thinking`":false}" } else { "{`"reasoning_effort`":`"$Effort`"}" }
    $env:PATH = "$Bin;" + $env:PATH          # the engine and the CUDA runtime DLLs are in bin\

    $KeyFile = Pick 'KEY_FILE' ''
    # Client address: the layer listens on it (or the server, with the layer off). The server behind the layer always
    # listens on 127.0.0.1.
    $ListenHost = Pick 'HOST' '127.0.0.1'
    if ($ListenHost -ne '127.0.0.1' -and $ListenHost -ne 'localhost' -and -not $KeyFile) {
        Write-Host "WARNING listening on $ListenHost with NO API key: anyone who can reach this computer can use the server (and your GPU). Set KINTSUGI_KEY_FILE to a file with a key."
    }

    # ---- Saved conversations: the layer (layer\bonsai_layer.py) runs in front of the server on the client port; the
    # ---- server moves to 127.0.0.1:<port + 10000>. After each answer it saves a long conversation to disk (a full save,
    # ---- then small deltas); after a restart, the next turn of a saved conversation reads only its new tokens.
    # The layer's other tools (code interpreter, API cards and checks) are off in this build. It needs Python 3.10 or newer
    # and nothing else (the standard library only). Without it the server runs plain, with a note.
    $UseLayer = (Pick 'LAYER' '1') -eq '1'
    if ($UseLayer) {
        $py = Get-Command python -ErrorAction SilentlyContinue
        if (-not $py) {
            Write-Host "note   no python in PATH: saved conversations are off (the plain server runs). Install Python 3.10+ to use them, or set KINTSUGI_LAYER=0 to hide this note."
            $UseLayer = $false
        } else {
            $ok = $false
            try { & python -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" 2>$null; $ok = $LASTEXITCODE -eq 0 } catch { }
            if (-not $ok) {
                Write-Host "note   saved conversations are off: they need Python 3.10 or newer (the python in PATH is older or does not run). The plain server runs."
                $UseLayer = $false
            }
        }
    }
    $ServerPort = $Port
    $SlotDir = $null
    if ($UseLayer) {
        $ServerPort = $Port + 10000
        $SlotDir = Pick 'SLOT_SAVE_PATH' (Join-Path $Root 'slots')
        if (-not $DryRun) { $SlotDir = (New-Item -ItemType Directory -Force -Path $SlotDir).FullName }
    }
    $Ub = [int](Pick 'UB' 1024)
    # Chat template: the bundle's (default), the model file's own ("model"), or a file of the user's.
    $Template = Pick 'CHAT_TEMPLATE' (Join-Path $Root 'templates\kintsugi.jinja')
    if ($Template -ne 'model' -and -not (Test-Path $Template)) { throw "chat template not found: $Template (set KINTSUGI_CHAT_TEMPLATE)" }
    $SrvArgs = @('-m', ('"' + $Model + '"'), '-ngl', '99', '-c', "$Ctx", '-fa', 'on', '-np', '1', '-ub', "$Ub", '--fit', 'off', '--jinja',
              '-ctk', $Kv, '-ctv', $Kv, '--kv-vram-cells', "$Cells", '--kq-mask-packed', '--cache-ram', "$CacheRam",
              '--reasoning-budget', '20480', '-n', '24576', '--reasoning-budget-message', '"Now produce the complete answer."',
              '--reasoning-effort-allow', 'xhigh,medium,low', '--reasoning-effort-fallback', $Effort, '--reasoning-max-tokens-floor', '24576', '--reasoning-max-tokens-honor', '4096',
              '--host', $(if ($UseLayer) { '127.0.0.1' } else { $ListenHost }), '--port', "$ServerPort") + $Spec
    if ($Template -ne 'model') { $SrvArgs += @('--chat-template-file', ('"' + $Template + '"')) }
    if ($KeyFile) { $SrvArgs += @('--api-key-file', ('"' + $KeyFile + '"')) }
    if ($SlotDir) { $SrvArgs += @('--slot-save-path', ('"' + $SlotDir + '"')) }
    if ($Vision -ne 'off') {
        $SrvArgs += @('--mmproj', ('"' + $Mmproj + '"'), '--image-max-tokens', "$ImgMax")
        if ($Vision -eq 'cpu') { $SrvArgs += '--no-mmproj-offload' }
    }
    $Extra = Pick 'EXTRA' ''
    # The model name clients see (/v1/models, the "model" of each reply): "kintsugi", not the model file's path.
    $Alias = Pick 'ALIAS' 'kintsugi'
    if (" $Extra " -notmatch '\s(--alias|-a)\s') { $SrvArgs += @('--alias', $Alias) }
    if ($Extra) { $SrvArgs += $Extra.Split(' ') }

    Write-Host "Kintsugi-27B"
    Write-Host "model  $(Split-Path $Model -Leaf)"
    Write-Host "listen $ListenHost`:$Port$(if ($UseLayer) { " (layer; llama-server on 127.0.0.1:$ServerPort)" })$(if ($KeyFile) { ", API key on" })"
    if ($Slots -gt 1 -and ($Fixed + $Growth + $Margin + 600) -gt $Ceiling) {
        Write-Host "WARN   $Slots slots with this drafting need about $Fixed MiB of VRAM before any K/V cell; the limit on this card is $Ceiling MiB. The load can fail with `"out of memory`". Use fewer slots, or KINTSUGI_DRAFT=mtp (fewer state copies) or KINTSUGI_DRAFT=off."
    }
    Write-Host "kv     $Kv, tiered: cells 0..$Cells in VRAM, $Cells..$Ctx in system RAM (free $($TotalMiB - $UsedMiB) MiB, reserved $ReservedMiB, ceiling $Ceiling, fixed $Fixed, growth $Growth, margin $Margin ($MarginWhy))"
    Write-Host "template $(if ($Template -eq 'model') { 'the model file''s own' } else { $Template })"
    if ($Head2 -gt 0) { Write-Host "head   two-stage: first stage q4_0 in VRAM, $Head2 exact candidates for each row from system RAM" }
    Write-Host "spec   $(if ($Mode -eq 'off') { 'off' } elseif ($Mode -eq 'dflash') { "DFlash2 draft 7 ($(Split-Path $Drafter -Leaf))" } else { "MTP draft $Draft + lookup" }); thinking $Think; effort $Effort; read $($env:GGML_ESCHA_PREFILL)"
    if ($DraftNote) { Write-Host $DraftNote }
    if ($RsNote) { Write-Host $RsNote }
    Write-Host "vision $(if ($Vision -eq 'off') { 'off' } else { "$Vision ($(Split-Path $Mmproj -Leaf)), up to $ImgMax tokens per image" })"
    if ($DryRun) { Write-Host ("args   " + ($SrvArgs -join ' ')); exit 0 }

    $logs = Join-Path $Root 'logs'; New-Item -ItemType Directory -Force $logs | Out-Null
    $log = Join-Path $logs "server-$Port.log"
    $p = Start-Process -FilePath (Join-Path $Bin 'llama-server.exe') -ArgumentList ($SrvArgs -join ' ') -WorkingDirectory $Bin `
        -RedirectStandardOutput ($log + '.out') -RedirectStandardError $log -WindowStyle Hidden -PassThru
    $null = $p.Handle   # Windows PowerShell 5.1 keeps the exit code of a -PassThru process only when its handle was read
    Write-Host "pid    $($p.Id)   log $log"
    $lp = $null
    $layerLog = Join-Path $logs "layer-$Port.log"
    if ($UseLayer) {
        $env:BONSAI_SLOT_SAVE_PATH  = $SlotDir
        $env:BONSAI_SLOT_CACHE_GB   = Pick 'SLOT_CACHE_GB' '20'
        $env:BONSAI_SLOT_CACHE_AFTER = '1'
        if ($KeyFile) { $env:BONSAI_LAYER_KEY = (Get-Content -Raw $KeyFile).Trim() }
        [string[]]$LayerFlags = @('--no-interpreter', '--no-cards', '--no-lint', '--no-example-check', '--no-finish-note', '--no-input-file', '--no-strip-stray-tags')
        $lp = Start-Process python -PassThru -WindowStyle Hidden -RedirectStandardOutput ($layerLog + '.out') -RedirectStandardError $layerLog `
            -ArgumentList (@(('"' + (Join-Path $Root 'layer\bonsai_layer.py') + '"'), '--host', $ListenHost, '--port', "$Port",
                             '--upstream', "http://127.0.0.1:$ServerPort") + $LayerFlags)
        Remove-Item Env:BONSAI_LAYER_KEY, Env:BONSAI_SLOT_SAVE_PATH, Env:BONSAI_SLOT_CACHE_GB, Env:BONSAI_SLOT_CACHE_AFTER -ErrorAction SilentlyContinue
        Write-Host "layer  pid $($lp.Id)   log $layerLog   clients use :$Port; saved conversations in $SlotDir"
    }
    Write-Host "stop   powershell -ExecutionPolicy Bypass -File `"$(Join-Path $Root 'stop.ps1')`""
    if (-not $Wait) {
        # No wait for the load: the engine writes its GPU set-up to the log before the model load starts.
        for ($i = 0; $i -lt 60; $i++) {
            if ($p.HasExited -or (Select-String -Path $log -SimpleMatch -Pattern 'load_model: loading model' -Quiet -ErrorAction SilentlyContinue)) { break }
            Start-Sleep 1
        }
        $m = Get-NoGpuMessage $log; if ($m) { Stop-NoGpu $m }
        if ($p.HasExited) { Stop-LoadFailed }
    }
    if ($Wait) {
        for ($i = 0; $i -lt 150; $i++) {
            $up = $false
            try { $up = (Invoke-WebRequest -UseBasicParsing -TimeoutSec 3 "http://127.0.0.1:$Port/health").StatusCode -eq 200 } catch { }
            if ($up) {
                $m = Get-NoGpuMessage $log; if ($m) { Stop-NoGpu $m }
                # Warm-up request (about 1,340 tokens in, 8 out; more with KINTSUGI_UB above 1024, so that the prompt fills one
                # full micro-batch): the engine takes its first-use memory and its prompt-read buffers now, so the VRAM printed
                # below is the peak, and a shortage shows here and not at the first request.
                $nWarm = [math]::Max(110, [math]::Floor($Ub / 11) + 16)
                $text = (1..$nWarm | ForEach-Object { "Entry $_ of the log has nothing to report." }) -join ' '
                $body = @{ model = 'warmup'; max_tokens = 8; temperature = 0; cache_prompt = $false
                           chat_template_kwargs = @{ enable_thinking = $false }
                           messages = @(@{ role = 'user'; content = "$text Say OK." }) } | ConvertTo-Json -Depth 5
                $hdr = @{}
                if ($KeyFile) { $hdr['Authorization'] = 'Bearer ' + (Get-Content -Raw $KeyFile).Trim() }
                $sw = [Diagnostics.Stopwatch]::StartNew()
                $WarmErr = $null
                try { Invoke-WebRequest -UseBasicParsing -TimeoutSec 600 -Method Post -ContentType 'application/json' -Headers $hdr -Body $body "http://127.0.0.1:$Port/v1/chat/completions" | Out-Null }
                catch { $WarmErr = $_.Exception.Message }
                Start-Sleep -Milliseconds 500
                if (-not $WarmErr -and $p.HasExited) { $WarmErr = 'the server stopped during the warm-up request' }
                if ($WarmErr -and (Test-MtpRetry "the warm-up request failed: $WarmErr")) { continue launch }
                $Oom = $false
                if ($WarmErr) { $Oom = [bool](Select-String -Path $log -Pattern 'out of memory|failed to allocate' -Quiet -ErrorAction SilentlyContinue) }
                if ($WarmErr -and $Mode -eq 'dflash' -and $DraftSet -eq 'dflash') {
                    # KINTSUGI_DRAFT=dflash: no fallback to MTP and no "ready"; keep the log, stop the server, exit 1
                    $failLog = $log -replace '\.log$', '.dflash-failed.log'
                    Copy-Item -LiteralPath $log -Destination $failLog -Force -ErrorAction SilentlyContinue
                    Write-Host "note   the warm-up request failed with KINTSUGI_DRAFT=dflash (no fallback to MTP when DFlash2 is set): $WarmErr. Log kept: $failLog"
                    @(Select-String -Path $log -Pattern 'out of memory|cudaMalloc|failed to allocate|CUDA error|error|failed|abort' -ErrorAction SilentlyContinue) |
                        Select-Object -Last 6 | ForEach-Object { Write-Host "  log  $($_.Line.Trim())" }
                    if ($Oom) { Write-Host "VRAM   the server is short of VRAM with DFlash2 (out of memory). To fix it: $LowerLine." }
                    foreach ($x in @($lp, $p)) {
                        if ($x -and -not $x.HasExited) { Stop-Process -Id $x.Id -Force -ErrorAction SilentlyContinue; try { $x.WaitForExit(30000) | Out-Null } catch { } }
                    }
                    Write-Host "stopped: the server is not ready (the server and the layer are stopped)"; exit 1
                }
                if ($WarmErr) { Write-Host "note   the warm-up request failed: $WarmErr" }
                if ($Oom) { Write-Host "VRAM   the server is short of VRAM (out of memory in $log): requests will fail. Stop it (stop.ps1), then $LowerLine." }
                if ($p.HasExited) { Write-Host "server stopped after the warm-up request: see $log"; exit 1 }
                $vram = [int]((& nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits) | Select-Object -First 1)
                # Spill check: the server's dedicated GPU memory against what the card holds. More in the process than on the
                # card means the driver moved the difference to system RAM (the engine prints "spill" lines for the same).
                $ded = 0
                try { $ded = [math]::Round(((Get-Counter "\GPU Process Memory(pid_$($p.Id)_*)\Dedicated Usage" -ErrorAction Stop).CounterSamples | Measure-Object CookedValue -Sum).Sum / 1MB) } catch { }
                $over = $ded - $vram
                $spillLines = @(Select-String -Path $log -Pattern 'memory_guard: spill' -ErrorAction SilentlyContinue).Count
                Write-Host "warm   $([math]::Round($sw.Elapsed.TotalSeconds, 1)) s; VRAM $vram MiB (ceiling $Ceiling), server process $ded MiB dedicated"
                if ($over -gt 64 -or $spillLines -gt 0) {
                    Write-Host "SPILL  about $([math]::Max($over, 0)) MiB of the server's GPU memory are in system RAM: everything will be slow. Stop the server and start again with a larger KINTSUGI_MARGIN (now $Margin) or fewer KINTSUGI_CELLS (now $Cells)."
                }
                Write-Host "ready  http://127.0.0.1:$Port/v1"; exit 0
            }
            if ($p.HasExited) { $m = Get-NoGpuMessage $log; if ($m) { Stop-NoGpu $m }
                Start-Sleep -Milliseconds 500   # the redirected log files are complete once the process is gone
                if (-not (Test-NotStarted) -and (Test-MtpRetry 'the server stopped during the load')) { continue launch }
                Stop-LoadFailed }
            if ($lp -and $lp.HasExited) { Write-Host "layer stopped: see $layerLog"; exit 1 }
            Start-Sleep 2
        }
        Write-Host "not ready after 300 s: see $log"; exit 1
    }
    break
}
