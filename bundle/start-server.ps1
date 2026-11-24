# Kintsugi-27B on one 12 GB NVIDIA card (RTX 20 / 30 / 40 / 50, H100): 262,144-token window, tiered K/V cache, MTP + lookup drafting.
#
#   powershell -ExecutionPolicy Bypass -File start-server.ps1 [-Port 8080] [-Wait] [-DryRun]
#
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
#   KINTSUGI_READ          fast (default: int8 read of long prompts) | exact
#   KINTSUGI_HEAD2         two-stage output head: exact candidates for each row (default 512; 0 = off)
#   KINTSUGI_HOST          address clients connect to (default 127.0.0.1 = this computer only; 0.0.0.0 = your network:
#                          then set KINTSUGI_KEY_FILE too, or anyone on the network can use your GPU)
#   KINTSUGI_KEY_FILE      API key file; with it, clients must send "Authorization: Bearer <key>"
#   KINTSUGI_LAYER         1 (default) = saved conversations through the layer (needs Python); 0 = plain server
#   KINTSUGI_SLOT_SAVE_PATH  folder of the saved conversations (default slots\ in this folder)
#   KINTSUGI_SLOT_CACHE_GB   size limit of that folder in GB (default 20; the oldest files are removed past it)
#   KINTSUGI_UB            micro-batch for prompt reads (default 1024; 512 saves about 146 MiB and reads 8-13 % slower)
#   KINTSUGI_ALIAS         the model name that /v1/models lists and replies carry (default kintsugi; an --alias in
#                          KINTSUGI_EXTRA wins)
#   KINTSUGI_EXTRA         more llama-server arguments
#   KINTSUGI_CHAT_TEMPLATE chat template file (default templates\kintsugi.jinja: the model's template, except that a
#                          system message after the first turn, as Claude Code sends, is accepted); "model" = the
#                          template inside the model file; any other value = the path of your own template
param([int]$Port = 8080, [switch]$Wait, [switch]$DryRun)
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
# server log, exit 1.
function Stop-LoadFailed {
    if ($lp -and -not $lp.HasExited) { Stop-Process -Id $lp.Id -Force -ErrorAction SilentlyContinue; try { $lp.WaitForExit(10000) | Out-Null } catch { } }
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

# ---- VRAM line: how many K/V cells fit in VRAM; the rest of the window lives in system RAM -----------------------------
$HalfBytes = @{ 'q8_0' = 17408; 'q4_0' = 9216; 'q3_0' = 7168 }   # bytes of K (or of V) per cell, all attention layers
if (-not $HalfBytes.ContainsKey($Kv)) { throw "KINTSUGI_KV must be one of: $($HalfBytes.Keys -join ', ')" }
$Cell = 2 * $HalfBytes[$Kv]
$q = (& nvidia-smi --query-gpu=memory.total,memory.used,display_active --format=csv,noheader,nounits | Select-Object -First 1).Split(',')
$TotalMiB = [int]$q[0]; $UsedMiB = [int]$q[1]; $Headless = $q[2].Trim() -ne 'Enabled'
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
        if ($u -le $script:UsedMiB + 64) { break }
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
    $ImgMax = [int](Pick 'IMAGE_MAX_TOKENS' 2048)
    # Measured for the 629,247,008-byte Q8_0 projector at a 2048-token cap; another file moves it by the difference.
    $VisionMiB = if ($Vision -eq 'gpu') { 852 + [int][math]::Round(((FileLength $Mmproj) - 629247008) / 1MB) } elseif ($Vision -eq 'cpu') { 0 } else { 0 }
    $Fixed = 9473 + $HeadMiB - $Head2MiB + $DrafterMiB + $VisionMiB
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
    $env:LLAMA_ARG_CHAT_TEMPLATE_KWARGS = if ($Think -eq 'off') { '{"reasoning_effort":"medium","enable_thinking":false}' } else { '{"reasoning_effort":"medium"}' }
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
              '-ctk', $Kv, '-ctv', $Kv, '--kv-vram-cells', "$Cells", '--kq-mask-packed', '--cache-ram', '4096', '--backend-sampling',
              '--reasoning-budget', '20480', '-n', '24576', '--reasoning-budget-message', '"Now produce the complete answer."',
              '--reasoning-effort-allow', 'medium', '--reasoning-effort-fallback', 'medium', '--reasoning-max-tokens-floor', '24576',
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
    Write-Host "kv     $Kv, tiered: cells 0..$Cells in VRAM, $Cells..$Ctx in system RAM (free $($TotalMiB - $UsedMiB) MiB, reserved $ReservedMiB, ceiling $Ceiling, fixed $Fixed, growth $Growth, margin $Margin ($MarginWhy))"
    Write-Host "template $(if ($Template -eq 'model') { 'the model file''s own' } else { $Template })"
    if ($Head2 -gt 0) { Write-Host "head   two-stage: first stage q4_0 in VRAM, $Head2 exact candidates for each row from system RAM" }
    Write-Host "spec   $(if ($Mode -eq 'off') { 'off' } elseif ($Mode -eq 'dflash') { "DFlash2 draft 7 ($(Split-Path $Drafter -Leaf))" } else { "MTP draft $Draft + lookup" }); thinking $Think; read $($env:GGML_ESCHA_PREFILL)"
    if ($DraftNote) { Write-Host $DraftNote }
    Write-Host "vision $(if ($Vision -eq 'off') { 'off' } else { "$Vision ($(Split-Path $Mmproj -Leaf)), up to $ImgMax tokens per image" })"
    if ($DryRun) { Write-Host ("args   " + ($SrvArgs -join ' ')); exit 0 }

    $logs = Join-Path $Root 'logs'; New-Item -ItemType Directory -Force $logs | Out-Null
    $log = Join-Path $logs "server-$Port.log"
    $p = Start-Process -FilePath (Join-Path $Bin 'llama-server.exe') -ArgumentList ($SrvArgs -join ' ') -WorkingDirectory $Bin `
        -RedirectStandardOutput ($log + '.out') -RedirectStandardError $log -WindowStyle Hidden -PassThru
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
                if (Test-MtpRetry 'the server stopped during the load') { continue launch }
                Stop-LoadFailed }
            if ($lp -and $lp.HasExited) { Write-Host "layer stopped: see $layerLog"; exit 1 }
            Start-Sleep 2
        }
        Write-Host "not ready after 300 s: see $log"; exit 1
    }
    break
}
