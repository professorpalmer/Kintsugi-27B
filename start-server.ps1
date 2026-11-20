# Kintsugi-27B on one 12 GB NVIDIA card (RTX 20 / 30 / 40): 262,144-token window, tiered K/V cache, MTP + lookup drafting.
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
#   KINTSUGI_DRAFT         draft size of the MTP drafter (default 3; 0 = no drafting)
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
#   KINTSUGI_EXTRA         more llama-server arguments
param([int]$Port = 8080, [switch]$Wait, [switch]$DryRun)
$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
function Pick($name, $default) {
    foreach ($n in @("KINTSUGI_$name", "ESCHA_$name")) {
        $v = [Environment]::GetEnvironmentVariable($n); if ($v) { return $v }
    }
    $default
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

# ---- VRAM line: how many K/V cells fit in VRAM; the rest of the window lives in system RAM -----------------------------
$HalfBytes = @{ 'q8_0' = 17408; 'q4_0' = 9216; 'q3_0' = 7168 }   # bytes of K (or of V) per cell, all attention layers
if (-not $HalfBytes.ContainsKey($Kv)) { throw "KINTSUGI_KV must be one of: $($HalfBytes.Keys -join ', ')" }
$Cell = 2 * $HalfBytes[$Kv]
$q = (& nvidia-smi --query-gpu=memory.total,memory.used,display_active --format=csv,noheader,nounits | Select-Object -First 1).Split(',')
$TotalMiB = [int]$q[0]; $UsedMiB = [int]$q[1]; $Headless = $q[2].Trim() -ne 'Enabled'
if ($TotalMiB -lt 11500) { Write-Host "note   this card has $TotalMiB MiB; Kintsugi-27B is made for 12 GB cards and may not load" }
# Margin under the card's limit, on top of the measured growth (a display on the card moves by itself, so more margin).
$Margin = [int](Pick 'MARGIN' $(if ($Headless) { 250 } else { 600 }))
# Fixed footprint at load without the K/V cells (measured on an RTX 4070 with the 10,546,686,688-byte file); a model file
# of another size moves it by the difference.
$HeadMiB = [int][math]::Round(((Get-Item $Model).Length - 10546686688) / 1MB)
# Two-stage output head (default 512 candidates): the exact Q8_0 head goes to system RAM, a small first stage stays in
# VRAM (606 MiB saved at load). The text is the same as with the full head on our tests.
$Head2 = [int](Pick 'HEAD2' 512)
$Head2MiB = if ($Head2 -gt 0) { 606 } else { 0 }
$Fixed = 9473 + $HeadMiB - $Head2MiB
# The first request adds about 222 MiB once (cuBLAS, kernel code at first use); the warm-up request pays it at start.
$Growth = 222
# Windows keeps part of the card: past about 97.7 % of it the driver moves memory to system RAM without an error and
# everything slows down (the "spill" below). KINTSUGI_CEILING_MIB overrides.
$Ceiling = [int](Pick 'CEILING_MIB' ([math]::Floor($TotalMiB * 0.977)))
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

# ---- Drafting: the model's own MTP block (draft 3) + lookup drafting out of the prompt -----------------------------------
$Draft = [int](Pick 'DRAFT' 3)
$Spec = @()
if ($Draft -gt 0) {
    $Spec = @('--spec-type', 'ngram-mod,draft-mtp', '--spec-draft-n-max', "$Draft", '--spec-draft-n-max-tail', "$([math]::Max(4, $Draft))",
              '-ctkd', 'q8_0', '-ctvd', 'q8_0', '--spec-lookup-n-max', '32', '--spec-draft-window', '16384')
}

# ---- Engine switches -------------------------------------------------------------------------------------------------
$env:GGML_CUDA_BATCH_INVARIANT = '1'          # a token's output does not depend on the batch it runs in
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
$SrvArgs = @('-m', ('"' + $Model + '"'), '-ngl', '99', '-c', "$Ctx", '-fa', 'on', '-np', '1', '-ub', "$Ub", '--fit', 'off', '--jinja',
          '-ctk', $Kv, '-ctv', $Kv, '--kv-vram-cells', "$Cells", '--kq-mask-packed', '--cache-ram', '4096', '--backend-sampling',
          '--reasoning-budget', '20480', '-n', '24576', '--reasoning-budget-message', '"Now produce the complete answer."',
          '--reasoning-effort-allow', 'medium', '--reasoning-effort-fallback', 'medium', '--reasoning-max-tokens-floor', '24576',
          '--host', $(if ($UseLayer) { '127.0.0.1' } else { $ListenHost }), '--port', "$ServerPort") + $Spec
if ($KeyFile) { $SrvArgs += @('--api-key-file', ('"' + $KeyFile + '"')) }
if ($SlotDir) { $SrvArgs += @('--slot-save-path', ('"' + $SlotDir + '"')) }
$Extra = Pick 'EXTRA' ''
if ($Extra) { $SrvArgs += $Extra.Split(' ') }

Write-Host "Kintsugi-27B"
Write-Host "model  $(Split-Path $Model -Leaf)"
Write-Host "listen $ListenHost`:$Port$(if ($UseLayer) { " (layer; llama-server on 127.0.0.1:$ServerPort)" })$(if ($KeyFile) { ", API key on" })"
Write-Host "kv     $Kv, tiered: cells 0..$Cells in VRAM, $Cells..$Ctx in system RAM (free $($TotalMiB - $UsedMiB) MiB, ceiling $Ceiling, fixed $Fixed, growth $Growth, margin $Margin$(if ($Headless) { ', no display on this card' }))"
if ($Head2 -gt 0) { Write-Host "head   two-stage: first stage q4_0 in VRAM, $Head2 exact candidates for each row from system RAM" }
Write-Host "spec   $(if ($Draft -le 0) { 'off' } else { "MTP draft $Draft + lookup" }); thinking $Think; read $($env:GGML_ESCHA_PREFILL)"
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
if ($Wait) {
    for ($i = 0; $i -lt 150; $i++) {
        $up = $false
        try { $up = (Invoke-WebRequest -UseBasicParsing -TimeoutSec 3 "http://127.0.0.1:$Port/health").StatusCode -eq 200 } catch { }
        if ($up) {
            # Warm-up request (about 1,100 tokens in, 8 out): the engine takes its first-use memory now, so the VRAM printed
            # below is the peak, and a shortage shows here and not later.
            $text = (1..110 | ForEach-Object { "Entry $_ of the log has nothing to report." }) -join ' '
            $body = @{ model = 'warmup'; max_tokens = 8; temperature = 0; cache_prompt = $false
                       chat_template_kwargs = @{ enable_thinking = $false }
                       messages = @(@{ role = 'user'; content = "$text Say OK." }) } | ConvertTo-Json -Depth 5
            $hdr = @{}
            if ($KeyFile) { $hdr['Authorization'] = 'Bearer ' + (Get-Content -Raw $KeyFile).Trim() }
            $sw = [Diagnostics.Stopwatch]::StartNew()
            try { Invoke-WebRequest -UseBasicParsing -TimeoutSec 600 -Method Post -ContentType 'application/json' -Headers $hdr -Body $body "http://127.0.0.1:$Port/v1/chat/completions" | Out-Null }
            catch { Write-Host "note   the warm-up request failed: $($_.Exception.Message)" }
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
        if ($p.HasExited) { Write-Host "server stopped during load: see $log"; exit 1 }
        if ($lp -and $lp.HasExited) { Write-Host "layer stopped: see $layerLog"; exit 1 }
        Start-Sleep 2
    }
    Write-Host "not ready after 300 s: see $log"; exit 1
}
