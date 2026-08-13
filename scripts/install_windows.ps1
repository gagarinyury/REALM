<#
.SYNOPSIS
    Install REALM and its engine on a clean Windows machine, natively -- no Docker.

.DESCRIPTION
    Brings up the stack this branch was developed against:

        Miniconda env "behavior" (Python 3.11)
          BEHAVIOR-1K 3.9.1  -- OmniGibson + BDDL + Isaac Sim 5.1 + dataset
          REALM (this fork)  -- used from PYTHONPATH, not pip-installed
        WSL2 (separate)
          openpi policy server -- serves pi0-FAST over websocket on port 8000

    The simulator runs on Windows and the policy server in WSL2, on the same GPU.
    That split is not a preference: Isaac Sim cannot render under WSL2 at all, and
    openpi has no Windows build. Section 5 of NATIVE_WINDOWS.md covers the split.

    Stages run in order and can be run one at a time with -Stage. Each is idempotent:
    re-running a completed stage detects the existing state and skips it.

.PARAMETER Stage
    all (default), check, behavior, vulkan, realm, deps, patches, verify.

.PARAMETER Root
    Install root. Default C:\thesis-yahor-pachkouski.

.PARAMETER Branch
    REALM branch to check out. Default win/dd091fd -- upstream's own og391 port plus
    the Windows fixes. Use main for the other route (repaired asset, stock engine).

.EXAMPLE
    .\install_windows.ps1 -Stage check
    .\install_windows.ps1

.NOTES
    NOT AUTOMATED, AND WHY
      The openpi side needs a WSL2 distro, a uv install and a ~10 GB checkpoint pulled
      from a Google Storage bucket. It is left as printed instructions at the end: it
      needs its own credentials and its own disk, and half of it is interactive.

    DISK
      The dataset is 141485 small files. Unpack it on a LOCAL disk. Over a network
      filesystem we measured ~31 files/s and then a hang inside FUSE request_wait_answer;
      the same machine's local disk did 150-320 files/s.

    LICENCES
      BEHAVIOR-1K assets are under an EULA and ship encrypted -- the installer prompts.
      The -Accept* switches below pass those prompts through; read them once first.

    WHAT THIS SCRIPT DELIBERATELY DOES NOT DO
      It does not reimplement BEHAVIOR-1K's own installer. That one already handles the
      conda env, the CUDA-matched torch wheels, Isaac Sim and the dataset download; we
      call it with the right flags and then fix the four things it leaves broken on
      Windows (stages vulkan / deps / patches).
#>
param(
    [ValidateSet("all","check","behavior","vulkan","realm","deps","patches","verify")]
    [string]$Stage = "all",
    [string]$Root = "C:\thesis-yahor-pachkouski",
    [string]$Branch = "win/dd091fd",
    [string]$ForkUrl = "https://github.com/gagarinyury/REALM.git",
    [string]$EnvName = "behavior",
    [string]$CondaRoot = "C:\Miniconda3"
)

$ErrorActionPreference = "Stop"
$repos  = Join-Path $Root "repos"
$b1k    = Join-Path $repos "BEHAVIOR-1K-main"
$realm  = Join-Path $repos "REALM"
$python = Join-Path $CondaRoot "envs\$EnvName\python.exe"
$pip    = Join-Path $CondaRoot "envs\$EnvName\Scripts\pip.exe"

function Say($msg, $color = "White") { Write-Host $msg -ForegroundColor $color }
function Step($n, $msg) { Write-Host ""; Write-Host "=== $n. $msg" -ForegroundColor Cyan }
function Ok($msg)   { Write-Host "  OK   $msg" -ForegroundColor Green }
function Warn($msg) { Write-Host "  WARN $msg" -ForegroundColor Yellow }
function Die($msg)  { Write-Host "  FAIL $msg" -ForegroundColor Red; exit 1 }

$runAll = ($Stage -eq "all")

# ---------------------------------------------------------------- check
if ($runAll -or $Stage -eq "check") {
    Step 1 "Prerequisites"

    $gpu = (nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>$null)
    if (-not $gpu) { Die "nvidia-smi not found -- install the NVIDIA driver first." }
    Ok "GPU: $gpu"
    # The simulator needs roughly 6 GB and the policy server another 9.5 GB. On a 16 GB
    # card they coexist only because the policy is capped at 0.47 of the card; below 16 GB
    # you will be swapping one out to run the other.
    $vramMb = [int](nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits)
    if ($vramMb -lt 15000) { Warn "under 16 GB of VRAM -- simulator and policy server will not fit together" }

    foreach ($tool in @("git","wsl")) {
        if (-not (Get-Command $tool -ErrorAction SilentlyContinue)) { Die "$tool not on PATH" }
        Ok "$tool present"
    }

    # conda is often absent from PATH in a non-interactive session (ssh, scheduled task)
    # while being perfectly installed -- its PATH entry is added by the shell hook that
    # only an interactive profile runs. Look for the executable itself before giving up.
    $condaExe = (Get-Command conda -ErrorAction SilentlyContinue).Source
    if (-not $condaExe) {
        foreach ($cand in @("$CondaRoot\Scripts\conda.exe", "$CondaRoot\condabin\conda.bat",
                            "$env:USERPROFILE\miniconda3\Scripts\conda.exe",
                            "$env:USERPROFILE\anaconda3\Scripts\conda.exe")) {
            if (Test-Path $cand) { $condaExe = $cand; break }
        }
    }
    if (-not $condaExe) { Die "conda not found on PATH nor under $CondaRoot -- install Miniconda first" }
    Ok "conda: $condaExe"

    # Git for Windows defaults to core.autocrlf=true, which rewrites realm/misc/*.patch
    # as CRLF on checkout. git apply then compares context lines byte-for-byte against
    # LF-terminated engine sources, finds nothing, and reports only "patch does not
    # apply" -- which reads like a version mismatch. The repo carries a .gitattributes
    # pinning *.patch to LF, so a fresh clone is safe; an older clone may not be.
    $autocrlf = (git config --global core.autocrlf) 2>$null
    if ($autocrlf -eq "true") { Warn "core.autocrlf=true globally -- fine for a fresh clone (.gitattributes pins *.patch to LF), but check old clones" }

    $free = [math]::Round((Get-PSDrive ($Root.Substring(0,1))).Free / 1GB)
    Ok "free disk on $($Root.Substring(0,1)): $free GB"
    if ($free -lt 120) { Warn "under 120 GB free -- dataset, Isaac Sim and the checkpoint together need about that" }

    if ($Stage -eq "check") { Say ""; Say "prerequisites checked; nothing installed." ; exit 0 }
}

# ---------------------------------------------------------------- behavior
if ($runAll -or $Stage -eq "behavior") {
    Step 2 "BEHAVIOR-1K (OmniGibson 3.9.1 + BDDL + Isaac Sim 5.1 + dataset)"

    New-Item -ItemType Directory -Force -Path $repos | Out-Null
    if (-not (Test-Path $b1k)) {
        Say "  cloning BEHAVIOR-1K..."
        git clone --depth 1 https://github.com/StanfordVL/BEHAVIOR-1K.git $b1k
    } else { Ok "already cloned" }

    if (Test-Path $python) {
        Ok "conda env '$EnvName' exists -- skipping installer"
    } else {
        # Their installer, not ours: it creates the env, picks the torch wheel matching
        # the local CUDA, installs Isaac Sim and downloads the dataset. Reimplementing it
        # would mean re-deriving all of that by hand and going stale on the next release.
        Say "  running BEHAVIOR-1K's own setup.ps1 (this takes a long time)"
        Warn "it will ask you to accept the conda ToS, the NVIDIA EULA and the dataset ToS"
        Push-Location $b1k
        try {
            & powershell -ExecutionPolicy Bypass -File .\setup.ps1 `
                -NewEnv -OmniGibson -BDDL -Dataset `
                -AcceptCondaTos -AcceptNvidiaEula -AcceptDatasetTos
            if ($LASTEXITCODE -ne 0) { Die "setup.ps1 exited with $LASTEXITCODE" }
        } finally { Pop-Location }
    }

    $ds = Join-Path $b1k "datasets\behavior-1k-assets"
    if (Test-Path $ds) {
        $n = (Get-ChildItem (Join-Path $ds "objects") -Directory -ErrorAction SilentlyContinue).Count
        Ok "dataset present, $n object categories"
    } else {
        # The archive unpacks WITHOUT a top-level directory while the engine expects
        # {DATA_PATH}/behavior-1k-assets/ -- if the download went somewhere else, this is why.
        Die "dataset missing at $ds"
    }
}

# ---------------------------------------------------------------- vulkan
if ($runAll -or $Stage -eq "vulkan") {
    Step 3 "Vulkan off in the Isaac Sim kit file"

    # omnigibson_5_1_0.kit force-enables vulkan unconditionally. Isaac Sim's Windows build
    # defaults it OFF -- the comment in that same file says so -- and forcing it on crashes
    # at startup. This is a packaging bug in the engine, so it has to be patched in the
    # installed engine, not in this repo.
    $kits = Get-ChildItem (Join-Path $b1k "OmniGibson") -Filter "*.kit" -Recurse -ErrorAction SilentlyContinue
    if (-not $kits) { Die "no .kit files found under OmniGibson" }
    foreach ($kit in $kits) {
        $txt = Get-Content $kit.FullName -Raw
        if ($txt -match "(?m)^\s*vulkan\s*=\s*false") { Ok "$($kit.Name): already false"; continue }
        if ($txt -match "(?m)^\s*vulkan\s*=\s*true") {
            Copy-Item $kit.FullName "$($kit.FullName).bak-preinstall" -Force
            ($txt -replace "(?m)^(\s*)vulkan\s*=\s*true", '$1vulkan = false') | Set-Content $kit.FullName -NoNewline
            Ok "$($kit.Name): set to false (backup .bak-preinstall)"
        } else { Warn "$($kit.Name): no vulkan line -- check manually" }
    }
}

# ---------------------------------------------------------------- realm
if ($runAll -or $Stage -eq "realm") {
    Step 4 "REALM fork, branch $Branch"

    if (-not (Test-Path $realm)) {
        git clone $ForkUrl $realm
        Push-Location $realm
        try { git checkout $Branch } finally { Pop-Location }
    } else {
        Push-Location $realm
        try {
            $cur = git branch --show-current
            if ($cur -ne $Branch) { Warn "on branch '$cur', expected '$Branch' -- not switching automatically" }
            else { Ok "already on $Branch" }
        } finally { Pop-Location }
    }

    # Verify the patches survived checkout as LF. This is the failure that costs the most
    # time to diagnose, because the error message points at the wrong thing entirely.
    Get-ChildItem (Join-Path $realm "realm\misc\*.patch") -ErrorAction SilentlyContinue | ForEach-Object {
        $bytes = [System.IO.File]::ReadAllBytes($_.FullName)
        $crlf = 0; for ($i = 1; $i -lt $bytes.Length; $i++) { if ($bytes[$i] -eq 10 -and $bytes[$i-1] -eq 13) { $crlf++ } }
        if ($crlf -gt 0) { Die "$($_.Name) has $crlf CRLF line endings -- git rewrote it; re-clone with .gitattributes in place" }
        Ok "$($_.Name): LF, patchable"
    }
}

# ---------------------------------------------------------------- deps
if ($runAll -or $Stage -eq "deps") {
    Step 5 "REALM dependencies"

    # REALM is not pip-installed: it is imported from PYTHONPATH. Only its bundled
    # websocket client is a real package.
    & $pip install -q -e (Join-Path $realm "packages\openpi-client")
    Ok "openpi-client installed"

    # Pins from .docker/realm_og391.Dockerfile + og391-constraints.txt. numpy above 1.26
    # breaks the mujoco/dm_control chain that replaces dm_robotics on Windows.
    $numpy = (& $python -c "import numpy; print(numpy.__version__)")
    if ($numpy -ne "1.26.0") { Warn "numpy is $numpy, expected 1.26.0" } else { Ok "numpy 1.26.0" }

    # dm_robotics ships manylinux wheels only -- no Windows wheel, no sdist -- so this
    # branch carries a replacement IK solver written against dm_control.mjcf + osqp.
    foreach ($mod in @("mujoco","dm_control","osqp","torch","gymnasium")) {
        $v = (& $python -c "import $mod, sys; print(getattr($mod,'__version__','?'))" 2>$null)
        if ($LASTEXITCODE -ne 0) { Die "python module '$mod' missing" }
        Ok "$mod $v"
    }
}

# ---------------------------------------------------------------- patches
if ($runAll -or $Stage -eq "patches") {
    Step 6 "Engine patches"

    # REALM drives OmniGibson in ways its asserts do not expect: it places a robot while
    # the simulation is stopped, and its DROID asset carries a root-link offset from the
    # entity prim. The patches relax exactly those checks. Each is applied against the
    # installed engine, with a backup, and skipped if already applied.
    $og = Join-Path $b1k "OmniGibson"
    $patchDir = Join-Path $realm "realm\misc"
    $patches = Get-ChildItem (Join-Path $patchDir "*.patch") -ErrorAction SilentlyContinue
    if (-not $patches) { Warn "no patches on this branch"; }

    Push-Location $og
    try {
        foreach ($p in $patches) {
            $check = (git apply --check --reverse $p.FullName 2>&1)
            if ($LASTEXITCODE -eq 0) { Ok "$($p.Name): already applied"; continue }
            $check = (git apply --check $p.FullName 2>&1)
            if ($LASTEXITCODE -ne 0) { Die "$($p.Name) does not apply: $check" }
            git apply $p.FullName
            Ok "$($p.Name): applied"
        }
    } finally { Pop-Location }

    # One more relaxation lives outside the patch files because it was found later:
    # entity_prim.py asserts the entity prim and the root link share a pose, which the
    # DROID asset violates whenever it is placed with the simulation stopped. Needed for
    # the IMPACT bench, where the robot stands on a countertop.
    $ep = Join-Path $og "omnigibson\prims\entity_prim.py"
    $txt = Get-Content $ep -Raw
    if ($txt -match "REALM: relaxed") { Ok "pose assert already relaxed" }
    else { Warn "pose assert NOT relaxed in entity_prim.py -- the IMPACT bench will fail with 'Position mismatch between entity prim and root link'; see NATIVE_WINDOWS.md" }
}

# ---------------------------------------------------------------- verify
if ($runAll -or $Stage -eq "verify") {
    Step 7 "Verify"

    $env:PYTHONPATH = $realm
    $env:PYTHONIOENCODING = "utf-8"
    $env:OMNIGIBSON_HEADLESS = "1"
    # Without this, importing isaacsim prompts for the Omniverse EULA on stdin and dies with
    # "Unable to bootstrap inner kit kernel: EOF when reading a line" in any non-interactive
    # session -- ssh, a scheduled task, CI. The variable is read in
    # site-packages/isaacsim/kit_app.py:19, which accepts y / yes / 1.
    $env:OMNI_KIT_ACCEPT_EULA = "YES"

    & $python -c "import omnigibson, isaacsim; print('omnigibson', omnigibson.__version__)"
    if ($LASTEXITCODE -ne 0) { Die "omnigibson does not import" }
    & $python -c "import realm.eval; print('realm imports')"
    if ($LASTEXITCODE -ne 0) { Die "realm does not import -- PYTHONPATH or dependencies" }
    Ok "imports fine"

    Say ""
    Say "Windows side is ready. The policy server is next, and it lives in WSL2:" -color Cyan
    Say ""
    Say "  wsl --install -d Ubuntu-22.04"
    Say "  wsl -d Ubuntu-22.04 -- bash -lc 'curl -LsSf https://astral.sh/uv/install.sh | sh'"
    Say "  wsl -d Ubuntu-22.04 -- bash -lc 'git clone https://github.com/Physical-Intelligence/openpi /root/openpi'"
    Say ""
    Say "  # then start it -- 0.47 is measured, not chosen: 0.45 dies with RESOURCE_EXHAUSTED"
    Say "  wsl -d Ubuntu-22.04 --cd /root/openpi -- bash -lc 'XLA_PYTHON_CLIENT_MEM_FRACTION=0.47 uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi0_fast_droid_jointpos_polaris --policy.dir=gs://openpi-assets/checkpoints/pi0_fast_droid_jointpos'"
    Say ""
    Say "  # the checkpoint is ~10 GB and downloads on first run. A partial download reports"
    Say "  # success and then fails at inference -- NATIVE_WINDOWS.md section 6."
    Say ""
    Say "Then run a task:" -color Cyan
    Say "  .\scripts\run_task_windows.ps1 -TaskId 6 -TaskCfgPath IMPACT/stack_cubes/default.yaml -Experiment smoke"
    Say ""
    Say "Expect task_progression 1.0 on that one; it is the fastest task in the bench."
}
