<#
.SYNOPSIS
    Run one REALM task on native Windows and report the result.

.DESCRIPTION
    Wraps examples/02_evaluate.py with everything a Windows run needs, and waits for the
    result the way that actually works. Every guard here exists because its absence cost
    us a debugging session -- see NOTES.

.PARAMETER TaskId
    Index into SUPPORTED_TASKS (realm/eval.py:20). Ignored when -TaskCfgPath is given,
    except that it still names the log files.
        0 put_green_block_into_bowl   5 pick_water_bottle
        1 put_banana_into_box         6 stack_cubes
        2 rotate_marker               7 push_switch
        3 rotate_mug                  8 open_drawer
        4 pick_spoon                  9 close_drawer

.PARAMETER TaskCfgPath
    Config relative to realm/config/tasks, e.g. IMPACT/pick_spoon/bbox_fixed.yaml.
    Pass it BARE -- no surrounding quotes. Quotes end up inside the path and the run
    dies with FileNotFoundError on "'IMPACT/pick_spoon/default.yaml'".

.PARAMETER RenderingMode
    rt (default), r, or pt. Use rt. In r the policy's gripper channel peaks at 0.487
    against the 0.5 threshold in eval.py:230, so the gripper never closes and every
    grasp task caps out at REACH.

.PARAMETER Experiment
    Log directory name under logs/. Use a fresh one per hypothesis so old numbers survive.

.EXAMPLE
    .\run_task_windows.ps1 -TaskId 4 -TaskCfgPath IMPACT/pick_spoon/bbox_fixed.yaml -Experiment impact_bbox

.NOTES
    WAITING FOR THE RESULT
      Poll for reports/<task>_<perturbation>.csv, never for the process. The Python
      process disappears from the task list up to two minutes before the report is
      written, and the GPU holds its memory even longer. We once declared a healthy run
      dead on that signal and spent an evening hunting a cause that did not exist.

    ENCODING
      PYTHONIOENCODING=utf-8 is mandatory: without it any non-ASCII byte printed by a
      script raises UnicodeEncodeError under cp1252. The same applies in reverse -- REALM
      reads YAML with a bare open(path, "r"), so a non-ASCII character anywhere in a task
      config makes the run HANG with the CPU near zero, no traceback.

    SILENT DEATHS
      A run whose log stops at "app ready" with no error has one of four causes, all seen:
      no_rendering=True; a non-ASCII byte in a task YAML; common_freq not matching the
      simulation step frequency; or not enough VRAM.

    VRAM
      The policy server takes ~9.5 GB of the 16 GB card. The boundary was measured, not
      guessed: XLA_PYTHON_CLIENT_MEM_FRACTION=0.47 works, 0.45 dies with RESOURCE_EXHAUSTED.

    WINDOWS
      Isaac Sim spawns visible windows from its subprocesses while extensions load. They
      are hidden for the first two minutes -- somebody is working at this machine.
#>
param(
    [int]$TaskId = 0,
    [string]$TaskCfgPath = "",
    [string]$Experiment = "win_run",
    [string]$RenderingMode = "rt",
    [int]$Repeats = 1,
    [int]$MaxSteps = 800,
    [int]$PerturbationId = 0,
    [string]$Robot = "DROID",
    [string]$Model = "pi0_fast_droid_jointpos",
    [int]$Port = 8000,
    [string]$Repo = "C:\thesis-yahor-pachkouski\repos\REALM",
    [string]$Python = "C:\Miniconda3\envs\behavior\python.exe",
    [switch]$ExtractVideo
)

$ErrorActionPreference = "Stop"

# The policy server runs in WSL and answers on $Port. Without it the run loads the whole
# scene, then blocks on the first inference call.
$policyUp = (Test-NetConnection -ComputerName 127.0.0.1 -Port $Port -WarningAction SilentlyContinue).TcpTestSucceeded
if (-not $policyUp) {
    Write-Host "policy server is not listening on $Port -- start it first:" -ForegroundColor Yellow
    Write-Host "  wsl -d Ubuntu-22.04 --cd /root/openpi -- bash -lc 'XLA_PYTHON_CLIENT_MEM_FRACTION=0.47 uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi0_fast_droid_jointpos_polaris --policy.dir=gs://openpi-assets/checkpoints/pi0_fast_droid_jointpos'"
    exit 1
}

$env:OMNIGIBSON_HEADLESS = "1"
$env:PYTHONIOENCODING = "utf-8"
# Isaac Sim prompts for the Omniverse EULA on stdin the first time it starts in a session
# without one accepted, and in a non-interactive session that is fatal: "Unable to bootstrap
# inner kit kernel: EOF when reading a line". Read in site-packages/isaacsim/kit_app.py:19.
$env:OMNI_KIT_ACCEPT_EULA = "YES"
# python puts the script's directory on sys.path, not the working directory, so the realm
# package is not importable from examples/ without this.
$env:PYTHONPATH = $Repo

$tasks = @("put_green_block_into_bowl","put_banana_into_box","rotate_marker","rotate_mug",
           "pick_spoon","pick_water_bottle","stack_cubes","push_switch","open_drawer","close_drawer")
$perturbations = @("Default","V-AUG","V-VIEW","V-SC","V-LIGHT")

# eval.py derives the report name from the config file, not from the task: a run of
# pick_spoon/bbox_fixed.yaml lands in pick_spoon_bbox_fixed_Default.csv.
if ($TaskCfgPath -ne "") {
    $parts = $TaskCfgPath -replace "\\","/" -split "/"
    $taskName = $parts[$parts.Length - 2]
    $cfgName  = ($parts[$parts.Length - 1]) -replace "\.ya?ml$",""
    $reportStem = if ($cfgName -eq "default") { $taskName } else { "${taskName}_${cfgName}" }
} else {
    $taskName = $tasks[$TaskId]
    $reportStem = $taskName
}
$report = Join-Path $Repo "logs\$Experiment\$Model\reports\${reportStem}_$($perturbations[$PerturbationId]).csv"
$stem = "C:\thesis-yahor-pachkouski\run_${Experiment}_t${TaskId}"

if (Test-Path $report) {
    Write-Host "report already exists, delete it to re-run: $report" -ForegroundColor Yellow
    exit 1
}

Add-Type @"
using System;
using System.Runtime.InteropServices;
public class WinHide {
    [DllImport("user32.dll")] public static extern bool ShowWindow(IntPtr hWnd, int nCmdShow);
}
"@

$argList = @(
    "examples\02_evaluate.py",
    "--task_id", "$TaskId",
    "--perturbation_id", "$PerturbationId",
    "--repeats", "$Repeats",
    "--max_steps", "$MaxSteps",
    "--model_name", $Model,
    "--model_type", "openpi",
    "--port", "$Port",
    "--experiment_name", $Experiment,
    "--robot", $Robot,
    "--rendering_mode", $RenderingMode
)
if ($TaskCfgPath -ne "") { $argList += @("--task_cfg_path", $TaskCfgPath) }

Write-Host "task    : $reportStem" -ForegroundColor Cyan
Write-Host "render  : $RenderingMode, max $MaxSteps steps"
Write-Host "logs    : $stem.log / .err.log"

$started = Get-Date
$proc = Start-Process -FilePath $Python -ArgumentList $argList -WorkingDirectory $Repo `
    -WindowStyle Hidden -RedirectStandardOutput "$stem.log" -RedirectStandardError "$stem.err.log" -PassThru

$hideUntil = (Get-Date).AddMinutes(2)
while ((Get-Date) -lt $hideUntil -and -not $proc.HasExited) {
    Get-Process | Where-Object { $_.Id -eq $proc.Id -or $_.Parent.Id -eq $proc.Id } | ForEach-Object {
        if ($_.MainWindowHandle -ne 0) { [WinHide]::ShowWindow($_.MainWindowHandle, 0) | Out-Null }
    }
    Start-Sleep -Milliseconds 300
}
$proc.WaitForExit()

# The process is gone; the report may not be. Give it the grace period it actually needs.
$deadline = (Get-Date).AddMinutes(3)
while (-not (Test-Path $report) -and (Get-Date) -lt $deadline) { Start-Sleep -Seconds 5 }

$elapsed = [int]((Get-Date) - $started).TotalSeconds
if (-not (Test-Path $report)) {
    Write-Host "NO REPORT after ${elapsed}s -- last lines of stderr:" -ForegroundColor Red
    Get-Content "$stem.err.log" -Tail 12
    exit 2
}

$row = Import-Csv $report | Select-Object -First 1
Write-Host ""
Write-Host "progression  : $($row.task_progression)" -ForegroundColor Green
Write-Host "stage        : $($row.stage)   binary_SR: $($row.binary_SR)"
Write-Host "stamps       : $($row.task_progression_timestamps)"
Write-Host "collisions   : env $($row.collisions_env), self $($row.collisions_self), drops $($row.object_drops)"
Write-Host "elapsed      : ${elapsed}s"

if ($ExtractVideo) {
    $parquet = Join-Path $Repo "logs\$Experiment\$Model\videos\$reportStem.parquet"
    $outDir = "C:\thesis-yahor-pachkouski\frames-$Experiment"
    & $Python "C:\thesis-yahor-pachkouski\extract_video.py" $parquet $outDir
}
