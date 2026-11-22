# Entry point on Windows. Installs uv if needed, syncs the project's Python 3.12 environment, then runs the CLI.
# Usage: .\voice-clone.cmd <command> [options]   (.\voice-clone.cmd --help)
# voice-clone.cmd runs this script without changing PowerShell's execution policy.
$Root = $PSScriptRoot
$env:PATH = "$(Join-Path $HOME '.local\bin');$env:PATH"
$env:PYTHONUTF8 = "1"

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    [Console]::Error.WriteLine("Installing uv (Python environment manager) into ~\.local\bin...")
    powershell -NoProfile -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/install.ps1 | iex" | Out-Host
    if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
        [Console]::Error.WriteLine("uv could not be installed; see https://docs.astral.sh/uv/ and try again.")
        exit 1
    }
}

uv sync --project "$Root" --quiet
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& (Join-Path $Root ".venv\Scripts\voice-clone.exe") @args
exit $LASTEXITCODE
