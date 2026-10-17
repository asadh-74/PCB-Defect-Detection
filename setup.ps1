param(
    [string]$TorchIndexUrl = "https://download.pytorch.org/whl/cpu"
)
$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot
function Invoke-Checked {
    param([string]$Program, [string[]]$Arguments)
    & $Program @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Setup stopped: $Program exited with $LASTEXITCODE" }
}
Invoke-Checked -Program "python" -Arguments @("-c", "import sys; assert sys.version_info >= (3, 10), 'Python 3.10+ required; Python 3.12 recommended'")
if (-not (Test-Path -LiteralPath ".venv\Scripts\python.exe")) {
    Invoke-Checked -Program "python" -Arguments @("-m", "venv", ".venv")
}
$ProjectPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
Invoke-Checked -Program $ProjectPython -Arguments @("-m", "pip", "install", "--upgrade", "pip")
Invoke-Checked -Program $ProjectPython -Arguments @("-m", "pip", "install", "torch", "torchvision", "--index-url", $TorchIndexUrl)
Invoke-Checked -Program $ProjectPython -Arguments @("-m", "pip", "install", "-r", "requirements.txt")
Invoke-Checked -Program $ProjectPython -Arguments @("-m", "pytest", "-q", "tests")
Write-Host "Setup and tests completed. See LOCAL_SETUP.md for training and the app."
