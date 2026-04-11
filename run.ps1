param(
    [Parameter(Position = 0)]
    [ValidateSet("install", "test", "run", "clean", "help")]
    [string]$Task = "help"
)

$ErrorActionPreference = "Stop"

function Invoke-Install {
    Write-Host "Installing dependencies..."
    pip install -r requirements.txt
    pip install -e .
}

function Invoke-Test {
    Write-Host "Running tests..."
    pytest -q
}

function Invoke-Run {
    Write-Host "Running example..."
    python examples/01_linear_regression.py
}

function Invoke-Clean {
    Write-Host "Cleaning cache and build artifacts..."

    if (Test-Path ".pytest_cache") {
        Remove-Item -Recurse -Force ".pytest_cache"
    }

    Get-ChildItem -Recurse -Directory -Filter "__pycache__" -ErrorAction SilentlyContinue |
        Remove-Item -Recurse -Force -ErrorAction SilentlyContinue

    Get-ChildItem -Recurse -Directory -Filter "*.egg-info" -ErrorAction SilentlyContinue |
        Remove-Item -Recurse -Force -ErrorAction SilentlyContinue

    if (Test-Path "build") {
        Remove-Item -Recurse -Force "build"
    }

    if (Test-Path "dist") {
        Remove-Item -Recurse -Force "dist"
    }
}

function Show-Help {
    Write-Host ""
    Write-Host "Usage:"
    Write-Host "  .\run.ps1 install"
    Write-Host "  .\run.ps1 test"
    Write-Host "  .\run.ps1 run"
    Write-Host "  .\run.ps1 clean"
    Write-Host ""
}

switch ($Task) {
    "install" { Invoke-Install }
    "test"    { Invoke-Test }
    "run"     { Invoke-Run }
    "clean"   { Invoke-Clean }
    "help"    { Show-Help }
}
