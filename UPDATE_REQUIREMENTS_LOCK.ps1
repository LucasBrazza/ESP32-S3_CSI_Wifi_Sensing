$ErrorActionPreference = "Stop"

$python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $python)) {
    throw "Ambiente virtual não encontrado em .venv"
}

& $python -m pip install --upgrade pip
& $python -m pip install -r (Join-Path $PSScriptRoot "requirements.txt")

& $python -m pip freeze |
    Sort-Object |
    Set-Content `
        -Path (Join-Path $PSScriptRoot "requirements-lock.txt") `
        -Encoding utf8

Write-Host ""
Write-Host "requirements-lock.txt atualizado com o ambiente validado."
