@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo.
    echo ERRO: ambiente virtual nao encontrado.
    echo Caminho esperado: %CD%\.venv\Scripts\python.exe
    echo.
    pause
    exit /b 1
)

".venv\Scripts\python.exe" -m Tools.app.gui_menu

if errorlevel 1 (
    echo.
    echo O programa foi encerrado com erro.
    pause
)

endlocal
