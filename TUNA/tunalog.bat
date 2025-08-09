@echo off
setlocal

set "scriptDir=%~dp0"

if "%1"=="--version" (
    for /f "tokens=2 delims== " %%i in ('findstr /r /c:__version__ "%scriptDir%__init__.py"') do (
        set "version=%%i"
        call :stripQuotes
    )
) else (
    python "%scriptDir%tuna.py" %* > tuna-calculation.log
)

endlocal
goto :eof

:stripQuotes

set "version=%version:"=%"
echo %version%
goto :eof
