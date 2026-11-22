@echo off
rem Entry point on Windows (cmd or PowerShell): voice-clone.cmd <command> [options]
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0voice-clone.ps1" %*
exit /b %ERRORLEVEL%
