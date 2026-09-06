@echo off
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0tests.ps1"
exit /b %ERRORLEVEL%