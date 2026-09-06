@echo off
REM Build fastsieve (CPU + OpenCL GPU module) with MSVC.
setlocal
set VSDIR=%ProgramFiles%\Microsoft Visual Studio\2022\Community
call "%VSDIR%\Common7\Tools\VsDevCmd.bat" -arch=amd64 -host_arch=amd64 >nul 2>&1
if not exist CL\cl.h (
  if not exist "%~dp0third_party\OpenCL-Headers\CL\cl.h" (
    echo OpenCL headers not found. Fetch them:
    echo   git clone --depth 1 https://github.com/KhronosGroup/OpenCL-Headers.git third_party\OpenCL-Headers
    exit /b 1
  )
  set OCLINC=/I"%~dp0third_party\OpenCL-Headers"
)
cl /nologo /O2 /arch:AVX2 /Oi /openmp %OCLINC% ^
   "%~dp0fastsieve.c" "%~dp0gpu.c" /Fe:"%~dp0fastsieve.exe"
echo.
echo done: fastsieve.exe   (tests: powershell -ExecutionPolicy Bypass .\tests.ps1)