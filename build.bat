@echo off
REM Build fastsieve (CPU + OpenCL GPU module) with MSVC.
setlocal
set VSDIR=%ProgramFiles%\Microsoft Visual Studio\2022\Community
call "%VSDIR%\Common7\Tools\VsDevCmd.bat" -arch=amd64 -host_arch=amd64 >nul 2>&1
if not exist CL\cl.h (
  if exist "%~dp0third_party\OpenCL-Headers\CL\cl.h" (
    set OCLINC=/I"%~dp0third_party\OpenCL-Headers"
  ) else if defined OCL_HOME (
    set OCLINC=/I"%OCL_HOME%"
  ) else (
    echo OpenCL headers not found. Set OCL_HOME to the headers dir or clone:
    echo   git clone --depth 1 https://github.com/KhronosGroup/OpenCL-Headers.git third_party\OpenCL-Headers
    exit /b 1
  )
)
cl /nologo /O2 /arch:AVX2 /Oi /openmp %OCLINC% ^
   "%~dp0fastsieve.c" "%~dp0gpu.c" /Fe:"%~dp0fastsieve.exe"
cl /nologo /O2 /arch:AVX2 /Oi /openmp %OCLINC% ^
   "%~dp0fastsieve.c" "%~dp0gpu.c" "%~dp0examples\api_demo.c" ^
   /DFASTSIEVE_NO_MAIN /Fe:"%~dp0api_demo.exe"
echo.
echo done: fastsieve.exe, api_demo.exe   (tests: powershell -ExecutionPolicy Bypass .\tests.ps1)