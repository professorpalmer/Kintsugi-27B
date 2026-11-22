@echo off
rem Build tools\engine_escha.exe (CPU only) with the MSVC build tools. Output: work\engine_escha.exe
setlocal
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul || exit /b 1
set HERE=%~dp0
if not exist "%HERE%..\work" mkdir "%HERE%..\work"
cl /nologo /O2 /arch:AVX2 /fp:precise /Fe"%HERE%..\work\engine_escha.exe" /Fo"%HERE%..\work\\" "%HERE%engine_escha.c" || exit /b 1
