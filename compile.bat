@echo off
setlocal

REM Use UTF-8 code page to avoid garbled output
chcp 65001 >nul

echo Start compiling thesis...
echo.

where xelatex >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: xelatex not found in PATH.
    echo Please install TeX Live or MiKTeX first.
    pause
    exit /b 1
)

where biber >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: biber not found in PATH.
    echo Please install biber first.
    pause
    exit /b 1
)

echo [1/4] xelatex pass 1
xelatex -synctex=1 -interaction=nonstopmode main.tex
if %errorlevel% neq 0 (
    echo Failed at xelatex pass 1.
    pause
    exit /b 1
)

echo [2/4] biber
biber main
if %errorlevel% neq 0 (
    echo Failed at biber.
    pause
    exit /b 1
)

echo [3/4] xelatex pass 2
xelatex -synctex=1 -interaction=nonstopmode main.tex
if %errorlevel% neq 0 (
    echo Failed at xelatex pass 2.
    pause
    exit /b 1
)

echo [4/4] xelatex pass 3
xelatex -synctex=1 -interaction=nonstopmode main.tex
if %errorlevel% neq 0 (
    echo Failed at xelatex pass 3.
    pause
    exit /b 1
)

echo.
echo Compilation finished. Output: main.pdf
pause
