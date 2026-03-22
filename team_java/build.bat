@echo off
setlocal enabledelayedexpansion

if not exist "bin" (
    mkdir bin
)

echo Compiling source files...
javac -encoding UTF-8 -d bin src\*.java
if %errorlevel% neq 0 (
    echo Compilation failed!
    exit /b %errorlevel%
)

echo Creating engine.jar...
jar cfe engine.jar Main -C bin .
if %errorlevel% neq 0 (
    echo JAR creation failed!
    exit /b %errorlevel%
)

echo Built engine.jar successfully.
pause
