@echo off
setlocal enabledelayedexpansion

:: 1. Setup - just create bin for class files
if not exist "bin" mkdir bin

echo Compiling source files...
javac -encoding UTF-8 -d bin src\*.java
if %errorlevel% neq 0 (
    echo Compilation failed!
    exit /b %errorlevel%
)

echo Creating engine.jar...
:: Create the JAR in the current folder
jar cfe engine.jar Main -C bin .
if %errorlevel% neq 0 (
    echo JAR creation failed!
    exit /b %errorlevel%
)

echo.
echo Success! 
echo JAR: engine.jar
pause
