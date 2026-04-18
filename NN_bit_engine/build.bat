@echo off
setlocal enabledelayedexpansion

echo Compiling source files with dependencies...
:: Use wildcard to include all JARs in the lib folder
javac -encoding UTF-8 -cp "lib/*" -d bin src\*
if %errorlevel% neq 0 (
    echo Compilation failed!
    exit /b %errorlevel%
)

echo Creating engine.jar...
echo Main-Class: Main > manifest.txt
echo Class-Path: lib/onnxruntime-1.24.3.jar >> manifest.txt

jar cfm engine.jar manifest.txt -C bin .
if %errorlevel% neq 0 (
    echo JAR creation failed!
    exit /b %errorlevel%
)
del manifest.txt

echo Built engine.jar successfully.

