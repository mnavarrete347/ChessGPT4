@echo off
setlocal enabledelayedexpansion

echo Compiling source files with dependencies...
:: Use wildcard to include all JARs in the lib folder
javac -encoding UTF-8 -cp "lib/*" -d bin src\Main.java
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

