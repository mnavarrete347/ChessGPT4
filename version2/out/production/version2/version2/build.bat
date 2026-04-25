@echo off
:: --- CONFIGURATION ---
set APP_NAME=Arena
:: The name of your .java file (without extension)
set JAVA_FILE=UciBoardArena2
:: The package name declared at the top of your Java file
set PACKAGE=version2
:: ---------------------

echo [1/2] Compiling %JAVA_FILE%.java into package structure...
:: The '-d .' flag tells javac to create the package folders automatically
javac -d . "%JAVA_FILE%.java"

if %errorlevel% neq 0 (
    echo.
    echo ERROR: Compilation failed.
    pause
    exit /b %errorlevel%
)

echo [2/2] Creating %APP_NAME%.jar with Entry Point %PACKAGE%.%JAVA_FILE%...
:: Notice we use %PACKAGE%.%JAVA_FILE% as the entry point 
:: and we include the folder %PACKAGE% in the JAR
jar cvfe "%APP_NAME%.jar" %PACKAGE%.%JAVA_FILE% %PACKAGE%/*.class

echo.
echo ---------------------------------------
echo Done! Created %APP_NAME%.jar successfully.
echo Run it with: java -jar %APP_NAME%.jar
echo ---------------------------------------
pause