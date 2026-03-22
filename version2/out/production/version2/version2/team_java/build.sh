#!/bin/bash
set -e
mkdir -p bin
javac -encoding UTF-8 -d bin src/*.java
jar cfe engine.jar Main -C bin .
echo "Built engine.jar"
