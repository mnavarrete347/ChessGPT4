#!/bin/bash
set -e

DIR="$(cd "$(dirname "$0")" && pwd)"
java -Xmx512m -jar "$DIR/engine.jar"
