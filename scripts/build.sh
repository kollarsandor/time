#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="${PROJECT_DIR}/build"
mkdir -p "${BUILD_DIR}"
if command -v futhark &> /dev/null; then
    echo "Building Futhark kernels..."
    futhark cuda --library "${PROJECT_DIR}/src/futhark/kernels.fut" -o "${BUILD_DIR}/kernels"
    echo "Futhark build complete"
else
    echo "Futhark not found, skipping kernel build"
fi
if command -v terra &> /dev/null; then
    echo "Building Terra engine..."
    cd "${BUILD_DIR}"
    terra "${PROJECT_DIR}/src/terra/engine.t"
    echo "Terra build complete"
else
    echo "Terra not found, skipping engine build"
fi
echo "Build complete"
