#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="${PROJECT_DIR}/build"
SRC_DIR="${PROJECT_DIR}/src"
mkdir -p "${BUILD_DIR}"
FUTHARK_SRC="${SRC_DIR}/futhark/kernels.fut"
FUTHARK_OUT="${BUILD_DIR}/kernels"
if command -v futhark &> /dev/null; then
    if [ -f "${FUTHARK_SRC}" ]; then
        if futhark cuda --library "${FUTHARK_SRC}" -o "${FUTHARK_OUT}" 2>/dev/null; then
            echo "Futhark CUDA build complete"
        elif futhark opencl --library "${FUTHARK_SRC}" -o "${FUTHARK_OUT}" 2>/dev/null; then
            echo "Futhark OpenCL build complete"
        elif futhark multicore --library "${FUTHARK_SRC}" -o "${FUTHARK_OUT}" 2>/dev/null; then
            echo "Futhark multicore build complete"
        elif futhark c --library "${FUTHARK_SRC}" -o "${FUTHARK_OUT}" 2>/dev/null; then
            echo "Futhark C build complete"
        else
            echo "Futhark build failed"
        fi
    else
        echo "Futhark source not found: ${FUTHARK_SRC}"
    fi
else
    echo "Futhark not found, skipping kernel build"
fi
TERRA_SRC="${SRC_DIR}/terra/engine.t"
TERRA_OUT="${BUILD_DIR}/engine.so"
if command -v terra &> /dev/null; then
    if [ -f "${TERRA_SRC}" ]; then
        cd "${BUILD_DIR}"
        if terra "${TERRA_SRC}" 2>/dev/null; then
            echo "Terra build complete"
        else
            echo "Terra build encountered issues"
        fi
        cd "${PROJECT_DIR}"
    else
        echo "Terra source not found: ${TERRA_SRC}"
    fi
else
    echo "Terra not found, skipping engine build"
fi
if [ -f "${BUILD_DIR}/kernels.c" ]; then
    echo "Futhark C output: ${BUILD_DIR}/kernels.c"
fi
if [ -f "${BUILD_DIR}/kernels.h" ]; then
    echo "Futhark header: ${BUILD_DIR}/kernels.h"
fi
if [ -f "${BUILD_DIR}/engine.so" ]; then
    echo "Engine library: ${BUILD_DIR}/engine.so"
fi
echo "Build process completed"
