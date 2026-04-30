#!/bin/bash

cd build
ninja -j$(nproc) miopen_gtest install
cd ../WIP
rm perf/*; rm -rf /tmp/.cache/miopen/*; rm -rf ~/.cache/comgr/*; AMD_COMGR_SAVE_TEMPS=1 AMD_COMGR_SAVE_LLVM_TEMPS=1 AMD_COMGR_REDIRECT_LOGS=stdout AMD_COMGR_EMIT_VERBOSE_LOGS=1 MIOPEN_LOG_LEVEL=7 MIOPEN_DEBUG_SAVE_TEMP_DIR=1 ./bench_convoclbwdwrw53.sh -r 4 6