#!/bin/bash
# ============================================================
#  编译 ffmpeg_worker.dll — FFmpeg C API 包装器
#  在 MSYS2 UCRT64 终端中运行: ./build_worker.sh
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
FFMPEG_BUILD="/d/AAALearning/tst/ffmpeg/build"
FFMPEG_DLLS="$SCRIPT_DIR/../ffmpeg_dlls"

echo "============================================"
echo "  编译 FFmpeg Worker DLL"
echo "============================================"
echo ""

gcc -shared -O2 \
    -o "$SCRIPT_DIR/ffmpeg_worker.dll" \
    "$SCRIPT_DIR/ffmpeg_worker.c" \
    -I"$FFMPEG_BUILD/include" \
    -L"$FFMPEG_BUILD/lib" \
    -L"$FFMPEG_BUILD/bin" \
    -lavcodec -lavformat -lavutil -lswscale \
    -static-libgcc -static-libstdc++

echo ""
echo "  复制 FFmpeg DLLs 到 $FFMPEG_DLLS ..."
cp -v "$FFMPEG_BUILD/bin/"*.dll "$FFMPEG_DLLS/" 2>/dev/null || true

echo ""
echo "============================================"
echo "  编译完成!"
echo "  Worker:  $SCRIPT_DIR/ffmpeg_worker.dll"
echo "  FFmpeg:  $FFMPEG_DLLS"
echo "============================================"
