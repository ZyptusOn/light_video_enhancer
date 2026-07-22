#!/bin/bash
# Build the Windows 7-compatible FFmpeg worker from an MSYS2 UCRT64 shell.
set -e

export MSYSTEM=UCRT64
export PATH="/ucrt64/bin:/usr/bin:$PATH"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
FFMPEG_BUILD="${FFMPEG_BUILD:-$PROJECT_DIR/../ffmpeg/build}"
FFMPEG_DLLS="$SCRIPT_DIR/../ffmpeg_dlls"

echo "============================================"
echo "  Build FFmpeg Worker DLL"
echo "============================================"

gcc -shared -O2 \
    -o "$SCRIPT_DIR/ffmpeg_worker.dll" \
    "$SCRIPT_DIR/ffmpeg_worker_v8.c" \
    -I"$FFMPEG_BUILD/include" \
    -L"$FFMPEG_BUILD/lib" \
    -L"$FFMPEG_BUILD/bin" \
    -lavformat -lavcodec -lavutil -lswscale \
    -lmfplat -lmfuuid -lole32 -luuid \
    -static-libgcc

echo "  Copy FFmpeg runtime DLLs to $FFMPEG_DLLS"
mkdir -p "$FFMPEG_DLLS"
cp -v "$FFMPEG_BUILD/bin/"*.dll "$FFMPEG_DLLS/"
for runtime in \
    /ucrt64/bin/libiconv-2.dll \
    /ucrt64/bin/zlib1.dll \
    /ucrt64/bin/libzstd.dll \
    /ucrt64/bin/libdav1d-*.dll \
    /ucrt64/bin/libaom.dll \
    /ucrt64/bin/libSvtAv1Enc-*.dll \
    /ucrt64/bin/libx264-*.dll \
    /ucrt64/bin/libx265-*.dll; do
    [ -f "$runtime" ] && cp -v "$runtime" "$FFMPEG_DLLS/"
done

echo "Worker: $SCRIPT_DIR/ffmpeg_worker.dll"
echo "FFmpeg: $FFMPEG_DLLS"
