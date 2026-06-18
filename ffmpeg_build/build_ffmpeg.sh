#!/bin/bash
set -e

export MSYSTEM=UCRT64
export PATH="/ucrt64/bin:/usr/bin:$PATH"
export PKG_CONFIG_PATH="/ucrt64/lib/pkgconfig"

SRC="/d/AAALearning/tst/ffmpeg"
OUT="/d/AAALearning/tst/ffmpeg/build"
JOBS=$(nproc 2>/dev/null || echo 4)

echo "=== Build FFmpeg (shared, NVENC) ==="
echo "Source: $SRC"
echo "Output: $OUT"

cd "$SRC"
make distclean 2>/dev/null || true

rm -rf "$OUT"
mkdir -p "$OUT"

echo "[Configure]"
./configure \
    --prefix="$OUT" \
    --enable-shared --disable-static \
    --disable-programs --disable-doc \
    --disable-ffplay --disable-ffprobe --disable-ffmpeg \
    --disable-avdevice --disable-network \
    --disable-everything \
    --enable-encoder=h264_nvenc \
    --enable-encoder=hevc_nvenc \
    --enable-encoder=av1_nvenc \
    --enable-hwaccel=h264_nvdec \
    --enable-hwaccel=hevc_nvdec \
    --enable-hwaccel=av1_nvdec \
    --enable-parser=h264 \
    --enable-parser=hevc \
    --enable-parser=av1 \
    --enable-demuxer=mov,matroska,avi,mp4,m4v,webm,rawvideo \
    --enable-muxer=mov,matroska,avi,mp4,webm,null \
    --enable-protocol=file \
    --enable-filter=null,anull,format,aformat,scale,transpose,hflip,vflip \
    --enable-bsf=h264_mp4toannex,hevc_mp4toannex,av1_frame_merge,av1_frame_split,av1_metadata \
    --enable-nvenc --enable-nvdec --enable-cuvid \
    --enable-ffnvcodec --enable-nonfree \
    --cc=gcc --cxx=g++ \
    --extra-cflags="-I/ucrt64/include" \
    --extra-ldflags="-L/ucrt64/lib"

echo ""
echo "[Make -j$JOBS]"
make -j$JOBS

echo ""
echo "[Install]"
make install

echo ""
echo "[Copy runtime DLLs]"
cp -v /ucrt64/bin/libgcc_s_seh-1.dll "$OUT/bin/" 2>/dev/null || true
cp -v /ucrt64/bin/libstdc++-6.dll "$OUT/bin/" 2>/dev/null || true
cp -v /ucrt64/bin/libwinpthread-1.dll "$OUT/bin/" 2>/dev/null || true

echo ""
echo "=== Done ==="
ls -la "$OUT/bin/"*.dll
