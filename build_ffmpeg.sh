#!/bin/bash
set -euo pipefail

export MSYSTEM=UCRT64
export PATH="/ucrt64/bin:/usr/bin:$PATH"
export PKG_CONFIG_PATH="/ucrt64/lib/pkgconfig"

SRC="$(cd "$(dirname "$0")/../ffmpeg" && pwd)"
OUT="$SRC/build"
JOBS=$(nproc 2>/dev/null || echo 4)

echo "=== Build cross-vendor FFmpeg runtime with software codecs ==="
cd "$SRC"
make distclean >/dev/null 2>&1 || true
rm -rf "$OUT"
mkdir -p "$OUT"

./configure \
    --prefix="$OUT" \
    --enable-shared --disable-static --enable-gpl \
    --disable-programs --disable-doc --disable-avdevice --disable-network \
    --disable-everything \
    --disable-bzlib --disable-lzma --disable-sdl2 --disable-vaapi \
    --enable-decoder=h264,hevc,av1,libdav1d,vp8,vp9,mpeg4,mpeg2video,mjpeg,png,prores,wmv1,wmv2,wmv3,vc1,theora \
    --enable-encoder=h264_nvenc,hevc_nvenc,av1_nvenc,h264_amf,hevc_amf,av1_amf,h264_mf,hevc_mf,libx264,libx265,libaom_av1,libsvtav1,mpeg4,mjpeg \
    --enable-hwaccel=h264_nvdec,hevc_nvdec,av1_nvdec \
    --enable-parser=h264,hevc,av1,vp8,vp9,mpeg4video,mpegvideo,mjpeg,vc1 \
    --enable-demuxer=mov,matroska,avi,webm,mpegts,mpegvideo,flv,ogg,image2 \
    --enable-muxer=mov,mp4,matroska,avi,webm,null \
    --enable-protocol=file \
    --enable-filter=null,anull,format,aformat,scale,transpose,hflip,vflip \
    --enable-bsf=h264_mp4toannexb,hevc_mp4toannexb,av1_frame_merge,av1_frame_split,av1_metadata \
    --enable-nvenc --enable-nvdec --enable-cuvid --enable-ffnvcodec \
    --enable-libx264 --enable-libx265 --enable-libaom --enable-libsvtav1 --enable-libdav1d \
    --enable-mediafoundation \
    --cc=gcc --cxx=g++ \
    --extra-cflags="-I/ucrt64/include" \
    --extra-ldflags="-L/ucrt64/lib"

make -j"$JOBS"
make install
cp -v /ucrt64/bin/libgcc_s_seh-1.dll "$OUT/bin/" 2>/dev/null || true
cp -v /ucrt64/bin/libstdc++-6.dll "$OUT/bin/" 2>/dev/null || true
cp -v /ucrt64/bin/libwinpthread-1.dll "$OUT/bin/" 2>/dev/null || true
echo "=== FFmpeg runtime ready: $OUT ==="
