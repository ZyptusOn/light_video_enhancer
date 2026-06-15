#!/bin/bash
# ============================================================
#  FFmpeg 精简 DLL 编译脚本 — NVIDIA Video Enhancer 专用
#  此脚本在 MSYS2 UCRT64 环境中运行
# ============================================================
#
# 编译产物:
#   avcodec-xx.dll   — 编解码器（含 NVDEC/NVENC）
#   avformat-xx.dll  — 容器封装/解封装
#   avutil-xx.dll    — 工具函数
#   swscale-x.dll    — 像素格式转换/缩放
#   swresample-x.dll — 音频重采样
#
# 大小: ~15-25 MB（仅保留必要组件）
#
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

if [ -n "$FFMPEG_SRC" ] && [ -d "$FFMPEG_SRC" ]; then
    true
elif [ -d "$SCRIPT_DIR/../../ffmpeg" ]; then
    FFMPEG_SRC="$(cd "$SCRIPT_DIR/../../ffmpeg" && pwd)"
elif [ -d /d/AAALearning/tst/ffmpeg ]; then
    FFMPEG_SRC=/d/AAALearning/tst/ffmpeg
else
    echo "[错误] 找不到 FFmpeg 源码目录"
    echo "  SCRIPT_DIR=$SCRIPT_DIR"
    echo "  尝试: $SCRIPT_DIR/../../ffmpeg"
    echo "  请设置: export FFMPEG_SRC=/你的/ffmpeg/路径"
    exit 1
fi
BUILD_DIR="$FFMPEG_SRC/build"
OUTPUT_DIR="$SCRIPT_DIR/../ffmpeg_dlls"

echo "============================================"
echo "  NVIDIA Video Enhancer — FFmpeg DLL 编译"
echo "============================================"
echo ""
echo "  FFmpeg 源码: $FFMPEG_SRC"
echo "  编译目录:    $BUILD_DIR"
echo "  输出目录:    $OUTPUT_DIR"
echo ""

# ============================================================
# 步骤 1：检查 MSYS2 环境
# ============================================================
if ! command -v gcc &> /dev/null; then
    echo "[错误] 未找到 gcc，请在 MSYS2 中运行："
    echo "  pacman -S mingw-w64-ucrt-x86_64-gcc"
    exit 1
fi

if ! command -v pkg-config &> /dev/null; then
    echo "[错误] 未找到 pkg-config，请在 MSYS2 中运行："
    echo "  pacman -S mingw-w64-ucrt-x86_64-pkg-config"
    exit 1
fi

echo "[OK] 编译环境检查通过"
echo "  gcc:         $(gcc --version | head -1)"
echo ""

# ============================================================
# 步骤 2：安装 NVIDIA Video Codec SDK 头文件
# ============================================================
if ! pkg-config --exists ffnvcodec 2>/dev/null; then
    echo "[信息] 安装 NVIDIA Video Codec SDK 头文件 ..."
    if [ ! -d "$FFMPEG_SRC/nv-codec-headers" ]; then
        git clone --depth 1 https://github.com/FFmpeg/nv-codec-headers.git \
            "$FFMPEG_SRC/nv-codec-headers" 2>/dev/null || {
            echo "[警告] 无法 clone nv-codec-headers，NVENC/NVDEC 将不可用"
            echo "  请手动下载: https://github.com/FFmpeg/nv-codec-headers"
            echo "  解压到: $FFMPEG_SRC/nv-codec-headers"
        }
    fi
    if [ -d "$FFMPEG_SRC/nv-codec-headers" ]; then
        cd "$FFMPEG_SRC/nv-codec-headers"
        make install PREFIX=/mingw64 2>/dev/null || true
        cd "$FFMPEG_SRC"
    fi
else
    echo "[OK] ffnvcodec 已安装"
fi

# ============================================================
# 步骤 3：Configure — 只启用需要的组件
# ============================================================
cd "$FFMPEG_SRC"

mkdir -p "$BUILD_DIR"

echo "[信息] 运行 configure ..."
echo ""

./configure \
    --prefix="$BUILD_DIR" \
    --enable-shared \
    --disable-static \
    --disable-programs \
    --disable-doc \
    --disable-avdevice \
    --disable-avfilter \
    --disable-network \
    --disable-swresample \
    --disable-iconv \
    --disable-bzlib \
    --disable-lzma \
    --disable-zlib \
    --disable-sdl2 \
    --disable-vaapi \
    --disable-vdpau \
    --disable-amf \
    --enable-ffnvcodec \
    --enable-nonfree \
    \
    --enable-decoder=h264,hevc,vp9,av1 \
    --enable-decoder=png \
    --enable-hwaccel=h264_nvdec,hevc_nvdec,vp9_nvdec,av1_nvdec \
    \
    --enable-encoder=h264_nvenc,hevc_nvenc,av1_nvenc \
    --enable-encoder=png \
    \
    --enable-demuxer=mov,matroska,avi,mpegts \
    --enable-muxer=mp4,matroska,h264,hevc,null \
    \
    --enable-parser=h264,hevc,vp9,av1,png \
    \
    --enable-protocol=file \
    \
    --enable-bsf=h264_mp4toannexb,hevc_mp4toannexb,extract_extradata,null \
    \
    --enable-swscale \
    --enable-small \
    --optflags="-O2" \
    \
    --extra-cflags="-I$FFMPEG_SRC/nv-codec-headers/include" \
    --extra-ldflags="-L$FFMPEG_SRC/nv-codec-headers/lib" \
    \
    "$@"

echo ""
echo "============================================"
echo "  配置完成！接下来编译..."
echo "============================================"
echo ""

# ============================================================
# 步骤 4：编译
# ============================================================
make -j$(nproc) 2>&1 | tee make.log

echo ""
echo "============================================"
echo "  编译完成！正在安装到 $BUILD_DIR ..."
echo "============================================"
echo ""

# ============================================================
# 步骤 5：安装
# ============================================================
make install

echo ""
echo "============================================"
echo "  安装完成！"
echo "============================================"
echo ""

# ============================================================
# 步骤 6：复制 DLL 到项目目录
# ============================================================
mkdir -p "$OUTPUT_DIR"

echo "[信息] 复制 DLL 到 $OUTPUT_DIR ..."

# ffmpeg 8.0 版本的 DLL 版本号
for pattern in avcodec avformat avutil swscale swresample; do
    for f in "$BUILD_DIR/bin/$pattern"*.dll; do
        if [ -f "$f" ]; then
            cp -v "$f" "$OUTPUT_DIR/"
        fi
    done
done

# 显示产物大小
echo ""
echo "=== DLL 产物 ==="
du -sh "$OUTPUT_DIR"/*.dll 2>/dev/null
echo ""
echo "总大小: $(du -sh "$OUTPUT_DIR" | cut -f1)"

echo ""
echo "============================================"
echo "  编译全部完成！"
echo "  DLL 位置: $OUTPUT_DIR"
echo "============================================"
echo ""
echo "接下来请在 PowerShell/CMD 中运行："
echo "  python -m nvidia_video_enhancer input.mp4 -o output.mp4"
echo ""
