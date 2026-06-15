#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "  编译 D3D11 VSR Bridge (零外部依赖)"

g++ -static-libgcc -static-libstdc++ \
    -std=c++17 -O2 -shared \
    -o "$SCRIPT_DIR/dxva_vsr_bridge.dll" \
    "$SCRIPT_DIR/dxva_vsr_bridge.cpp" \
    -ld3d11 -ldxgi -ldxguid -luuid -lole32

echo "  编译完成: $SCRIPT_DIR/dxva_vsr_bridge.dll"
