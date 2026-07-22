#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "Building the dependency-free D3D11 video processor bridge..."
g++ -std=c++17 -O2 -shared -static \
    -o "$SCRIPT_DIR/dxva_vsr_bridge.dll" \
    "$SCRIPT_DIR/dxva_vsr_bridge.cpp" \
    -ld3d11 -ldxgi -ldxguid -luuid -lole32
echo "Built: $SCRIPT_DIR/dxva_vsr_bridge.dll"
