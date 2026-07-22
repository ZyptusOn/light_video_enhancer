/**
 * dxva_vsr_bridge.cpp — D3D11 Video Super Resolution 桥接（零外部依赖）
 *
 * 仅依赖: d3d11.lib dxgi.lib dxguid.lib uuid.lib ole32.lib
 * 不做任何像素格式转换。Python 端负责 BGR→NV12 和 BGRA→BGR。
 */

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <initguid.h>
#include <d3d11.h>
#include <d3d11_1.h>
#include <dxgi1_2.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

static const GUID kNvidiaPPE =
    {0xd43ce1b3, 0x1f4b, 0x48ac, {0xba, 0xee, 0xc3, 0xc2, 0x53, 0x75, 0xe6, 0xf7}};

static const GUID kIntelVPE =
    {0xedd1d4b9, 0x8659, 0x4cbc, {0xa4, 0xd6, 0x98, 0x31, 0xa2, 0x16, 0x3a, 0xc3}};

enum Vendor { VENDOR_UNKNOWN, VENDOR_NVIDIA, VENDOR_INTEL, VENDOR_AMD };

struct VSRContext {
    ID3D11Device*              d3dDevice      = nullptr;
    ID3D11DeviceContext*       d3dContext     = nullptr;
    ID3D11VideoDevice*         videoDevice    = nullptr;
    ID3D11VideoContext*        videoContext   = nullptr;
    ID3D11VideoProcessorEnumerator* vpEnum    = nullptr;
    ID3D11VideoProcessor*      videoProcessor = nullptr;

    ID3D11Texture2D*           inputNv12Hw    = nullptr;
    ID3D11Texture2D*           inputStaging   = nullptr;
    ID3D11Texture2D*           outputTex      = nullptr;
    ID3D11Texture2D*           outputStaging  = nullptr;
    ID3D11VideoProcessorInputView*  inputView = nullptr;
    ID3D11VideoProcessorOutputView* outputView = nullptr;

    int srcW = 0, srcH = 0, alignSrcW = 0, alignSrcH = 0;
    int dstW = 0, dstH = 0;

    Vendor vendor = VENDOR_UNKNOWN;
    bool vsrSupported = false;
    bool vsrEnabled   = false;

    std::vector<uint8_t> bgraOutBuf;
};

static Vendor detect_vendor(ID3D11Device* dev)
{
    IDXGIDevice* dxgiDev = nullptr;
    if (FAILED(dev->QueryInterface(__uuidof(IDXGIDevice), (void**)&dxgiDev))) return VENDOR_UNKNOWN;
    IDXGIAdapter* adapter = nullptr;
    if (FAILED(dxgiDev->GetAdapter(&adapter))) { dxgiDev->Release(); return VENDOR_UNKNOWN; }
    DXGI_ADAPTER_DESC desc;
    HRESULT hr = adapter->GetDesc(&desc);
    adapter->Release(); dxgiDev->Release();
    if (FAILED(hr)) return VENDOR_UNKNOWN;
    fprintf(stderr, "[VSR] GPU vendor: 0x%04X\n", desc.VendorId);
    switch (desc.VendorId) {
        case 0x10DE: return VENDOR_NVIDIA;
        case 0x8086: return VENDOR_INTEL;
        case 0x1002: return VENDOR_AMD;
        default:     return VENDOR_UNKNOWN;
    }
}

static bool CreateDevice(VSRContext* c)
{
    D3D_FEATURE_LEVEL levels[] = { D3D_FEATURE_LEVEL_11_1, D3D_FEATURE_LEVEL_11_0 };
    HRESULT hr = D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr,
        D3D11_CREATE_DEVICE_VIDEO_SUPPORT, levels, ARRAYSIZE(levels),
        D3D11_SDK_VERSION, &c->d3dDevice, nullptr, &c->d3dContext);
    // The Windows 7 D3D11 runtime returns E_INVALIDARG when 11_1 is present.
    // Retry with 11_0 exactly as recommended by the D3D11 API documentation.
    if (hr == E_INVALIDARG)
        hr = D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr,
            D3D11_CREATE_DEVICE_VIDEO_SUPPORT, &levels[1], 1,
            D3D11_SDK_VERSION, &c->d3dDevice, nullptr, &c->d3dContext);
    if (FAILED(hr)) { fprintf(stderr, "[VSR] D3D11CreateDevice: 0x%08X\n", hr); return false; }
    c->vendor = detect_vendor(c->d3dDevice);
    fprintf(stderr, "[VSR] Vendor: %s\n",
        c->vendor==VENDOR_NVIDIA?"NVIDIA":c->vendor==VENDOR_INTEL?"Intel":
        c->vendor==VENDOR_AMD?"AMD":"Unknown");
    hr = c->d3dDevice->QueryInterface(__uuidof(ID3D11VideoDevice), (void**)&c->videoDevice);
    if (FAILED(hr)) { fprintf(stderr, "[VSR] VideoDevice: 0x%08X\n", hr); return false; }
    hr = c->d3dContext->QueryInterface(__uuidof(ID3D11VideoContext), (void**)&c->videoContext);
    if (FAILED(hr)) { fprintf(stderr, "[VSR] VideoContext: 0x%08X\n", hr); return false; }
    return true;
}

static bool CreateVP(VSRContext* c)
{
    D3D11_VIDEO_PROCESSOR_CONTENT_DESC cd = {};
    cd.InputFrameFormat    = D3D11_VIDEO_FRAME_FORMAT_PROGRESSIVE;
    cd.InputFrameRate      = { 30000, 1001 };
    cd.InputWidth          = c->alignSrcW;
    cd.InputHeight         = c->alignSrcH;
    cd.OutputWidth         = c->dstW;
    cd.OutputHeight        = c->dstH;
    cd.OutputFrameRate     = { 30000, 1001 };
    cd.Usage               = D3D11_VIDEO_USAGE_PLAYBACK_NORMAL;

    HRESULT hr = c->videoDevice->CreateVideoProcessorEnumerator(&cd, &c->vpEnum);
    if (FAILED(hr)) { fprintf(stderr, "[VSR] CreateVPEnum: 0x%08X\n", hr); return false; }
    hr = c->videoDevice->CreateVideoProcessor(c->vpEnum, 0, &c->videoProcessor);
    if (FAILED(hr)) { fprintf(stderr, "[VSR] CreateVP: 0x%08X\n", hr); return false; }

    if (c->vendor == VENDOR_NVIDIA) {
        UINT avail = 0;
        hr = c->videoContext->VideoProcessorGetStreamExtension(
            c->videoProcessor, 0, &kNvidiaPPE, sizeof(avail), &avail);
        c->vsrSupported = SUCCEEDED(hr) && avail != 0;
        fprintf(stderr, "[VSR] NVIDIA RTX VSR: %s\n", c->vsrSupported?"YES":"NO");
    } else if (c->vendor == VENDOR_INTEL) {
        c->vsrSupported = true;
        fprintf(stderr, "[VSR] Intel VSR: enabled\n");
    } else {
        fprintf(stderr, "[VSR] AMD: D3D11 VP only\n");
    }

    D3D11_VIDEO_PROCESSOR_COLOR_SPACE cs = {};
    cs.YCbCr_Matrix  = 1;
    cs.Nominal_Range = 2;
    c->videoContext->VideoProcessorSetOutputColorSpace(c->videoProcessor, &cs);
    c->videoContext->VideoProcessorSetStreamColorSpace(c->videoProcessor, 0, &cs);
    RECT r = {0, 0, (LONG)c->dstW, (LONG)c->dstH};
    c->videoContext->VideoProcessorSetOutputTargetRect(c->videoProcessor, TRUE, &r);
    c->videoContext->VideoProcessorSetStreamAutoProcessingMode(c->videoProcessor, 0, TRUE);
    return true;
}

static void SetVSR(VSRContext* c, bool en)
{
    if (c->vsrEnabled == en) return;

    if (c->vendor == VENDOR_NVIDIA && c->vsrSupported) {
        struct { UINT v, m, e; } info = { 1, 2, en ? 1u : 0u };
        HRESULT hr = c->videoContext->VideoProcessorSetStreamExtension(
            c->videoProcessor, 0, &kNvidiaPPE, sizeof(info), &info);
        if (SUCCEEDED(hr)) {
            c->vsrEnabled = en;
            fprintf(stderr, "[VSR] NVIDIA VSR %s\n", en?"ON":"OFF");
        }
    } else if (c->vendor == VENDOR_INTEL && c->vsrSupported) {
        UINT p;
        struct { UINT fn; void* p; } ext = { 0, &p };
        p = 3; ext.fn = 1;
        c->videoContext->VideoProcessorSetOutputExtension(c->videoProcessor, &kIntelVPE, sizeof(ext), &ext);
        p = en ? 1u : 0u; ext.fn = 0x20;
        c->videoContext->VideoProcessorSetOutputExtension(c->videoProcessor, &kIntelVPE, sizeof(ext), &ext);
        p = en ? 2u : 0u; ext.fn = 0x37;
        c->videoContext->VideoProcessorSetStreamExtension(c->videoProcessor, 0, &kIntelVPE, sizeof(ext), &ext);
        c->vsrEnabled = en;
        fprintf(stderr, "[VSR] Intel VSR %s\n", en?"ON":"OFF");
    }
}

static bool CreateTextures(VSRContext* c)
{
    HRESULT hr;
    D3D11_TEXTURE2D_DESC d = {};

    d.Width=c->alignSrcW; d.Height=c->alignSrcH; d.MipLevels=1; d.ArraySize=1;
    d.Format=DXGI_FORMAT_NV12; d.SampleDesc={1,0};
    d.Usage=D3D11_USAGE_DEFAULT; d.BindFlags=D3D11_BIND_DECODER;
    hr=c->d3dDevice->CreateTexture2D(&d,nullptr,&c->inputNv12Hw);
    if(FAILED(hr)){fprintf(stderr,"[VSR] inpNv12: 0x%08X\n",hr);return false;}

    d.Usage=D3D11_USAGE_STAGING; d.BindFlags=0; d.CPUAccessFlags=D3D11_CPU_ACCESS_WRITE;
    hr=c->d3dDevice->CreateTexture2D(&d,nullptr,&c->inputStaging);
    if(FAILED(hr)){fprintf(stderr,"[VSR] inpStg: 0x%08X\n",hr);return false;}

    D3D11_VIDEO_PROCESSOR_INPUT_VIEW_DESC ivd={};
    ivd.ViewDimension=D3D11_VPIV_DIMENSION_TEXTURE2D;
    hr=c->videoDevice->CreateVideoProcessorInputView(c->inputNv12Hw,c->vpEnum,&ivd,&c->inputView);
    if(FAILED(hr)){fprintf(stderr,"[VSR] inpView: 0x%08X\n",hr);return false;}

    d={}; d.Width=c->dstW; d.Height=c->dstH; d.MipLevels=1; d.ArraySize=1;
    d.Format=DXGI_FORMAT_B8G8R8A8_UNORM; d.SampleDesc={1,0};
    d.Usage=D3D11_USAGE_DEFAULT; d.BindFlags=D3D11_BIND_RENDER_TARGET|D3D11_BIND_SHADER_RESOURCE;
    d.CPUAccessFlags=0;
    hr=c->d3dDevice->CreateTexture2D(&d,nullptr,&c->outputTex);
    if(FAILED(hr)){fprintf(stderr,"[VSR] outTex: 0x%08X\n",hr);return false;}

    d.Usage=D3D11_USAGE_STAGING; d.BindFlags=0; d.CPUAccessFlags=D3D11_CPU_ACCESS_READ;
    hr=c->d3dDevice->CreateTexture2D(&d,nullptr,&c->outputStaging);
    if(FAILED(hr)){fprintf(stderr,"[VSR] outStg: 0x%08X\n",hr);return false;}

    D3D11_VIDEO_PROCESSOR_OUTPUT_VIEW_DESC ovd={};
    ovd.ViewDimension=D3D11_VPOV_DIMENSION_TEXTURE2D;
    hr=c->videoDevice->CreateVideoProcessorOutputView(c->outputTex,c->vpEnum,&ovd,&c->outputView);
    if(FAILED(hr)){fprintf(stderr,"[VSR] outView: 0x%08X\n",hr);return false;}

    c->bgraOutBuf.resize((size_t)c->dstW*c->dstH*4);
    return true;
}

extern "C" {

__declspec(dllexport) void* dxva_vsr_create() { return new VSRContext(); }

__declspec(dllexport) int dxva_vsr_initialize(void* handle, int sw, int sh, int dw, int dh)
{
    auto* c = (VSRContext*)handle;
    c->srcW = sw; c->srcH = sh; c->dstW = dw; c->dstH = dh;
    c->alignSrcW = ((sw + 1) / 2) * 2;
    c->alignSrcH = ((sh + 1) / 2) * 2;
    fprintf(stderr, "[VSR] %dx%d -> %dx%d\n", sw, sh, dw, dh);
    if (!CreateDevice(c))   return -1;
    if (!CreateVP(c))       return -2;
    if (!CreateTextures(c)) return -3;
    SetVSR(c, true);
    return 0;
}

/*
 * dxva_vsr_process - input NV12, output raw BGRA (Python does BGRA->BGR)
 */
__declspec(dllexport) int dxva_vsr_process(void* handle,
    const uint8_t* nv12_data, int w, int h, int _unused_ch)
{
    auto* c = (VSRContext*)handle;
    if (!c || !nv12_data) return -1;
    if (w != c->srcW || h != c->srcH) {
        fprintf(stderr, "[VSR] 尺寸不匹配: 传入 %dx%d, 初始化 %dx%d\n", w, h, c->srcW, c->srcH);
        return -12;
    }
    HRESULT hr;

    int ySize  = c->alignSrcW * c->alignSrcH;
    int uvSize = ySize / 2;

    D3D11_MAPPED_SUBRESOURCE map = {};
    hr = c->d3dContext->Map(c->inputStaging, 0, D3D11_MAP_WRITE, 0, &map);
    if (FAILED(hr)) return -10;

    uint8_t* dstY = (uint8_t*)map.pData;
    for (int y = 0; y < c->alignSrcH; y++)
        memcpy(dstY + y * map.RowPitch, nv12_data + y * c->alignSrcW, c->alignSrcW);

    uint8_t* dstUV = dstY + c->alignSrcH * map.RowPitch;
    const uint8_t* srcUV = nv12_data + ySize;
    for (int y = 0; y < c->alignSrcH / 2; y++)
        memcpy(dstUV + y * map.RowPitch, srcUV + y * c->alignSrcW, c->alignSrcW);

    c->d3dContext->Unmap(c->inputStaging, 0);

    D3D11_BOX srcBox = {0, 0, 0, (UINT)c->alignSrcW, (UINT)c->alignSrcH, 1};
    c->d3dContext->CopySubresourceRegion(
        c->inputNv12Hw, 0, 0, 0, 0, c->inputStaging, 0, &srcBox);

    RECT sr = {0, 0, (LONG)c->srcW, (LONG)c->srcH};
    c->videoContext->VideoProcessorSetStreamSourceRect(c->videoProcessor, 0, TRUE, &sr);

    D3D11_VIDEO_PROCESSOR_STREAM stream = {};
    stream.Enable = TRUE;
    stream.pInputSurface = c->inputView;
    hr = c->videoContext->VideoProcessorBlt(c->videoProcessor, c->outputView, 0, 1, &stream);
    if (FAILED(hr)) return -11;

    D3D11_BOX outBox = {0, 0, 0, (UINT)c->dstW, (UINT)c->dstH, 1};
    c->d3dContext->CopySubresourceRegion(
        c->outputStaging, 0, 0, 0, 0, c->outputTex, 0, &outBox);

    D3D11_MAPPED_SUBRESOURCE outMap = {};
    hr = c->d3dContext->Map(c->outputStaging, 0, D3D11_MAP_READ, 0, &outMap);
    if (FAILED(hr)) return -13;

    uint8_t* out = c->bgraOutBuf.data();
    const uint8_t* gpu = (const uint8_t*)outMap.pData;
    for (int y = 0; y < c->dstH; y++)
        memcpy(out + y * c->dstW * 4, gpu + y * (int)outMap.RowPitch, c->dstW * 4);
    c->d3dContext->Unmap(c->outputStaging, 0);
    return 0;
}

__declspec(dllexport) int dxva_vsr_get_output_size(void* h)
    { return (int)((VSRContext*)h)->bgraOutBuf.size(); }

__declspec(dllexport) int dxva_vsr_get_output(void* h, uint8_t* buf, int sz)
{
    auto* c = (VSRContext*)h;
    if (sz < (int)c->bgraOutBuf.size()) return -20;
    memcpy(buf, c->bgraOutBuf.data(), c->bgraOutBuf.size());
    return 0;
}

__declspec(dllexport) void dxva_vsr_release(void* h)
{
    auto* c = (VSRContext*)h;
    SetVSR(c, false);
    if (c->outputView)   c->outputView->Release();
    if (c->inputView)    c->inputView->Release();
    if (c->outputStaging)c->outputStaging->Release();
    if (c->outputTex)    c->outputTex->Release();
    if (c->inputStaging) c->inputStaging->Release();
    if (c->inputNv12Hw)  c->inputNv12Hw->Release();
    if (c->videoProcessor) c->videoProcessor->Release();
    if (c->vpEnum)       c->vpEnum->Release();
    if (c->videoContext) c->videoContext->Release();
    if (c->videoDevice)  c->videoDevice->Release();
    if (c->d3dContext)   c->d3dContext->Release();
    if (c->d3dDevice)    c->d3dDevice->Release();
    delete c;
}

} // extern "C"
