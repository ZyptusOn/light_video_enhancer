#include "span.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <vector>

#include "mat.h"

namespace
{
FILE* open_binary(const std::wstring& path)
{
    return _wfopen(path.c_str(), L"rb");
}
}

Span::Span(int gpuid, int scale, int tilesize)
    : scale_(scale), tilesize_(tilesize)
{
    net_.opt.use_vulkan_compute = true;
    // SPAN's repeated multiplicative attention amplifies FP16 quantization:
    // both packed and storage FP16 produce visible errors on the official
    // checkpoints, while FP32 Vulkan costs effectively no time on this graph.
    net_.opt.use_fp16_packed = false;
    net_.opt.use_fp16_storage = false;
    net_.opt.use_fp16_arithmetic = false;
    net_.set_vulkan_device(gpuid);
}

int Span::load(const std::wstring& param_path,
               const std::wstring& model_path)
{
    FILE* param = open_binary(param_path);
    if (!param)
        return -1;
    const int param_status = net_.load_param(param);
    std::fclose(param);
    if (param_status != 0)
        return param_status;

    FILE* model = open_binary(model_path);
    if (!model)
        return -1;
    const int model_status = net_.load_model(model);
    std::fclose(model);
    return model_status;
}

int Span::process_tile(const unsigned char* pixels, int width, int height,
                       int stride, unsigned char* output) const
{
    ncnn::Mat input = ncnn::Mat::from_pixels(
        pixels, ncnn::Mat::PIXEL_BGR2RGB, width, height, stride);
    const float mean[3] = {
        0.4488f * 255.f, 0.4371f * 255.f, 0.4040f * 255.f};
    input.substract_mean_normalize(mean, nullptr);

    ncnn::Extractor extractor = net_.create_extractor();
    extractor.set_light_mode(true);
    int status = extractor.input("in0", input);
    if (status != 0)
        return status;

    // Keep the extracted Mat inside the Extractor lifetime. NCNN allocates
    // Vulkan output from the Extractor's pool; returning the Mat would leave
    // it referencing a pool that has already been destroyed.
    ncnn::Mat result;
    status = extractor.extract("out0", result);
    if (status == 0)
    {
        // The converted graph follows the official PyTorch model and emits
        // RGB floats in roughly [0, 1]. Convert back to byte-range BGR for
        // the shared-memory pipeline; calling to_pixels directly would round
        // almost every value to zero and produce a black video.
        const float byte_scale[3] = {255.f, 255.f, 255.f};
        result.substract_mean_normalize(nullptr, byte_scale);
        result.to_pixels(output, ncnn::Mat::PIXEL_RGB2BGR);
    }
    return status;
}

int Span::process(const ncnn::Mat& input, ncnn::Mat& output) const
{
    const int output_width = input.w * scale_;
    const int output_height = input.h * scale_;
    if (output.empty() || output.w != output_width ||
        output.h != output_height)
    {
        output.create(output_width, output_height,
                      static_cast<size_t>(3), 3);
    }

    const auto* input_bytes =
        static_cast<const unsigned char*>(input.data);
    auto* output_bytes = static_cast<unsigned char*>(output.data);
    const int input_stride = input.w * 3;
    const int output_stride = output_width * 3;
    if (tilesize_ <= 0 ||
        (input.w <= tilesize_ && input.h <= tilesize_))
    {
        return process_tile(
            input_bytes, input.w, input.h, input_stride, output_bytes);
    }

    // The reparameterized graph has a 21-pixel receptive-field radius.
    // Twenty-four pixels keeps each retained tile independent of its seam.
    constexpr int overlap = 24;
    for (int core_y = 0; core_y < input.h; core_y += tilesize_)
    {
        const int core_y_end = std::min(core_y + tilesize_, input.h);
        const int top = std::max(0, core_y - overlap);
        const int bottom = std::min(input.h, core_y_end + overlap);
        for (int core_x = 0; core_x < input.w; core_x += tilesize_)
        {
            const int core_x_end = std::min(core_x + tilesize_, input.w);
            const int left = std::max(0, core_x - overlap);
            const int right = std::min(input.w, core_x_end + overlap);
            const int tile_width = right - left;
            const int tile_height = bottom - top;

            std::vector<unsigned char> tile(
                static_cast<size_t>(tile_width) * tile_height *
                scale_ * scale_ * 3);
            const int status = process_tile(
                input_bytes + static_cast<size_t>(top) * input_stride +
                    static_cast<size_t>(left) * 3,
                tile_width, tile_height, input_stride, tile.data());
            if (status != 0)
                return status;

            const int tile_stride = tile_width * scale_ * 3;
            const int source_x = (core_x - left) * scale_ * 3;
            const int source_y = (core_y - top) * scale_;
            const int copy_bytes = (core_x_end - core_x) * scale_ * 3;
            const int rows = (core_y_end - core_y) * scale_;
            for (int row = 0; row < rows; ++row)
            {
                const auto* source = tile.data() +
                    static_cast<size_t>(source_y + row) * tile_stride +
                    source_x;
                auto* destination = output_bytes +
                    static_cast<size_t>(core_y * scale_ + row) *
                        output_stride +
                    core_x * scale_ * 3;
                std::memcpy(destination, source, copy_bytes);
            }
        }
    }
    return 0;
}
