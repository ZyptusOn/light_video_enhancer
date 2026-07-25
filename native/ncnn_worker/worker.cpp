#include <windows.h>
#include <fcntl.h>
#include <io.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "gpu.h"
#include "mat.h"
#include "realcugan.h"
#include "realesrgan.h"
#include "rife.h"

namespace
{
constexpr uint32_t kMagic = 0x4E45564Cu; // "LVEN" little endian
constexpr uint32_t kVersion = 1;

#pragma pack(push, 1)
struct Request
{
    uint32_t magic;
    uint32_t version;
    uint32_t command;
    uint32_t count;
    uint32_t skip_first;
};

struct Reply
{
    uint32_t magic;
    uint32_t version;
    int32_t status;
    uint32_t count;
    uint32_t message_size;
    double elapsed_ms;
};
#pragma pack(pop)

bool read_all(void* data, size_t size)
{
    auto* output = static_cast<unsigned char*>(data);
    while (size)
    {
        const size_t chunk = std::min<size_t>(size, 1u << 20);
        const size_t got = std::fread(output, 1, chunk, stdin);
        if (!got)
            return false;
        output += got;
        size -= got;
    }
    return true;
}

void write_reply(int status, uint32_t count, const std::string& message,
                 double elapsed_ms = 0.0)
{
    Reply reply{kMagic, kVersion, status, count,
                static_cast<uint32_t>(message.size()), elapsed_ms};
    std::fwrite(&reply, 1, sizeof(reply), stdout);
    if (!message.empty())
        std::fwrite(message.data(), 1, message.size(), stdout);
    std::fflush(stdout);
}

std::map<std::wstring, std::wstring> parse_args(int argc, wchar_t** argv)
{
    std::map<std::wstring, std::wstring> result;
    for (int index = 1; index + 1 < argc; index += 2)
    {
        std::wstring key = argv[index];
        if (key.rfind(L"--", 0) != 0)
            throw std::runtime_error("invalid worker argument");
        result[key.substr(2)] = argv[index + 1];
    }
    return result;
}

std::wstring required(const std::map<std::wstring, std::wstring>& args,
                      const wchar_t* name)
{
    const auto found = args.find(name);
    if (found == args.end() || found->second.empty())
        throw std::runtime_error("missing worker argument");
    return found->second;
}

int integer(const std::map<std::wstring, std::wstring>& args,
            const wchar_t* name, int fallback = 0)
{
    const auto found = args.find(name);
    return found == args.end() ? fallback : std::stoi(found->second);
}

class GpuInstance
{
public:
    GpuInstance() { ncnn::create_gpu_instance(); }
    ~GpuInstance() { ncnn::destroy_gpu_instance(); }

    GpuInstance(const GpuInstance&) = delete;
    GpuInstance& operator=(const GpuInstance&) = delete;
};

class SharedMapping
{
public:
    explicit SharedMapping(const std::wstring& name)
    {
        handle_ = OpenFileMappingW(FILE_MAP_ALL_ACCESS, FALSE, name.c_str());
        if (!handle_)
            throw std::runtime_error("OpenFileMapping failed");
        data_ = static_cast<unsigned char*>(
            MapViewOfFile(handle_, FILE_MAP_ALL_ACCESS, 0, 0, 0));
        if (!data_)
        {
            CloseHandle(handle_);
            handle_ = nullptr;
            throw std::runtime_error("MapViewOfFile failed");
        }
    }

    ~SharedMapping()
    {
        if (data_)
            UnmapViewOfFile(data_);
        if (handle_)
            CloseHandle(handle_);
    }

    unsigned char* data() const { return data_; }

    SharedMapping(const SharedMapping&) = delete;
    SharedMapping& operator=(const SharedMapping&) = delete;

private:
    HANDLE handle_ = nullptr;
    unsigned char* data_ = nullptr;
};

int cugan_tile(int gpuid, int scale, int requested)
{
    if (requested >= 32)
        return requested;
    const uint32_t budget = ncnn::get_gpu_device(gpuid)->get_heap_budget();
    if (scale == 2)
        return budget > 1300 ? 400 : budget > 800 ? 300 :
               budget > 400 ? 200 : budget > 200 ? 100 : 32;
    if (scale == 3)
        return budget > 3300 ? 400 : budget > 1900 ? 300 :
               budget > 950 ? 200 : budget > 320 ? 100 : 32;
    return budget > 1690 ? 400 : budget > 980 ? 300 :
           budget > 530 ? 200 : budget > 240 ? 100 : 32;
}

int esrgan_tile(int gpuid, int requested)
{
    if (requested >= 32)
        return requested;
    const uint32_t budget = ncnn::get_gpu_device(gpuid)->get_heap_budget();
    return budget > 1900 ? 200 : budget > 550 ? 100 :
           budget > 190 ? 64 : 32;
}

void swap_red_blue(unsigned char* data, size_t pixels)
{
    for (size_t index = 0; index < pixels; ++index, data += 3)
        std::swap(data[0], data[2]);
}

class Pipeline
{
public:
    Pipeline(const std::map<std::wstring, std::wstring>& args,
             unsigned char* input, unsigned char* output)
        : input_(input), output_(output)
    {
        src_w_ = integer(args, L"src-w");
        src_h_ = integer(args, L"src-h");
        dst_w_ = integer(args, L"dst-w");
        dst_h_ = integer(args, L"dst-h");
        max_input_ = integer(args, L"max-input");
        max_output_ = integer(args, L"max-output");
        multiplier_ = integer(args, L"multiplier", 1);
        gpuid_ = integer(args, L"gpu", -2);
        if (gpuid_ == -2)
            gpuid_ = ncnn::get_default_gpu_index();
        if (gpuid_ < 0 || gpuid_ >= ncnn::get_gpu_count())
            throw std::runtime_error("invalid Vulkan GPU");
        if (src_w_ < 1 || src_h_ < 1 || dst_w_ < 1 || dst_h_ < 1 ||
            max_input_ < 1 || max_output_ < 1)
            throw std::runtime_error("invalid frame geometry");
        source_cache_.resize(max_input_);
        prediction_.create(
            src_w_, src_h_, static_cast<size_t>(3), 3);

        const std::wstring rife_model = required(args, L"rife-model");
        if (rife_model != L"none")
        {
            const bool tta = integer(args, L"rife-tta") != 0;
            const bool uhd = integer(args, L"rife-uhd") != 0;
            rife_.reset(new RIFE(gpuid_, tta, false, uhd, 1, false, true));
            if (multiplier_ < 2)
                throw std::runtime_error("invalid RIFE multiplier");
            if (rife_->load(rife_model) != 0)
                throw std::runtime_error("failed to load RIFE model");
        }

        sr_kind_ = required(args, L"sr-kind");
        if (sr_kind_ == L"realcugan")
        {
            const int scale = integer(args, L"sr-scale", 2);
            cugan_.reset(new RealCUGAN(
                gpuid_, integer(args, L"sr-tta") != 0, 1));
            if (cugan_->load(required(args, L"sr-param"),
                             required(args, L"sr-model")) != 0)
                throw std::runtime_error("failed to load Real-CUGAN model");
            cugan_->scale = scale;
            cugan_->noise = integer(args, L"sr-noise", -1);
            cugan_->syncgap = integer(args, L"sr-syncgap", 3);
            cugan_->prepadding = scale == 2 ? 18 : scale == 3 ? 14 : 19;
            cugan_->tilesize = cugan_tile(
                gpuid_, scale, integer(args, L"sr-tile"));
            native_w_ = src_w_ * scale;
            native_h_ = src_h_ * scale;
        }
        else if (sr_kind_ == L"realesrgan" || sr_kind_ == L"esrgan")
        {
            const int scale = integer(args, L"sr-scale", 4);
            esrgan_.reset(new RealESRGAN(
                gpuid_, integer(args, L"sr-tta") != 0));
            if (esrgan_->load(required(args, L"sr-param"),
                              required(args, L"sr-model")) != 0)
                throw std::runtime_error("failed to load ESRGAN model");
            esrgan_->scale = scale;
            esrgan_->prepadding = 10;
            esrgan_->tilesize = esrgan_tile(
                gpuid_, integer(args, L"sr-tile"));
            native_w_ = src_w_ * scale;
            native_h_ = src_h_ * scale;
        }
        else if (sr_kind_ == L"none")
        {
            native_w_ = src_w_;
            native_h_ = src_h_;
        }
        else
        {
            throw std::runtime_error("invalid SR kind");
        }
        native_frame_.create(
            native_w_, native_h_, static_cast<size_t>(3), 3);
    }

    std::string gpu_name() const
    {
        return ncnn::get_gpu_info(gpuid_).device_name();
    }

    uint32_t process(uint32_t count, bool skip_first,
                     const std::vector<unsigned char>& pair_modes)
    {
        if (count < 1 || count > static_cast<uint32_t>(max_input_))
            throw std::runtime_error("invalid input frame count");
        if (rife_ && pair_modes.size() != count - 1)
            throw std::runtime_error("invalid pair-mode count");
        if (std::any_of(pair_modes.begin(), pair_modes.end(),
                        [](unsigned char mode) { return mode > 2; }))
            throw std::runtime_error("invalid pair mode");

        const uint32_t output_count =
            (count - 1) * static_cast<uint32_t>(rife_ ? multiplier_ : 1) +
            (skip_first ? 0u : 1u);
        if (output_count > static_cast<uint32_t>(max_output_))
            throw std::runtime_error("output shared memory is too small");

        // FFmpeg/OpenCV expose BGR24 while the upstream NCNN applications
        // load and save RGB PNG data.  Convert the shared-memory input in
        // place; Python overwrites these slots before the next request.
        const size_t source_pixels = static_cast<size_t>(src_w_) * src_h_;
        for (uint32_t index = 0; index < count; ++index)
            swap_red_blue(input_ + index * src_bytes(), source_pixels);

        for (uint32_t index = 0; index < count; ++index)
            source_cache_[index].clear();
        uint32_t output_index = 0;
        if (!skip_first)
            emit_source(0, output_index++);

        const auto convert_output_to_bgr = [this](uint32_t count_to_convert) {
            swap_red_blue(output_, static_cast<size_t>(count_to_convert) *
                          dst_w_ * dst_h_);
        };
        if (!rife_)
        {
            for (uint32_t index = 1; index < count; ++index)
                emit_source(index, output_index++);
            convert_output_to_bgr(output_index);
            return output_index;
        }

        for (uint32_t pair = 0; pair + 1 < count; ++pair)
        {
            for (int step = 1; step < multiplier_; ++step)
            {
                const unsigned char mode = pair_modes[pair];
                if (mode == 1)
                    emit_source(pair, output_index++);
                else if (mode == 2)
                    emit_source(step * 2 <= multiplier_ ? pair : pair + 1,
                                output_index++);
                else
                    emit_interpolated(pair, static_cast<float>(step) /
                                      static_cast<float>(multiplier_),
                                      output_index++);
            }
            emit_source(pair + 1, output_index++);
        }
        convert_output_to_bgr(output_index);
        return output_index;
    }

private:
    size_t src_bytes() const
    {
        return static_cast<size_t>(src_w_) * src_h_ * 3;
    }

    size_t dst_bytes() const
    {
        return static_cast<size_t>(dst_w_) * dst_h_ * 3;
    }

    ncnn::Mat source(uint32_t index) const
    {
        return ncnn::Mat(src_w_, src_h_,
                         input_ + static_cast<size_t>(index) * src_bytes(),
                         static_cast<size_t>(3), 3);
    }

    void enhance(const ncnn::Mat& frame, unsigned char* destination)
    {
        if (sr_kind_ == L"none")
        {
            if (src_w_ == dst_w_ && src_h_ == dst_h_)
                std::memcpy(destination, frame.data, dst_bytes());
            else
                ncnn::resize_bilinear_c3(
                    static_cast<const unsigned char*>(frame.data),
                    frame.w, frame.h, destination, dst_w_, dst_h_);
            return;
        }

        const int status = cugan_ ? cugan_->process(frame, native_frame_)
                                  : esrgan_->process(frame, native_frame_);
        if (status != 0)
            throw std::runtime_error("NCNN super resolution failed");
        if (native_w_ == dst_w_ && native_h_ == dst_h_)
            std::memcpy(destination, native_frame_.data, dst_bytes());
        else
            ncnn::resize_bilinear_c3(
                static_cast<const unsigned char*>(native_frame_.data),
                native_w_, native_h_, destination, dst_w_, dst_h_);
    }

    void emit_source(uint32_t source_index, uint32_t output_index)
    {
        if (output_index >= static_cast<uint32_t>(max_output_))
            throw std::runtime_error("output shared memory is too small");
        auto& cached = source_cache_[source_index];
        if (cached.empty())
        {
            cached.resize(dst_bytes());
            enhance(source(source_index), cached.data());
        }
        std::memcpy(output_ + static_cast<size_t>(output_index) * dst_bytes(),
                    cached.data(), dst_bytes());
    }

    void emit_interpolated(uint32_t pair, float timestep,
                           uint32_t output_index)
    {
        if (output_index >= static_cast<uint32_t>(max_output_))
            throw std::runtime_error("output shared memory is too small");
        if (rife_->process(source(pair), source(pair + 1),
                           timestep, prediction_) != 0)
            throw std::runtime_error("RIFE interpolation failed");
        enhance(prediction_,
                output_ + static_cast<size_t>(output_index) * dst_bytes());
    }

    unsigned char* input_ = nullptr;
    unsigned char* output_ = nullptr;
    int src_w_ = 0;
    int src_h_ = 0;
    int dst_w_ = 0;
    int dst_h_ = 0;
    int native_w_ = 0;
    int native_h_ = 0;
    int max_input_ = 0;
    int max_output_ = 0;
    int multiplier_ = 1;
    int gpuid_ = 0;
    std::wstring sr_kind_;
    std::unique_ptr<RIFE> rife_;
    std::unique_ptr<RealCUGAN> cugan_;
    std::unique_ptr<RealESRGAN> esrgan_;
    ncnn::Mat prediction_;
    ncnn::Mat native_frame_;
    std::vector<std::vector<unsigned char>> source_cache_;
};
} // namespace

int wmain(int argc, wchar_t** argv)
{
    _setmode(_fileno(stdin), _O_BINARY);
    _setmode(_fileno(stdout), _O_BINARY);
    try
    {
        const auto args = parse_args(argc, argv);
        GpuInstance gpu;
        SharedMapping input(required(args, L"input-shm"));
        SharedMapping output(required(args, L"output-shm"));
        {
            Pipeline pipeline(args, input.data(), output.data());
            write_reply(0, 0, pipeline.gpu_name());
            for (;;)
            {
                Request request{};
                if (!read_all(&request, sizeof(request)))
                    break;
                if (request.magic != kMagic || request.version != kVersion)
                    throw std::runtime_error("invalid worker protocol");
                if (request.command == 2)
                    break;
                if (request.command != 1)
                    throw std::runtime_error("invalid worker command");
                std::vector<unsigned char> pair_modes;
                if (request.count > 1)
                {
                    pair_modes.resize(request.count - 1);
                    if (!read_all(pair_modes.data(), pair_modes.size()))
                        throw std::runtime_error("incomplete pair modes");
                }
                try
                {
                    const auto started = std::chrono::steady_clock::now();
                    const uint32_t count = pipeline.process(
                        request.count, request.skip_first != 0, pair_modes);
                    const double elapsed =
                        std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - started).count();
                    write_reply(0, count, "", elapsed);
                }
                catch (const std::exception& error)
                {
                    // The protocol remains synchronized after pair modes have
                    // been consumed, so a failed batch need not kill the worker.
                    write_reply(-1, 0, error.what());
                }
            }
        }
        return 0;
    }
    catch (const std::exception& error)
    {
        write_reply(-1, 0, error.what());
        return 2;
    }
}
