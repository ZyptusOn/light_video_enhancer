#pragma once

#include <string>

#include "net.h"

class Span
{
public:
    Span(int gpuid, int scale, int tilesize);

    int load(const std::wstring& param_path, const std::wstring& model_path);
    int process(const ncnn::Mat& input, ncnn::Mat& output) const;

private:
    int process_tile(const unsigned char* pixels, int width, int height,
                     int stride, unsigned char* output) const;

    ncnn::Net net_;
    int scale_ = 2;
    int tilesize_ = 0;
};
