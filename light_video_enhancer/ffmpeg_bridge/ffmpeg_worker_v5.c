/* Final export and packet-draining overlay. */

#define nve_decoder_get_info nve_decoder_get_info_legacy
#define nve_decoder_close nve_decoder_close_legacy
#define nve_encoder_open nve_encoder_open_legacy
#define nve_encoder_write_frame nve_encoder_write_frame_legacy
#define nve_encoder_write_yuv nve_encoder_write_yuv_legacy
#include "ffmpeg_worker_v2.c"
#undef nve_decoder_get_info
#undef nve_decoder_close
#undef nve_encoder_open
#undef nve_encoder_write_frame
#undef nve_encoder_write_yuv

__declspec(dllexport) int nve_decoder_get_info(
    void *handle, int *width, int *height, double *fps, int64_t *frames)
{
    return nve_decoder_get_info_legacy(handle, width, height, fps, frames);
}

__declspec(dllexport) void nve_decoder_close(void *handle)
{
    nve_decoder_close_legacy(handle);
}

__declspec(dllexport) void* nve_encoder_open(
    const char *path, int width, int height, double fps,
    const char *codec, int quality, const char *preset, const char *source)
{
    return nve_encoder_open_legacy(path, width, height, fps, codec,
                                   quality, preset, source);
}

static int write_available_packets(EncoderState *state)
{
    while (1) {
        int result = avcodec_receive_packet(state->codec_ctx, state->pkt);
        if (result == AVERROR(EAGAIN) || result == AVERROR_EOF) return 0;
        if (result < 0) return result;
        /* Some encoders leave this unset. MP4 then treats the last frame as
         * zero-length and marks it for discard during demuxing. */
        if (state->pkt->duration <= 0) state->pkt->duration = 1;
        av_packet_rescale_ts(state->pkt, state->codec_ctx->time_base,
                             state->stream->time_base);
        state->pkt->stream_index = state->stream->index;
        result = av_interleaved_write_frame(state->fmt_ctx, state->pkt);
        av_packet_unref(state->pkt);
        if (result < 0) return result;
    }
}

static int send_video_frame(EncoderState *state)
{
    while (1) {
        int result = avcodec_send_frame(state->codec_ctx, state->frame);
        if (result == AVERROR(EAGAIN)) {
            result = write_available_packets(state);
            if (result < 0) return result;
            continue;
        }
        if (result < 0) return result;
        return write_available_packets(state);
    }
}

__declspec(dllexport) int nve_encoder_write_frame(
    void *handle, const uint8_t *bgr_data, int width, int height)
{
    EncoderState *state = (EncoderState*)handle;
    if (!state || !bgr_data) return -1;
    if (!state->sws_ctx || state->sws_src_w != width || state->sws_src_h != height) {
        if (state->sws_ctx) sws_freeContext(state->sws_ctx);
        state->sws_ctx = sws_getContext(width, height, AV_PIX_FMT_BGR24,
            state->enc_w, state->enc_h, AV_PIX_FMT_YUV420P,
            SWS_BILINEAR, NULL, NULL, NULL);
        state->sws_src_w = width;
        state->sws_src_h = height;
    }
    if (!state->sws_ctx || av_frame_make_writable(state->frame) < 0) return -2;
    {
        const uint8_t *source[4] = {bgr_data, NULL, NULL, NULL};
        int source_stride[4] = {width * 3, 0, 0, 0};
        sws_scale(state->sws_ctx, source, source_stride, 0, height,
                  state->frame->data, state->frame->linesize);
    }
    state->frame->pts = state->pts++;
    state->frame->duration = 1;
    return send_video_frame(state);
}

__declspec(dllexport) int nve_encoder_write_yuv(
    void *handle, const uint8_t *data, int width, int height)
{
    EncoderState *state = (EncoderState*)handle;
    int y_size, uv_size, row;
    const uint8_t *source_y, *source_u, *source_v;
    if (!state || !data) return -1;
    if (width != state->enc_w || height != state->enc_h) return -2;
    if (av_frame_make_writable(state->frame) < 0) return -3;
    y_size = width * height;
    uv_size = y_size / 4;
    source_y = data;
    source_u = data + y_size;
    source_v = data + y_size + uv_size;
    for (row = 0; row < height; row++)
        memcpy(state->frame->data[0] + row * state->frame->linesize[0],
               source_y + row * width, width);
    for (row = 0; row < height / 2; row++) {
        memcpy(state->frame->data[1] + row * state->frame->linesize[1],
               source_u + row * (width / 2), width / 2);
        memcpy(state->frame->data[2] + row * state->frame->linesize[2],
               source_v + row * (width / 2), width / 2);
    }
    state->frame->pts = state->pts++;
    state->frame->duration = 1;
    return send_video_frame(state);
}
