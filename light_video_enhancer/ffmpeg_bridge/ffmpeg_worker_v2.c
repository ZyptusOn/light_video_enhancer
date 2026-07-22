/**
 * Compatibility overlay for ffmpeg_worker.c.
 *
 * Keeps the original implementation readable while replacing three ABI
 * entry points that need state-machine correctness or newer capabilities.
 */

#define nve_decoder_open nve_decoder_open_legacy
#define nve_decoder_read_frame nve_decoder_read_frame_legacy
#define nve_encoder_close nve_encoder_close_legacy
#include "ffmpeg_worker.c"
#undef nve_decoder_open
#undef nve_decoder_read_frame
#undef nve_encoder_close

typedef struct AudioRangeEntry {
    void *handle;
    double start;
    double duration;
    struct AudioRangeEntry *next;
} AudioRangeEntry;

static AudioRangeEntry *g_audio_ranges = NULL;

static AudioRangeEntry *find_audio_range(void *handle)
{
    for (AudioRangeEntry *item = g_audio_ranges; item; item = item->next)
        if (item->handle == handle) return item;
    return NULL;
}

__declspec(dllexport) void nve_encoder_set_audio_range(
    void *handle, double start, double duration)
{
    if (!handle) return;
    AudioRangeEntry *item = find_audio_range(handle);
    if (!item) {
        item = (AudioRangeEntry*)calloc(1, sizeof(AudioRangeEntry));
        if (!item) return;
        item->handle = handle;
        item->next = g_audio_ranges;
        g_audio_ranges = item;
    }
    item->start = start > 0.0 ? start : 0.0;
    item->duration = duration;
}

static void remove_audio_range(void *handle)
{
    AudioRangeEntry **cursor = &g_audio_ranges;
    while (*cursor) {
        if ((*cursor)->handle == handle) {
            AudioRangeEntry *old = *cursor;
            *cursor = old->next;
            free(old);
            return;
        }
        cursor = &(*cursor)->next;
    }
}

__declspec(dllexport) int nve_encoder_is_available(const char *codec_name)
{
    return codec_name && avcodec_find_encoder_by_name(codec_name) != NULL;
}

__declspec(dllexport) void* nve_decoder_open(const char *path, int hw_mode)
{
    DecoderState *s = (DecoderState*)calloc(1, sizeof(DecoderState));
    if (!s) return NULL;
    if (avformat_open_input(&s->fmt_ctx, path, NULL, NULL) < 0)
        { free(s); return NULL; }
    if (avformat_find_stream_info(s->fmt_ctx, NULL) < 0)
        { avformat_close_input(&s->fmt_ctx); free(s); return NULL; }

    s->video_stream_idx = -1;
    for (unsigned i = 0; i < s->fmt_ctx->nb_streams; i++) {
        if (s->fmt_ctx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
            s->video_stream_idx = (int)i;
            break;
        }
    }
    if (s->video_stream_idx < 0)
        { avformat_close_input(&s->fmt_ctx); free(s); return NULL; }

    AVStream *stream = s->fmt_ctx->streams[s->video_stream_idx];
    AVCodecParameters *parameters = stream->codecpar;
    s->width = parameters->width;
    s->height = parameters->height;
    s->fps = av_q2d(stream->avg_frame_rate);
    if (s->fps <= 0) s->fps = av_q2d(stream->r_frame_rate);
    if (s->fps <= 0) s->fps = 30.0;
    s->total_frames = stream->nb_frames;
    if (s->total_frames <= 0 && s->fmt_ctx->duration > 0)
        s->total_frames = (int64_t)(s->fmt_ctx->duration * s->fps / AV_TIME_BASE + 0.5);

    const AVCodec *codec = NULL;
    /* FFmpeg's native AV1 decoder is hardware-only. Prefer dav1d so AV1
     * input remains usable on Windows 7 and on GPUs without AV1 decode. */
    if (parameters->codec_id == AV_CODEC_ID_AV1)
        codec = avcodec_find_decoder_by_name("libdav1d");
    if (!codec)
        codec = avcodec_find_decoder(parameters->codec_id);
    if (!codec) goto fail;
    s->codec_ctx = avcodec_alloc_context3(codec);
    if (!s->codec_ctx) goto fail;
    if (avcodec_parameters_to_context(s->codec_ctx, parameters) < 0) goto fail;
    s->codec_ctx->pkt_timebase = stream->time_base;
    s->hw_mode = hw_mode;
    s->codec_ctx->opaque = s;
    s->codec_ctx->get_format = nve_decoder_get_format;

    if (hw_mode) {
        enum AVHWDeviceType type = hw_mode == 1 ? AV_HWDEVICE_TYPE_CUDA : AV_HWDEVICE_TYPE_D3D11VA;
        if (av_hwdevice_ctx_create(&s->hw_device_ctx, type, NULL, NULL, 0) >= 0)
            s->codec_ctx->hw_device_ctx = av_buffer_ref(s->hw_device_ctx);
    }
    if (avcodec_open2(s->codec_ctx, codec, NULL) < 0) {
        if (s->codec_ctx->hw_device_ctx) av_buffer_unref(&s->codec_ctx->hw_device_ctx);
        if (s->hw_device_ctx) av_buffer_unref(&s->hw_device_ctx);
        if (avcodec_open2(s->codec_ctx, codec, NULL) < 0) goto fail;
    }
    s->frame = av_frame_alloc();
    s->sw_frame = av_frame_alloc();
    s->pkt = av_packet_alloc();
    s->out_buf_size = s->width * s->height * 3;
    s->out_buf = (uint8_t*)malloc(s->out_buf_size);
    if (!s->frame || !s->sw_frame || !s->pkt || !s->out_buf) goto fail;
    return s;

fail:
    if (s->sws_ctx) sws_freeContext(s->sws_ctx);
    if (s->pkt) av_packet_free(&s->pkt);
    if (s->sw_frame) av_frame_free(&s->sw_frame);
    if (s->frame) av_frame_free(&s->frame);
    if (s->codec_ctx) avcodec_free_context(&s->codec_ctx);
    if (s->hw_device_ctx) av_buffer_unref(&s->hw_device_ctx);
    if (s->fmt_ctx) avformat_close_input(&s->fmt_ctx);
    free(s->out_buf);
    free(s);
    return NULL;
}

static uint8_t *convert_decoded_frame(DecoderState *s)
{
    AVFrame *source = s->frame;
    enum AVPixelFormat format = (enum AVPixelFormat)s->frame->format;
    if (s->frame->hw_frames_ctx) {
        av_frame_unref(s->sw_frame);
        if (av_hwframe_transfer_data(s->sw_frame, s->frame, 0) < 0) return NULL;
        source = s->sw_frame;
        format = (enum AVPixelFormat)source->format;
    }
    int width = source->width;
    int height = source->height;
    if (!s->sws_ctx || s->sws_src_w != width || s->sws_src_h != height ||
        s->sws_src_fmt != format) {
        if (s->sws_ctx) sws_freeContext(s->sws_ctx);
        s->sws_ctx = sws_getContext(width, height, format, s->width, s->height,
            AV_PIX_FMT_BGR24, SWS_BILINEAR | SWS_ACCURATE_RND, NULL, NULL, NULL);
        s->sws_src_w = width;
        s->sws_src_h = height;
        s->sws_src_fmt = format;
    }
    if (!s->sws_ctx) return NULL;
    const uint8_t *source_data[4] = {
        source->data[0], source->data[1], source->data[2], source->data[3]
    };
    int source_linesize[4] = {
        source->linesize[0], source->linesize[1], source->linesize[2], source->linesize[3]
    };
    uint8_t *target_data[4] = {s->out_buf, NULL, NULL, NULL};
    int target_linesize[4] = {s->width * 3, 0, 0, 0};
    sws_scale(s->sws_ctx, source_data, source_linesize, 0, height,
              target_data, target_linesize);
    return s->out_buf;
}

__declspec(dllexport) uint8_t* nve_decoder_read_frame(void *handle)
{
    DecoderState *s = (DecoderState*)handle;
    if (!s) return NULL;
    while (1) {
        int result = avcodec_receive_frame(s->codec_ctx, s->frame);
        if (result == 0) return convert_decoded_frame(s);
        if (result == AVERROR_EOF) return NULL;
        if (result != AVERROR(EAGAIN)) return NULL;

        result = av_read_frame(s->fmt_ctx, s->pkt);
        if (result < 0) {
            avcodec_send_packet(s->codec_ctx, NULL);
            result = avcodec_receive_frame(s->codec_ctx, s->frame);
            if (result == 0) return convert_decoded_frame(s);
            return NULL;
        }
        if (s->pkt->stream_index != s->video_stream_idx) {
            av_packet_unref(s->pkt);
            continue;
        }
        result = avcodec_send_packet(s->codec_ctx, s->pkt);
        av_packet_unref(s->pkt);
        if (result < 0 && result != AVERROR(EAGAIN)) continue;
    }
}

__declspec(dllexport) void nve_encoder_close(void *handle)
{
    EncoderState *s = (EncoderState*)handle;
    if (!s) return;
    AudioRangeEntry *range = find_audio_range(handle);
    double start = range ? range->start : 0.0;
    double duration = range ? range->duration : -1.0;
    double end = duration >= 0.0 ? start + duration : -1.0;

    if (s->codec_ctx) {
        avcodec_send_frame(s->codec_ctx, NULL);
        while (s->pkt && s->stream && s->fmt_ctx &&
               avcodec_receive_packet(s->codec_ctx, s->pkt) >= 0) {
            av_packet_rescale_ts(s->pkt, s->codec_ctx->time_base, s->stream->time_base);
            s->pkt->stream_index = s->stream->index;
            av_interleaved_write_frame(s->fmt_ctx, s->pkt);
            av_packet_unref(s->pkt);
        }
    }

    if (s->fmt_ctx && s->source_path && s->audio_count > 0) {
        AVFormatContext *source_context = NULL;
        if (avformat_open_input(&source_context, s->source_path, NULL, NULL) >= 0 &&
            avformat_find_stream_info(source_context, NULL) >= 0) {
            AVPacket *packet = av_packet_alloc();
            while (packet && av_read_frame(source_context, packet) >= 0) {
                int audio_index = -1;
                for (int index = 0; index < s->audio_count; index++) {
                    if (packet->stream_index == s->audio_indices[index]) {
                        audio_index = index;
                        break;
                    }
                }
                if (audio_index >= 0) {
                    AVStream *input_stream = source_context->streams[packet->stream_index];
                    int64_t timestamp = packet->pts != AV_NOPTS_VALUE ? packet->pts : packet->dts;
                    double seconds = timestamp == AV_NOPTS_VALUE ? start :
                        timestamp * av_q2d(input_stream->time_base);
                    if (seconds >= start && (end < 0.0 || seconds < end)) {
                        int64_t shift = av_rescale_q((int64_t)(start * AV_TIME_BASE),
                                                     AV_TIME_BASE_Q, input_stream->time_base);
                        if (packet->pts != AV_NOPTS_VALUE) packet->pts -= shift;
                        if (packet->dts != AV_NOPTS_VALUE) packet->dts -= shift;
                        AVStream *output_stream = s->fmt_ctx->streams[
                            (int)s->fmt_ctx->nb_streams - s->audio_count + audio_index];
                        av_packet_rescale_ts(packet, input_stream->time_base,
                                             output_stream->time_base);
                        packet->stream_index = output_stream->index;
                        av_interleaved_write_frame(s->fmt_ctx, packet);
                    }
                }
                av_packet_unref(packet);
            }
            if (packet) av_packet_free(&packet);
        }
        if (source_context) avformat_close_input(&source_context);
    }
    if (s->fmt_ctx) av_write_trailer(s->fmt_ctx);
    if (s->sws_ctx) sws_freeContext(s->sws_ctx);
    if (s->pkt) av_packet_free(&s->pkt);
    if (s->frame) av_frame_free(&s->frame);
    if (s->codec_ctx) avcodec_free_context(&s->codec_ctx);
    if (s->fmt_ctx) {
        if (s->fmt_ctx->pb) avio_closep(&s->fmt_ctx->pb);
        avformat_free_context(s->fmt_ctx);
    }
    free(s->source_path);
    free(s->audio_indices);
    remove_audio_range(handle);
    free(s);
}
