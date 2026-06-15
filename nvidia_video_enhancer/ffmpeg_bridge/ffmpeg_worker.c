/**
 * ffmpeg_worker.c — FFmpeg 解码/编码 C 包装器
 *
 * 编译为独立的 DLL，通过 -static 链接避免 MinGW 运行时依赖问题。
 * 暴露极简的 C API 供 Python ctypes 调用，无需在 Python 中定义
 * 脆弱的 FFmpeg 结构体布局。
 */

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define __STDC_CONSTANT_MACROS
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libswscale/swscale.h>
#include <libavutil/hwcontext.h>
#include <libavutil/imgutils.h>
#include <libavutil/opt.h>
#include <libavutil/pixdesc.h>

// ============================================================
// 解码器
// ============================================================
typedef struct {
    AVFormatContext *fmt_ctx;
    AVCodecContext  *codec_ctx;
    AVBufferRef     *hw_device_ctx;
    struct SwsContext *sws_ctx;
    AVFrame *frame;
    AVFrame *sw_frame;
    AVPacket *pkt;
    int video_stream_idx;
    int width, height;
    double fps;
    int64_t total_frames;
    uint8_t *out_buf;
    int out_buf_size;
    int sws_src_w, sws_src_h, sws_src_fmt;
} DecoderState;

void* nve_decoder_open(const char *path, int use_hw)
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
            s->video_stream_idx = i;
            break;
        }
    }
    if (s->video_stream_idx < 0)
        { avformat_close_input(&s->fmt_ctx); free(s); return NULL; }

    AVStream *st = s->fmt_ctx->streams[s->video_stream_idx];
    AVCodecParameters *par = st->codecpar;

    s->width  = par->width;
    s->height = par->height;
    s->fps    = av_q2d(st->r_frame_rate);
    if (s->fps <= 0) s->fps = av_q2d(st->avg_frame_rate);
    if (s->fps <= 0) s->fps = 30.0;
    s->total_frames = st->nb_frames;
    if (s->total_frames <= 0)
        s->total_frames = (int64_t)(s->fmt_ctx->duration * s->fps / AV_TIME_BASE);

    const AVCodec *codec = avcodec_find_decoder(par->codec_id);
    if (!codec)
        { avformat_close_input(&s->fmt_ctx); free(s); return NULL; }

    s->codec_ctx = avcodec_alloc_context3(codec);
    avcodec_parameters_to_context(s->codec_ctx, par);
    s->codec_ctx->pkt_timebase = st->time_base;

    if (use_hw) {
        int ret = av_hwdevice_ctx_create(&s->hw_device_ctx,
            AV_HWDEVICE_TYPE_CUDA, NULL, NULL, 0);
        if (ret >= 0)
            s->codec_ctx->hw_device_ctx = av_buffer_ref(s->hw_device_ctx);
    }

    if (avcodec_open2(s->codec_ctx, codec, NULL) < 0)
        { avcodec_free_context(&s->codec_ctx); avformat_close_input(&s->fmt_ctx); free(s); return NULL; }

    s->frame    = av_frame_alloc();
    s->sw_frame = av_frame_alloc();
    s->pkt      = av_packet_alloc();

    s->out_buf_size = s->width * s->height * 3;
    s->out_buf = (uint8_t*)malloc(s->out_buf_size);

    return s;
}

int nve_decoder_get_info(void *handle, int *w, int *h, double *fps, int64_t *frames)
{
    DecoderState *s = (DecoderState*)handle;
    if (!s) return -1;
    *w = s->width;
    *h = s->height;
    *fps = s->fps;
    *frames = s->total_frames;
    return 0;
}

uint8_t* nve_decoder_read_frame(void *handle)
{
    DecoderState *s = (DecoderState*)handle;
    if (!s) return NULL;

    while (1) {
        int ret = av_read_frame(s->fmt_ctx, s->pkt);
        if (ret < 0) return NULL;

        if (s->pkt->stream_index != s->video_stream_idx) {
            av_packet_unref(s->pkt);
            continue;
        }

        ret = avcodec_send_packet(s->codec_ctx, s->pkt);
        av_packet_unref(s->pkt);
        if (ret < 0) continue;

        while (1) {
            ret = avcodec_receive_frame(s->codec_ctx, s->frame);
            if (ret < 0) break;

            AVFrame *src = s->frame;
            int src_w = s->frame->width;
            int src_h = s->frame->height;
            enum AVPixelFormat src_fmt = s->frame->format;

            if (s->hw_device_ctx && src_fmt == AV_PIX_FMT_CUDA) {
                av_frame_unref(s->sw_frame);
                av_hwframe_transfer_data(s->sw_frame, s->frame, 0);
                src = s->sw_frame;
                src_w = s->sw_frame->width;
                src_h = s->sw_frame->height;
                src_fmt = s->sw_frame->format;
            }

            if (!s->sws_ctx || s->sws_src_w != src_w || s->sws_src_h != src_h ||
                s->sws_src_fmt != src_fmt) {
                if (s->sws_ctx) sws_freeContext(s->sws_ctx);
                s->sws_ctx = sws_getContext(src_w, src_h, src_fmt,
                    s->width, s->height, AV_PIX_FMT_BGR24,
                    SWS_BILINEAR | SWS_ACCURATE_RND, NULL, NULL, NULL);
                s->sws_src_w = src_w;
                s->sws_src_h = src_h;
                s->sws_src_fmt = src_fmt;
            }
            if (s->sws_ctx) {
                int src_linesize[4] = {src->linesize[0], src->linesize[1],
                                       src->linesize[2], src->linesize[3]};
                const uint8_t *src_data[4] = {src->data[0], src->data[1],
                                              src->data[2], src->data[3]};
                int dst_linesize[1] = {s->width * 3};
                uint8_t *dst_data[1] = {s->out_buf};
                sws_scale(s->sws_ctx, src_data, src_linesize, 0, src_h,
                          dst_data, dst_linesize);
                return s->out_buf;
            }
        }
    }
}

void nve_decoder_close(void *handle)
{
    DecoderState *s = (DecoderState*)handle;
    if (!s) return;
    if (s->sws_ctx) sws_freeContext(s->sws_ctx);
    if (s->pkt) av_packet_free(&s->pkt);
    if (s->sw_frame) av_frame_free(&s->sw_frame);
    if (s->frame) av_frame_free(&s->frame);
    if (s->codec_ctx) avcodec_free_context(&s->codec_ctx);
    if (s->hw_device_ctx) av_buffer_unref(&s->hw_device_ctx);
    if (s->fmt_ctx) avformat_close_input(&s->fmt_ctx);
    free(s->out_buf);
    free(s);
}

// ============================================================
// 编码器
// ============================================================
typedef struct {
    AVFormatContext *fmt_ctx;
    AVCodecContext  *codec_ctx;
    struct SwsContext *sws_ctx;
    int sws_src_w, sws_src_h;
    AVStream *stream;
    AVFrame *frame;
    AVPacket *pkt;
    int64_t pts;
    int dst_w, dst_h;
    int enc_w, enc_h;
    int audio_source_idx;
    int audio_count;
    int *audio_indices;
    char *source_path;
} EncoderState;

void* nve_encoder_open(const char *path, int w, int h, double fps,
                       const char *codec_name, int crf, const char *preset,
                       const char *source_path)
{
    EncoderState *s = (EncoderState*)calloc(1, sizeof(EncoderState));
    if (!s) return NULL;

    if (source_path)
        s->source_path = strdup(source_path);

    s->dst_w = w; s->dst_h = h;
    s->enc_w = (w / 2) * 2;
    s->enc_h = (h / 2) * 2;

    const char *fmt_name = NULL;
    const char *ext = strrchr(path, '.') ? strrchr(path, '.') + 1 : "mp4";
    if (!strcmp(ext, "mkv")) fmt_name = "matroska";
    else if (!strcmp(ext, "mov")) fmt_name = "mov";
    else if (!strcmp(ext, "avi")) fmt_name = "avi";
    else fmt_name = "mp4";

    if (avformat_alloc_output_context2(&s->fmt_ctx, NULL, fmt_name, path) < 0)
        { free(s); return NULL; }

    const AVCodec *codec = avcodec_find_encoder_by_name(codec_name);
    if (!codec) codec = avcodec_find_encoder(AV_CODEC_ID_H264);
    if (!codec)
        { avformat_free_context(s->fmt_ctx); free(s); return NULL; }

    s->stream = avformat_new_stream(s->fmt_ctx, NULL);
    s->codec_ctx = avcodec_alloc_context3(codec);
    s->codec_ctx->width    = s->enc_w;
    s->codec_ctx->height   = s->enc_h;
    {
        AVRational fr = av_d2q(fps, 1000000);
        s->codec_ctx->time_base = av_inv_q(fr);
        s->codec_ctx->framerate = fr;
    }
    s->codec_ctx->gop_size = (int)(fps * 2.0 + 0.5);
    s->codec_ctx->max_b_frames = 0;

    s->codec_ctx->pix_fmt = AV_PIX_FMT_YUV420P;

    AVDictionary *opts = NULL;
    if (preset)
        av_dict_set(&opts, "preset", preset, 0);
    char crf_str[16];
    snprintf(crf_str, sizeof(crf_str), "%d", crf);

    av_dict_set(&opts, "pix_fmt", "yuv420p", 0);

    if (strstr(codec_name, "nvenc")) {
        av_dict_set(&opts, "cq", crf_str, 0);
        av_dict_set(&opts, "rc", "vbr", 0);
        av_dict_set(&opts, "tune", "ll", 0);
        av_dict_set(&opts, "profile", "high", 0);
        av_dict_set(&opts, "async_depth", "4", 0);
        av_dict_set(&opts, "no-scenecut", "1", 0);
    } else {
        av_dict_set(&opts, "crf", crf_str, 0);
    }

    if (avcodec_open2(s->codec_ctx, codec, &opts) < 0) {
        av_dict_free(&opts);
        avcodec_free_context(&s->codec_ctx);
        avformat_free_context(s->fmt_ctx);
        free(s);
        return NULL;
    }
    av_dict_free(&opts);

    avcodec_parameters_from_context(s->stream->codecpar, s->codec_ctx);
    s->stream->time_base = s->codec_ctx->time_base;

    fprintf(stderr, "[Encoder] %dx%d pix=%d tb=%d/%d rate=%d/%d\n",
        s->enc_w, s->enc_h, s->codec_ctx->pix_fmt,
        s->codec_ctx->time_base.num, s->codec_ctx->time_base.den,
        s->codec_ctx->framerate.num, s->codec_ctx->framerate.den);

    if (s->source_path) {
        AVFormatContext *probe_ctx = NULL;
        if (avformat_open_input(&probe_ctx, s->source_path, NULL, NULL) >= 0) {
            if (avformat_find_stream_info(probe_ctx, NULL) >= 0) {
                s->audio_count = 0;
                for (unsigned i = 0; i < probe_ctx->nb_streams; i++)
                    if (probe_ctx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_AUDIO)
                        s->audio_count++;
                s->audio_indices = (int*)malloc(sizeof(int) * s->audio_count);
                int idx = 0;
                for (unsigned i = 0; i < probe_ctx->nb_streams; i++) {
                    if (probe_ctx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_AUDIO) {
                        s->audio_indices[idx] = (int)i;
                        AVStream *out_st = avformat_new_stream(s->fmt_ctx, NULL);
                        avcodec_parameters_copy(out_st->codecpar, probe_ctx->streams[i]->codecpar);
                        if (probe_ctx->streams[i]->time_base.num > 0 && probe_ctx->streams[i]->time_base.den > 0)
                            out_st->time_base = probe_ctx->streams[i]->time_base;
                        else
                            out_st->time_base = (AVRational){1, 48000};
                        idx++;
                    }
                }
            }
            avformat_close_input(&probe_ctx);
        }
    }

    if (!(s->fmt_ctx->oformat->flags & AVFMT_NOFILE)) {
        if (avio_open(&s->fmt_ctx->pb, path, AVIO_FLAG_WRITE) < 0) {
            avcodec_free_context(&s->codec_ctx);
            avformat_free_context(s->fmt_ctx);
            free(s);
            return NULL;
        }
    }

    if (avformat_write_header(s->fmt_ctx, NULL) < 0) {
        avcodec_free_context(&s->codec_ctx);
        avio_closep(&s->fmt_ctx->pb);
        avformat_free_context(s->fmt_ctx);
        free(s);
        return NULL;
    }

    s->frame = av_frame_alloc();
    s->frame->format = s->codec_ctx->pix_fmt;
    s->frame->width  = s->enc_w;
    s->frame->height = s->enc_h;
    av_frame_get_buffer(s->frame, 0);

    s->pkt = av_packet_alloc();
    s->pts = 0;

    return s;
}

int nve_encoder_write_frame(void *handle, const uint8_t *bgr_data, int w, int h)
{
    EncoderState *s = (EncoderState*)handle;
    if (!s) return -1;

    if (!s->sws_ctx || s->sws_src_w != w || s->sws_src_h != h) {
        if (s->sws_ctx) sws_freeContext(s->sws_ctx);
        s->sws_ctx = sws_getContext(w, h, AV_PIX_FMT_BGR24,
            s->enc_w, s->enc_h, AV_PIX_FMT_YUV420P,
            SWS_BILINEAR, NULL, NULL, NULL);
        s->sws_src_w = w; s->sws_src_h = h;
    }

    int src_linesize[1] = {w * 3};
    const uint8_t *src_data[1] = {bgr_data};

    av_frame_make_writable(s->frame);
    int dst_linesize[4] = {s->frame->linesize[0], s->frame->linesize[1],
                           s->frame->linesize[2], s->frame->linesize[3]};
    uint8_t *dst_data[4] = {s->frame->data[0], s->frame->data[1],
                            s->frame->data[2], s->frame->data[3]};

    sws_scale(s->sws_ctx, src_data, src_linesize, 0, h, dst_data, dst_linesize);

    s->frame->pts = s->pts++;
    avcodec_send_frame(s->codec_ctx, s->frame);

    while (1) {
        int ret = avcodec_receive_packet(s->codec_ctx, s->pkt);
        if (ret < 0) break;
        av_packet_rescale_ts(s->pkt, s->codec_ctx->time_base, s->stream->time_base);
        s->pkt->stream_index = s->stream->index;
        av_interleaved_write_frame(s->fmt_ctx, s->pkt);
        av_packet_unref(s->pkt);
    }
    return 0;
}

/*
 * nve_encoder_write_yuv — feed YUV420p planar bytes (Python already converted).
 * yuv_data layout: [Y plane w*h] [U plane w*h/4] [V plane w*h/4]
 * Much faster than write_frame because sws_scale is skipped.
 */
int nve_encoder_write_yuv(void *handle, const uint8_t *yuv_data, int w, int h)
{
    EncoderState *s = (EncoderState*)handle;
    if (!s) return -1;

    av_frame_make_writable(s->frame);

    int y_size  = w * h;
    int uv_size = y_size / 4;
    int y_ls  = s->frame->linesize[0];
    int u_ls  = s->frame->linesize[1];
    int v_ls  = s->frame->linesize[2];

    const uint8_t *src_y  = yuv_data;
    const uint8_t *src_u  = yuv_data + y_size;
    const uint8_t *src_v  = yuv_data + y_size + uv_size;

    for (int row = 0; row < h; row++)
        memcpy(s->frame->data[0] + row * y_ls, src_y + row * w, w);
    for (int row = 0; row < h / 2; row++)
        memcpy(s->frame->data[1] + row * u_ls, src_u + row * (w / 2), w / 2);
    for (int row = 0; row < h / 2; row++)
        memcpy(s->frame->data[2] + row * v_ls, src_v + row * (w / 2), w / 2);

    s->frame->pts = s->pts++;
    avcodec_send_frame(s->codec_ctx, s->frame);

    while (1) {
        int ret = avcodec_receive_packet(s->codec_ctx, s->pkt);
        if (ret < 0) break;
        av_packet_rescale_ts(s->pkt, s->codec_ctx->time_base, s->stream->time_base);
        s->pkt->stream_index = s->stream->index;
        av_interleaved_write_frame(s->fmt_ctx, s->pkt);
        av_packet_unref(s->pkt);
    }
    return 0;
}

void nve_encoder_close(void *handle)
{
    EncoderState *s = (EncoderState*)handle;
    if (!s) return;

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
        AVFormatContext *src_ctx = NULL;
        if (avformat_open_input(&src_ctx, s->source_path, NULL, NULL) >= 0) {
            if (avformat_find_stream_info(src_ctx, NULL) >= 0) {
                AVPacket *apkt = av_packet_alloc();
                while (av_read_frame(src_ctx, apkt) >= 0) {
                    int is_audio = 0;
                    int out_idx = -1;
                    for (int ai = 0; ai < s->audio_count; ai++) {
                        if (apkt->stream_index == s->audio_indices[ai]) {
                            is_audio = 1;
                            out_idx = (int)s->fmt_ctx->nb_streams - s->audio_count + ai;
                            break;
                        }
                    }
                    if (is_audio && out_idx >= 0) {
                        AVStream *in_st = src_ctx->streams[apkt->stream_index];
                        AVStream *out_st = s->fmt_ctx->streams[out_idx];
                        if (in_st->time_base.den > 0 && out_st->time_base.den > 0)
                            av_packet_rescale_ts(apkt, in_st->time_base, out_st->time_base);
                        apkt->stream_index = out_idx;
                        av_interleaved_write_frame(s->fmt_ctx, apkt);
                    }
                    av_packet_unref(apkt);
                }
                av_packet_free(&apkt);
            }
            avformat_close_input(&src_ctx);
        }
    }

    if (s->fmt_ctx) {
        av_write_trailer(s->fmt_ctx);
    }

    if (s->sws_ctx) sws_freeContext(s->sws_ctx);
    if (s->pkt) av_packet_free(&s->pkt);
    if (s->frame) av_frame_free(&s->frame);
    if (s->codec_ctx) avcodec_free_context(&s->codec_ctx);
    if (s->fmt_ctx) {
        if (s->fmt_ctx->pb)
            avio_closep(&s->fmt_ctx->pb);
        avformat_free_context(s->fmt_ctx);
    }
    if (s->source_path) free(s->source_path);
    if (s->audio_indices) free(s->audio_indices);
    free(s);
}
