/* Decoder packet retention and final container timestamp fixes. */

#include "ffmpeg_worker_v7.c"

typedef struct DecoderReadState {
    void *handle;
    int packet_pending;
    int input_eof;
    int flush_sent;
    struct DecoderReadState *next;
} DecoderReadState;

static DecoderReadState *decoder_read_states = NULL;

static DecoderReadState *get_decoder_read_state(void *handle)
{
    DecoderReadState *entry = decoder_read_states;
    while (entry) {
        if (entry->handle == handle) return entry;
        entry = entry->next;
    }
    entry = (DecoderReadState*)calloc(1, sizeof(DecoderReadState));
    if (!entry) return NULL;
    entry->handle = handle;
    entry->next = decoder_read_states;
    decoder_read_states = entry;
    return entry;
}

static void remove_decoder_read_state(void *handle)
{
    DecoderReadState **link = &decoder_read_states;
    while (*link) {
        if ((*link)->handle == handle) {
            DecoderReadState *dead = *link;
            *link = dead->next;
            free(dead);
            return;
        }
        link = &(*link)->next;
    }
}

__declspec(dllexport) uint8_t *nve_decoder_read_frame2(void *handle)
{
    DecoderState *state = (DecoderState*)handle;
    DecoderReadState *read_state;
    if (!state) return NULL;
    read_state = get_decoder_read_state(handle);
    if (!read_state) return NULL;

    while (1) {
        int result = avcodec_receive_frame(state->codec_ctx, state->frame);
        if (result == 0) return convert_decoded_frame(state);
        if (result == AVERROR_EOF) return NULL;
        if (result != AVERROR(EAGAIN)) return NULL;

        if (read_state->packet_pending) {
            result = avcodec_send_packet(state->codec_ctx, state->pkt);
            if (result == 0) {
                av_packet_unref(state->pkt);
                read_state->packet_pending = 0;
            } else if (result != AVERROR(EAGAIN)) {
                av_packet_unref(state->pkt);
                read_state->packet_pending = 0;
            }
            continue;
        }

        if (read_state->input_eof) {
            if (!read_state->flush_sent) {
                result = avcodec_send_packet(state->codec_ctx, NULL);
                if (result == 0 || result == AVERROR_EOF)
                    read_state->flush_sent = 1;
                else if (result != AVERROR(EAGAIN))
                    return NULL;
                continue;
            }
            return NULL;
        }

        result = av_read_frame(state->fmt_ctx, state->pkt);
        if (result < 0) {
            read_state->input_eof = 1;
            continue;
        }
        if (state->pkt->stream_index != state->video_stream_idx) {
            av_packet_unref(state->pkt);
            continue;
        }
        read_state->packet_pending = 1;
    }
}

__declspec(dllexport) void nve_decoder_close2(void *handle)
{
    remove_decoder_read_state(handle);
    nve_decoder_close(handle);
}

__declspec(dllexport) void nve_encoder_finish3(void *handle)
{
    EncoderState *state = (EncoderState*)handle;
    AudioRangeEntry *range;
    double start;
    double duration;
    double end;
    if (!state) return;

    range = find_audio_range(handle);
    start = range ? range->start : 0.0;
    duration = range ? range->duration : -1.0;
    end = duration >= 0.0 ? start + duration : -1.0;

    if (state->codec_ctx) {
        while (1) {
            int result = avcodec_send_frame(state->codec_ctx, NULL);
            if (result == AVERROR(EAGAIN)) {
                if (write_available_packets(state) < 0) break;
                continue;
            }
            break;
        }
        write_available_packets(state);
    }

    if (state->fmt_ctx && state->source_path && state->audio_count > 0) {
        AVFormatContext *source_context = NULL;
        if (avformat_open_input(&source_context, state->source_path, NULL, NULL) >= 0 &&
            avformat_find_stream_info(source_context, NULL) >= 0) {
            AVPacket *packet = av_packet_alloc();
            while (packet && av_read_frame(source_context, packet) >= 0) {
                int audio_index = -1;
                int index;
                for (index = 0; index < state->audio_count; index++) {
                    if (packet->stream_index == state->audio_indices[index]) {
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
                        int64_t shift = av_rescale_q(
                            (int64_t)(start * AV_TIME_BASE), AV_TIME_BASE_Q,
                            input_stream->time_base);
                        AVStream *output_stream = state->fmt_ctx->streams[
                            (int)state->fmt_ctx->nb_streams - state->audio_count + audio_index];
                        if (packet->pts != AV_NOPTS_VALUE) packet->pts -= shift;
                        if (packet->dts != AV_NOPTS_VALUE) packet->dts -= shift;
                        av_packet_rescale_ts(packet, input_stream->time_base,
                                             output_stream->time_base);
                        packet->stream_index = output_stream->index;
                        av_interleaved_write_frame(state->fmt_ctx, packet);
                    }
                }
                av_packet_unref(packet);
            }
            if (packet) av_packet_free(&packet);
        }
        if (source_context) avformat_close_input(&source_context);
    }

    if (state->stream && state->codec_ctx) {
        int64_t video_duration = av_rescale_q(
            state->pts, state->codec_ctx->time_base, state->stream->time_base);
        state->stream->duration = video_duration;
        state->stream->avg_frame_rate = state->codec_ctx->framerate;
        state->stream->r_frame_rate = state->codec_ctx->framerate;
        if (state->fmt_ctx)
            state->fmt_ctx->duration = av_rescale_q(
                state->pts, state->codec_ctx->time_base, AV_TIME_BASE_Q);
    }

    if (state->fmt_ctx) av_write_trailer(state->fmt_ctx);
    if (state->sws_ctx) sws_freeContext(state->sws_ctx);
    if (state->pkt) av_packet_free(&state->pkt);
    if (state->frame) av_frame_free(&state->frame);
    if (state->codec_ctx) avcodec_free_context(&state->codec_ctx);
    if (state->fmt_ctx) {
        if (state->fmt_ctx->pb) avio_closep(&state->fmt_ctx->pb);
        avformat_free_context(state->fmt_ctx);
    }
    free(state->source_path);
    free(state->audio_indices);
    remove_audio_range(handle);
    free(state);
}
