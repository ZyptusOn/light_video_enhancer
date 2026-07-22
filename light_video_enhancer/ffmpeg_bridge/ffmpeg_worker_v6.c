/* Robust finalisation API. The legacy close path did not retry a flush when
 * avcodec_send_frame(NULL) returned EAGAIN, leaving the final frame delayed. */

#include "ffmpeg_worker_v5.c"

__declspec(dllexport) void nve_encoder_finish(void *handle)
{
    EncoderState *state = (EncoderState*)handle;
    if (!state) return;
    AudioRangeEntry *range = find_audio_range(handle);
    double start = range ? range->start : 0.0;
    double duration = range ? range->duration : -1.0;
    double end = duration >= 0.0 ? start + duration : -1.0;

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
                for (int index = 0; index < state->audio_count; index++) {
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
                        int64_t shift = av_rescale_q((int64_t)(start * AV_TIME_BASE),
                                                     AV_TIME_BASE_Q, input_stream->time_base);
                        if (packet->pts != AV_NOPTS_VALUE) packet->pts -= shift;
                        if (packet->dts != AV_NOPTS_VALUE) packet->dts -= shift;
                        AVStream *output_stream = state->fmt_ctx->streams[
                            (int)state->fmt_ctx->nb_streams - state->audio_count + audio_index];
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
