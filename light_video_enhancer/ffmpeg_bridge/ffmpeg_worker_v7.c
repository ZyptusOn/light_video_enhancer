/* Timestamp metadata fixes for constant-frame-rate output. */

#include "ffmpeg_worker_v6.c"

__declspec(dllexport) void nve_encoder_prepare(void *handle)
{
    EncoderState *state = (EncoderState*)handle;
    if (state && state->frame) state->frame->duration = 1;
}

__declspec(dllexport) void nve_encoder_finish2(void *handle)
{
    EncoderState *state = (EncoderState*)handle;
    if (state && state->stream && state->codec_ctx) {
        state->stream->duration = av_rescale_q(state->pts,
            state->codec_ctx->time_base, state->stream->time_base);
    }
    nve_encoder_finish(handle);
}
