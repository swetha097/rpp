// resampler.Resample(audio.data, 0, audio.shape[0], target_sample_rate, decode_scratch_mem.data(),
//                        decoded_audio_len, meta.sample_rate, meta.channels);

// void Resample(
//         Out *__restrict__ out, int64_t outBegin, int64_t outEnd, double outRate,
//         const float *__restrict__ in, int64_t n_in, double inRate)

#include "rppdefs.h"
#include<iomanip>

RppStatus down_mixing_host_tensor(Rpp32f *srcPtr,
                                  RpptDescPtr srcDescPtr,
                                  int64_t *nInTensor,
                                  Rpp64f *inRateTensor,
                                  Rpp32f *dstPtr,
                                  int64_t *outBeginTensor,
                                  int64_t *outEndTensor,
                                  Rpp64f *outRateTensor)
{
    for(int batchCount = 0; batchCount < 1; batchCount++)
    {
        int64_t nIn = nInTensor[batchCount];
        int64_t outBegin = outBeginTensor[batchCount];
        int64_t outEnd = outEndTensor[batchCount];
        double inRate = inRateTensor[batchCount];
        double outRate = outRateTensor[batchCount];

        int64_t in_pos = 0;
        int64_t block = 1 << 10;  // still leaves 13 significant bits for fractional part
        double scale = inRate / outRate;
        float fscale = scale;
        for (int64_t out_block = outBegin; out_block < outEnd; out_block += block) 
        {
            int64_t block_end = std::min(out_block + block, outEnd);
            double in_block_f = out_block * scale;
    //     int64_t in_block_i = std::floor(in_block_f);
    //     float in_pos = in_block_f - in_block_i;
    //     const float *__restrict__ in_block_ptr = srcPtr + in_block_i;
    //     for (int64_t out_pos = out_block; out_pos < block_end; out_pos++, in_pos += fscale) {
    //         int i0, i1;
    //         std::tie(i0, i1) = window.input_range(in_pos);
    //         if (i0 + in_block_i < 0)
    //         i0 = -in_block_i;
    //         if (i1 + in_block_i >= nIn)
    //         i1 = n_in - 1 - in_block_i;
    //         float f = 0;
    //         int i = i0;

    // #ifdef __SSE2__
    //         __m128 f4 = _mm_setzero_ps();
    //         __m128 x4 = _mm_setr_ps(i - in_pos, i+1 - in_pos, i+2 - in_pos, i+3 - in_pos);
    //         for (; i + 3 <= i1; i += 4) {
    //         __m128 w4 = window(x4);

    //         f4 = _mm_add_ps(f4, _mm_mul_ps(_mm_loadu_ps(in_block_ptr + i), w4));
    //         x4 = _mm_add_ps(x4, _mm_set1_ps(4));
    //         }

    //         f4 = _mm_add_ps(f4, _mm_shuffle_ps(f4, f4, _MM_SHUFFLE(1, 0, 3, 2)));
    //         f4 = _mm_add_ps(f4, _mm_shuffle_ps(f4, f4, _MM_SHUFFLE(0, 1, 0, 1)));
    //         f = _mm_cvtss_f32(f4);
    // #endif

    //         float x = i - in_pos;
    //         for (; i <= i1; i++, x++) {
    //         float w = window(x);
    //         f += in_block_ptr[i] * w;
    //         }
    //         dstPtr[out_pos] = (f);
    //     }
    //     }
        }
    }
    return RPP_SUCCESS;
}