#include "rppdefs.h"
#include "rpp_cpu_simd.hpp"
#include "rpp_cpu_common.hpp"

inline double Hann(double x) {
  return 0.5 * (1 + std::cos(x * M_PI));
}

struct ResamplingWindow {
  inline std::pair<int, int> input_range(float x) const {
    int i0 = std::ceil(x) - lobes;
    int i1 = std::floor(x) + lobes;
    return {i0, i1};
  }

  inline float operator()(float x) const {
    float fi = x * scale + center;
    int i = std::floor(fi);
    float di = fi - i;
    return LUT[i] + di * (LUT[i + 1] - LUT[i]);
  }

#ifdef __SSE2__
  inline __m128 operator()(__m128 x) const {
    __m128 fi = _mm_add_ps(x * _mm_set1_ps(scale), _mm_set1_ps(center));
    __m128i i = _mm_cvtps_epi32(fi);
    __m128 fifloor = _mm_cvtepi32_ps(i);
    __m128 di = _mm_sub_ps(fi, fifloor);
    int idx[4];
    _mm_storeu_si128(reinterpret_cast<__m128i*>(idx), i);
    __m128 curr = _mm_setr_ps(LUT[idx[0]],   LUT[idx[1]],
                              LUT[idx[2]],   LUT[idx[3]]);
    __m128 next = _mm_setr_ps(LUT[idx[0]+1], LUT[idx[1]+1],
                              LUT[idx[2]+1], LUT[idx[3]+1]);
    return _mm_add_ps(curr, _mm_mul_ps(di, _mm_sub_ps(next, curr)));
  }
#endif


  float scale = 1, center = 1;
  int lobes = 0, coeffs = 0;
  std::vector<float> LUT;
};

inline void windowed_sinc(ResamplingWindow &window,
    int coeffs, int lobes, std::function<double(double)> envelope = Hann) {
  float scale = 2.0f * lobes / (coeffs - 1);
  float scale_envelope = 2.0f / coeffs;
  window.coeffs = coeffs;
  window.lobes = lobes;
  window.LUT.resize(coeffs + 2);  // add zeros
  int center = (coeffs - 1) * 0.5f;
  for (int i = 0; i < coeffs; i++) {
    float x = (i - center) * scale;
    float y = (i - center) * scale_envelope;
    float w = sinc(x) * envelope(y);
    window.LUT[i + 1] = w;
  }
  window.center = center + 1;  // allow for leading zero
  window.scale = 1 / scale;
}


inline int64_t resampled_length(int64_t in_length, double inRate, double outRate) {
  return std::ceil(in_length * outRate / inRate);
}

RppStatus audio_resample_host_tensor(Rpp32f *srcPtr,
                                     RpptDescPtr srcDescPtr,
                                     Rpp32f *dstPtr,
								     RpptDescPtr dstDescPtr,
                                     Rpp32f *inRateTensor,
                                     Rpp32f *outRateTensor,
                                     Rpp32s *srcLengthTensor,
                                     Rpp32s *channelTensor,
                                     Rpp32f quality,
                                     Rpp32f scale)
{
	omp_set_dynamic(0);
#pragma omp parallel for num_threads(srcDescPtr->n)
	for(int batchCount = 0; batchCount < srcDescPtr->n; batchCount++)
	{
		Rpp32f *srcPtrTemp = srcPtr + batchCount * srcDescPtr->strides.nStride;
		Rpp32f *dstPtrTemp = dstPtr + batchCount * dstDescPtr->strides.nStride;

        ResamplingWindow window;
        int lobes = 16, LUT_size = 2048;
        windowed_sinc(window, LUT_size, lobes);

        Rpp32f inRate = inRateTensor[batchCount];
        Rpp32f outRate = outRateTensor[batchCount];
        Rpp32s srcLength = srcLengthTensor[batchCount];

        int64_t in_pos = 0;
        int64_t block = 1 << 10;  // still leaves 13 significant bits for fractional part
        double scale = inRate / outRate;
        float fscale = scale;
        int64_t out_begin = 0;
        int64_t out_end = (outRate / inRate) * srcLength;
        int64_t n_in = srcLength;

        for (int64_t out_block = out_begin; out_block < out_end; out_block += block)
        {
            int64_t block_end = std::min(out_block + block, out_end);
            double in_block_f = out_block * scale;
            int64_t in_block_i = std::floor(in_block_f);
            float in_pos = in_block_f - in_block_i;
            const float *__restrict__ in_block_ptr = srcPtrTemp + in_block_i;
            for (int64_t out_pos = out_block; out_pos < block_end; out_pos++, in_pos += fscale) {
                int i0, i1;
                std::tie(i0, i1) = window.input_range(in_pos);
                if (i0 + in_block_i < 0)
                i0 = -in_block_i;
                if (i1 + in_block_i >= n_in)
                i1 = n_in - 1 - in_block_i;
                float f = 0;
                int i = i0;

        #ifdef __SSE2__
                __m128 f4 = _mm_setzero_ps();
                __m128 x4 = _mm_setr_ps(i - in_pos, i+1 - in_pos, i+2 - in_pos, i+3 - in_pos);
                for (; i + 3 <= i1; i += 4) {
                __m128 w4 = window(x4);

                f4 = _mm_add_ps(f4, _mm_mul_ps(_mm_loadu_ps(in_block_ptr + i), w4));
                x4 = _mm_add_ps(x4, _mm_set1_ps(4));
                }

                f4 = _mm_add_ps(f4, _mm_shuffle_ps(f4, f4, _MM_SHUFFLE(1, 0, 3, 2)));
                f4 = _mm_add_ps(f4, _mm_shuffle_ps(f4, f4, _MM_SHUFFLE(0, 1, 0, 1)));
                f = _mm_cvtss_f32(f4);
        #endif

                float x = i - in_pos;
                for (; i <= i1; i++, x++)
                {
                    float w = window(x);
                    f += in_block_ptr[i] * w;
                }

                dstPtrTemp[out_pos] = f;
            }
        }
    }
	return RPP_SUCCESS;
}
