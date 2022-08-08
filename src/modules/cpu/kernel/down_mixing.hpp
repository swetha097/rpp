#include "rppdefs.h"
#include <omp.h>

float sum4(__m128 x)
{
    const __m128 hsum_0 = _mm_hadd_ps(x, x);
    const __m128 hsum_1 = _mm_hadd_ps(hsum_0, hsum_0);
    return _mm_cvtss_f32(hsum_1);
}

RppStatus down_mixing_host_tensor(Rpp32f *srcPtr,
                                  RpptDescPtr srcDescPtr,
                                  Rpp32f *dstPtr,
                                  RpptDescPtr dstDescPtr,
                                  Rpp32s *srcLengthTensor,
                                  Rpp32s *channelsTensor,
                                  bool normalizeWeights)
{
    omp_set_dynamic(0);
#pragma omp parallel for num_threads(srcDescPtr->n)
    for(int batchCount = 0; batchCount < srcDescPtr->n; batchCount++)
    {
        Rpp32f *srcPtrTemp = srcPtr + batchCount * srcDescPtr->strides.nStride;
        Rpp32f *dstPtrTemp = dstPtr + batchCount * dstDescPtr->strides.nStride;

        Rpp32s channels = channelsTensor[batchCount];
        Rpp32s samples = srcLengthTensor[batchCount];
        std::vector<float> weights;
        weights.resize(channels, 1.f / channels);
        std::vector<float> normalizedWeights;

        if(normalizeWeights)
        {
            normalizedWeights.resize(channels);

            // Compute sum of the weights
            Rpp32f sum = 0.0;
            for(int i = 0; i < channels; i++)
                sum += weights[i];

            // Normalize the weights
            Rpp32f invSum = 1.0 / sum;
            for(int i = 0; i < channels; i++)
                normalizedWeights[i] = weights[i] * invSum;

            weights = normalizedWeights;
        }

        int channelIncrement = 4;
		int alignedChannels = (channels / 4) * 4;

        // use weights to downmix to mono
        for(int64_t dstIdx = 0; dstIdx < samples; dstIdx++)
        {
            __m128 pDst = _mm_setzero_ps();
            int channelLoopCount = 0;
            for(; channelLoopCount < alignedChannels; channelLoopCount += channelIncrement)
            {
                __m128 pSrc, pWeights;
                pWeights = _mm_setr_ps(weights[channelLoopCount], weights[channelLoopCount + 1], weights[channelLoopCount + 2], weights[channelLoopCount + 3]);
                pSrc = _mm_loadu_ps(srcPtrTemp);
                pSrc = _mm_mul_ps(pSrc, pWeights);
                pDst = _mm_add_ps(pDst, pSrc);
                srcPtrTemp += channelIncrement;
            }
            dstPtrTemp[dstIdx] = sum4(pDst);
            for(; channelLoopCount < channels; channelLoopCount++)
            {
                dstPtrTemp[dstIdx] += ((*srcPtrTemp) * weights[channelLoopCount]);
                srcPtrTemp++;
            }
        }
    }

    return RPP_SUCCESS;
}
