#include "rppdefs.h"

RppStatus down_mixing_host_tensor(Rpp32f *srcPtr,
                                  RpptDescPtr srcDescPtr,
                                  Rpp32f *dstPtr,
                                  Rpp64s *samplesPerChannelTensor,
                                  Rpp32s *channelsTensor,
                                  bool  normalizeWeights)
{
    for(int batchCount = 0; batchCount < srcDescPtr->n; batchCount++)
    {
        Rpp32f *srcPtrTemp = srcPtr + batchCount * srcDescPtr->strides.nStride;
        Rpp32f *dstPtrTemp = dstPtr + batchCount * srcDescPtr->strides.nStride;

        Rpp32s channels = channelsTensor[batchCount];
        Rpp64s samples = samplesPerChannelTensor[batchCount];
        std::vector<float> weights;
        weights.resize(channels, 1.f / channels);
        std::vector<float> normalizedWeights;

        if (normalizeWeights)
        {
            normalizedWeights.resize(channels);

            // Compute sum of the weights
            double sum = 0.0;
            for (int i = 0; i < channels; i++)
                sum += weights[i];

            // Normalize the weights
            float invSum = 1.0 / sum;
            for (int i = 0; i < channels; i++)
                normalizedWeights[i] = weights[i] * invSum;

            weights = normalizedWeights;
        }

        // use weights to downmix for stereo to mono
        for (int64_t o = 0, i = 0; o < samples; o++, i += channels)
        {
            float sum = srcPtrTemp[i] * weights[0];
            for (int c = 1; c < channels; c++)
                sum += srcPtrTemp[i + c] * weights[c];

            dstPtrTemp[o] = sum;
        }
    }

    return RPP_SUCCESS;
}