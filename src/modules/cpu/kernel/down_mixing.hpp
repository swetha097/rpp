#include "rppdefs.h"
#include<iomanip>

RppStatus down_mixing_host_tensor(Rpp32f *srcPtr,
                                  RpptDescPtr srcDescPtr,
                                  Rpp32f *dstPtr,
                                  Rpp64s *samplesPerChannelTensor,
                                  Rpp32s *channelsTensor,
                                  Rpp32f *weightsTensor,
                                  bool  normalizeWeights)
{
    for(int batchCount = 0; batchCount < 1; batchCount++)
    {
        std::vector<float> normalizedWeights(8);
        float *weights = &weightsTensor[batchCount];
        int channels = channelsTensor[batchCount];
        int64_t samples = samplesPerChannelTensor[batchCount];
        if (normalizeWeights) 
        {
            normalizedWeights.resize(channels);

            //Compute sum of the weights
            double sum = 0.0;
            for (int i = 0; i < channels; i++)
                sum += weights[i];
            
            //Normalize the weights 
            for (int i = 0; i < channels; i++) 
                normalizedWeights[i] = weights[i] / sum;
            
            weights = normalizedWeights.data();
        }

        //use weights to downmix for stereo to mono
        for (int64_t o = 0, i = 0; o < samples; o++, i += channels) 
        {
            float sum = srcPtr[i] * weights[0];
            for (int c = 1; c < channels; c++) 
                sum += srcPtr[i + c] * weights[c];
            
            dstPtr[o] = sum;
        }
    }

    return RPP_SUCCESS;
}