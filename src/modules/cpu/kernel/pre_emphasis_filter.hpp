#include "rppdefs.h"
#include "cpu/rpp_cpu_simd.hpp"
#include "cpu/rpp_cpu_common.hpp"
#include<iomanip>

RppStatus pre_emphasis_filter_host_tensor(Rpp32f *srcPtr,
                                          RpptDescPtr srcDescPtr,
                                          Rpp32f *dstPtr,
                                          Rpp32s *srcSizeTensor,
                                          Rpp32f *coeffTensor,
                                          Rpp32u borderType)
{
#pragma omp parallel for num_threads(srcDescPtr->n)    
    for(int batchCount = 0; batchCount < srcDescPtr->n; batchCount++)
    {
      Rpp32f *srcPtrTemp = srcPtr + batchCount * srcDescPtr->strides.nStride;
      Rpp32f *dstPtrTemp = dstPtr + batchCount * srcDescPtr->strides.nStride;
      Rpp32s srcSize = srcSizeTensor[batchCount];
      Rpp32f coeff = coeffTensor[batchCount];

      if(borderType == RpptAudioBorderType::Zero)
        dstPtrTemp[0] = srcPtrTemp[0];
      else if(borderType == RpptAudioBorderType::Clamp)
        dstPtrTemp[0] = srcPtrTemp[0] * (1 - coeff); 
      else if(borderType == RpptAudioBorderType::Reflect)
        dstPtrTemp[0] = srcPtrTemp[0] - coeff * srcPtrTemp[1]; 

      for(int i = 1; i < srcSize; i++)
        dstPtrTemp[i] = srcPtrTemp[i] - coeff * srcPtrTemp[i - 1];
    }
    return RPP_SUCCESS;
}