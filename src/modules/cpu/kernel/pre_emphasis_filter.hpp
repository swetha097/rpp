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
      Rpp32s srcSize = srcSizeTensor[batchCount];
      Rpp32f coeff = coeffTensor[batchCount];

      if(borderType == 0)
        dstPtr[0] = srcPtr[0];
      else if(borderType == 1)
        dstPtr[0] = srcPtr[0] * (1 - coeff); 
      else
        dstPtr[0] = srcPtr[0] - coeff * srcPtr[1]; 

      for(int i = 1; i < srcSize; i++)
        dstPtr[i] = srcPtr[i] - coeff * srcPtr[i - 1];
    }
    return RPP_SUCCESS;
}