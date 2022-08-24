#include "rppdefs.h"
#include "rpp_cpu_simd.hpp"
#include "rpp_cpu_common.hpp"

RppStatus slice_host_tensor(Rpp32f *srcPtr,
                            RpptDescPtr srcDescPtr,
                            Rpp32f *dstPtr,
							RpptDescPtr dstDescPtr,
                            Rpp32s *srcLengthTensor,
                            Rpp32s *anchorTensor,
                            Rpp32s *shapeTensor,
                            Rpp32s *axes,
                            Rpp32f *fillValues,
                            bool normalizedAnchor,
                            bool normalizedShape)
{
	omp_set_dynamic(0);
#pragma omp parallel for num_threads(srcDescPtr->n)
	for(int batchCount = 0; batchCount < srcDescPtr->n; batchCount++)
	{
		Rpp32f *srcPtrTemp = srcPtr + batchCount * srcDescPtr->strides.nStride;
		Rpp32f *dstPtrTemp = dstPtr + batchCount * dstDescPtr->strides.nStride;

    }
	return RPP_SUCCESS;
}
