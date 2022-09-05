#include "rppdefs.h"
#include "rpp_cpu_simd.hpp"
#include "rpp_cpu_common.hpp"

void applyPolicy(RpptOutOfBoundsPolicy policyType, Rpp32s *slice_start, Rpp32s *slice_end, Rpp32s *in_shape)
{
    switch (policyType)
    {
        case RpptOutOfBoundsPolicy::PAD:
            break;
        case RpptOutOfBoundsPolicy::TRIMTOSHAPE:
            *slice_start = std::min(std::max(*slice_start, 0), *in_shape);
            *slice_end = std::min(std::max(*slice_start + *slice_end, 0), *in_shape);
            break;
        case RpptOutOfBoundsPolicy::ERROR:
        default:
            bool isOutOfBounds = (*slice_start < 0) || (*slice_start > *in_shape);
            break;
    }
}

RppStatus slice_host_tensor(Rpp32f *srcPtr,
                            RpptDescPtr srcDescPtr,
                            Rpp32f *dstPtr,
							RpptDescPtr dstDescPtr,
                            Rpp32s *srcLengthTensor,
                            Rpp32s *anchorTensor,
                            Rpp32s *outShapeTensor,
                            Rpp32s *axes,
                            Rpp32f *fillValues,
                            bool normalizedAnchor,
                            bool normalizedShape,
                            RpptOutOfBoundsPolicy policyType)
{
	omp_set_dynamic(0);
#pragma omp parallel for num_threads(srcDescPtr->n)
	for(int batchCount = 0; batchCount < srcDescPtr->n; batchCount++)
	{
		Rpp32f *srcPtrTemp = srcPtr + batchCount * srcDescPtr->strides.nStride;
		Rpp32f *dstPtrTemp = dstPtr + batchCount * dstDescPtr->strides.nStride;

        Rpp32f fillValue = fillValues[batchCount];
        Rpp32s in_shape = srcLengthTensor[batchCount];
        Rpp32s slice_start = anchorTensor[batchCount];
        Rpp32s slice_end = slice_start + outShapeTensor[batchCount];
        Rpp32s out_shape = slice_end - slice_start;

        // If normalized between 0 - 1 convert to actual indices
        if(normalizedAnchor)
            slice_start *= out_shape;
        if(normalizedShape)
            out_shape *= out_shape;

        applyPolicy(policyType, &slice_start, &slice_end, &in_shape);  // check the policy and update the values accordingly
        out_shape = slice_end - slice_start;

        Rpp32s out_idx = 0;
        bool needPad = (slice_start < 0) || ((slice_start + out_shape) > in_shape);
        if (needPad)
        {
            // out of bounds (left side)
            for (; slice_start < 0 && out_idx < out_shape; slice_start++, out_idx++)
                dstPtrTemp[out_idx] = fillValue;
        }

        // within input bounds
        for (; slice_start < in_shape && out_idx < out_shape; slice_start++, out_idx++)
            dstPtrTemp[out_idx] = srcPtrTemp[slice_start];

        if (needPad)
        {
            // out of bounds (right side)
            for (; out_idx < out_shape; slice_start++, out_idx++)
                dstPtrTemp[out_idx] = fillValue;
        }
    }

	return RPP_SUCCESS;
}
