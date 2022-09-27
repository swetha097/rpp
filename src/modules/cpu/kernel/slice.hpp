#include "rppdefs.h"
#include "rpp_cpu_simd.hpp"
#include "rpp_cpu_common.hpp"

void applyPolicy(RpptOutOfBoundsPolicy policyType, Rpp32s *anchor, Rpp32s *sliceEnd, Rpp32s *srcBufferLength)
{
    switch (policyType)
    {
        case RpptOutOfBoundsPolicy::PAD:
            break;
        case RpptOutOfBoundsPolicy::TRIMTOSHAPE:
            *anchor = std::min(std::max(*anchor, 0), *srcBufferLength);
            *sliceEnd = std::min(std::max(*anchor + *sliceEnd, 0), *srcBufferLength);
            break;
        case RpptOutOfBoundsPolicy::ERROR:
        default:
            bool isOutOfBounds = (*anchor < 0) || (*anchor > *srcBufferLength);
            break;
    }
}

RppStatus slice_host_tensor(Rpp32f *srcPtr,
                            RpptDescPtr srcDescPtr,
                            Rpp32f *dstPtr,
							RpptDescPtr dstDescPtr,
                            Rpp32s *srcLengthTensor,
                            Rpp32s *anchorTensor,
                            Rpp32s *dstBufferLengthTensor,
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
        Rpp32s srcBufferLength = srcLengthTensor[batchCount];
        Rpp32s anchor = anchorTensor[batchCount];
        Rpp32s sliceEnd = anchor + dstBufferLengthTensor[batchCount];
        Rpp32s dstBufferLength = sliceEnd - anchor;

        // If normalized between 0 - 1 convert to actual indices
        if(normalizedAnchor)
            anchor *= dstBufferLength;
        if(normalizedShape)
            dstBufferLength *= dstBufferLength;

        applyPolicy(policyType, &anchor, &sliceEnd, &srcBufferLength);  // check the policy and update the values accordingly
        dstBufferLength = sliceEnd - anchor;

        if(srcDescPtr->c == 1 && anchor == 0 && dstBufferLength == srcBufferLength)
        {
            // Do a memcpy if output dimension input dimension
            memcpy(dstPtrTemp, srcPtrTemp, dstBufferLength * sizeof(Rpp32f));
        }
        else
        {
            Rpp32s vectorIncrement = 8;
            Rpp32s alignedLength = (dstBufferLength / 8) * 8;
            __m256 pFillValue = _mm256_set1_ps(fillValue);

            bool needPad = (anchor < 0) || ((anchor + dstBufferLength) > srcBufferLength);
            Rpp32s dstIdx = 0;
            if (needPad)
            {
                // out of bounds (left side)
                Rpp32s numIndices = std::abs(std::min(anchor, 0));
                Rpp32s leftPadLength = std::min(numIndices, dstBufferLength);
                Rpp32s alignedLeftPadLength = (leftPadLength / 8) * 8;

                for (; dstIdx < alignedLeftPadLength; dstIdx += vectorIncrement)
                {
                    _mm256_storeu_ps(dstPtrTemp, pFillValue);
                    dstPtrTemp += vectorIncrement;
                }
                for (; dstIdx < leftPadLength; dstIdx++)
                {
                    *dstPtrTemp = fillValue;
                    dstPtrTemp++;
                }

                anchor += leftPadLength;
            }

            // within input bounds
            Rpp32s srcLengthInBounds = std::max(srcBufferLength - anchor, 0);
            Rpp32s dstLengthInBounds = std::max(dstBufferLength - dstIdx, 0);
            Rpp32s lengthInBounds = std::min(srcLengthInBounds, dstLengthInBounds);
            memcpy(dstPtrTemp, &srcPtrTemp[anchor], (size_t)(lengthInBounds * sizeof(Rpp32f)));
            dstIdx += lengthInBounds;
            dstPtrTemp += lengthInBounds;

            if (needPad)
            {
                // out of bounds (right side)
                for (; dstIdx < alignedLength; dstIdx += vectorIncrement)
                {
                    _mm256_storeu_ps(dstPtrTemp, pFillValue);
                    dstPtrTemp += vectorIncrement;
                }
                for (; dstIdx < dstBufferLength; dstIdx++)
                {
                    *dstPtrTemp = fillValue;
                    dstPtrTemp++;
                }
            }
        }
    }

	return RPP_SUCCESS;
}
