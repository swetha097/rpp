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
                            Rpp32f *anchorTensor,
                            Rpp32f *shapeTensor,
                            Rpp32s *axes,
                            Rpp32f *fillValues,
                            Rpp32s numOfDims,
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

        Rpp32f fillValue = *fillValues;
        Rpp32s sampleBatchCount = batchCount * numOfDims;

        if(srcDescPtr->c == 1)
        {
            // If normalized between 0 - 1 convert to actual indices
            Rpp32s srcBufferLength = srcLengthTensor[batchCount];
            Rpp32f anchorRaw = anchorTensor[batchCount];
            Rpp32f shapeRaw = shapeTensor[batchCount];

            if(normalizedAnchor) {
                anchorRaw *= srcBufferLength;
            }
            if(normalizedShape) { // Doubt
                shapeRaw *= srcBufferLength;
            }

            Rpp32s anchor = std::llround(anchorRaw);
            Rpp32s shape = std::llround(shapeRaw);

            Rpp32s sliceEnd = anchor + shape;
            applyPolicy(policyType, &anchor, &sliceEnd, &srcBufferLength);  // check the policy and update the values accordingly
            shape = sliceEnd - anchor;

            if(anchor == 0 && shape == srcBufferLength) {
            // Do a memcpy if output dimension input dimension
                memcpy(dstPtrTemp, srcPtrTemp, shape * sizeof(Rpp32f));
            } else {
                Rpp32s vectorIncrement = 8;
                Rpp32s alignedLength = (shape / 8) * 8;
                __m256 pFillValue = _mm256_set1_ps(fillValue);

                bool needPad = (anchor < 0) || ((anchor + shape) > srcBufferLength);
                Rpp32s dstIdx = 0;
                if (needPad)
                {
                    // out of bounds (left side)
                    Rpp32s numIndices = std::abs(std::min(anchor, 0));
                    Rpp32s leftPadLength = std::min(numIndices, shape);
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
                Rpp32s dstLengthInBounds = std::max(shape - dstIdx, 0);
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
                    for (; dstIdx < shape; dstIdx++)
                    {
                        *dstPtrTemp = fillValue;
                        dstPtrTemp++;
                    }
                }
            }
        } else if(srcDescPtr->c > 1) {
            Rpp32s srcBufferLength = srcLengthTensor[sampleBatchCount + 1];
            Rpp32f anchorRaw[numOfDims], shapeRaw[numOfDims];
            Rpp32s anchor[numOfDims], shape[numOfDims];

            anchorRaw[0] = anchorTensor[sampleBatchCount];
            anchorRaw[1] = anchorTensor[sampleBatchCount + 1];
            shapeRaw[0] = shape[0];
            shapeRaw[1] = shape[1];
            if(normalizedAnchor) {
                anchorRaw[0] *= srcBufferLength;
                anchorRaw[1] *= srcBufferLength;
            }
            if(normalizedShape) {
                shapeRaw[0] *= srcLengthTensor[sampleBatchCount];
                shapeRaw[1] *= srcLengthTensor[sampleBatchCount];
            }

            anchor[0] = std::llround(anchorRaw[0]);
            shape[0] = std::llround(shapeRaw[0]);
            anchor[1] = std::llround(anchorRaw[1]);
            shape[1] = std::llround(shapeRaw[1]);

            // Rpp32s sliceEnd = anchor + shape;
            // bool channelsProcessed = false;

            Rpp32s channelBound = std::min(srcLengthTensor[sampleBatchCount + 1], shape[1]);
            Rpp32s frameLengthBound = std::min(srcLengthTensor[sampleBatchCount], shape[0]);

            bool needPad = (anchor[0] < 0) || ((anchor[0] + shape[0]) > srcLengthTensor[sampleBatchCount]);
            bool needPad1 = (anchor[1] < 0) || ((anchor[1] + shape[1]) > srcLengthTensor[sampleBatchCount + 1]);
            srcPtrTemp = srcPtrTemp + anchor[0] * srcDescPtr->strides.wStride;
            int row = 0;
            for(; row < frameLengthBound; row++)
            {
                Rpp32f * srcPtrTempRow = srcPtrTemp + row * srcDescPtr->strides.wStride + anchor[1];
                int col = 0;
                Rpp32f *dstPtrTempRow = dstPtrTemp;
                for(; col < channelBound; col++) {
                    *dstPtrTempRow++ = *(srcPtrTempRow + col);
                }
                Rpp32s rem = anchor[1] + shape[1];
                if(col < shape[1] && needPad1)
                {
                    for(; col < shape[1]; col++)
                        *dstPtrTempRow++ = fillValue;
                }
                dstPtrTemp += dstDescPtr->strides.wStride;
            }
            if(row < shape[0]  && needPad)
            {
                for(; row < shape[0]; row++)
                {
                    Rpp32f * dstPtrRowTemp = dstPtrTemp;
                    memset(dstPtrRowTemp, fillValue,  dstDescPtr->strides.wStride);
                    dstPtrTemp += dstDescPtr->strides.wStride;
                }

            }
        }
    }

	return RPP_SUCCESS;
}