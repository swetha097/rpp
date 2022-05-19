/*
Copyright (c) 2019 - 2022 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#ifndef RPPT_TENSOR_AUDIO_H
#define RPPT_TENSOR_AUDIO_H
#include "rpp.h"
#include "rppdefs.h"
#ifdef __cplusplus
extern "C" {
#endif

//Non Silent Region Detection
RppStatus rppt_non_silent_region_detection_host(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, Rpp32s *srcSize, Rpp32s *detectedIndex, Rpp32s *detectionLength, Rpp32f *cutOffDB, Rpp32s *windowLength, Rpp32f *referencePower, Rpp32s *resetInterval, bool *referenceMax, rppHandle_t handle);

//To Decibels
RppStatus rppt_to_decibels_host(RppPtr_t magnitudePtr, RppPtr_t DBPtr, Rpp32u batchSize, Rpp32f cutOffDB = -200.0, Rpp32f multiplier = 10.0, Rpp32f referenceMagnitude = 0.0);

//Pre emphasis filter
RppStatus rppt_pre_emphasis_filter_host(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t dstPtr, Rpp32s *srcSizeTensor, Rpp32f *coeffTensor, Rpp32u borderType = 1);

//Downmixing
RppStatus rppt_down_mixing_host(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t dstPtr, Rpp64s *samplesPerChannelTensor, Rpp32s *channelsTensor, bool normalizeWeights = false);

//Resampling
RppStatus rppt_resampling_host(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, int64_t *nInTensor, Rpp64f *inRateTensor, RppPtr_t dstPtr, int64_t *outBeginTensor, int64_t *outEndTensor, Rpp64f *outRateTensor);

#ifdef __cplusplus
}
#endif
#endif // RPPT_TENSOR_AUDIO_H