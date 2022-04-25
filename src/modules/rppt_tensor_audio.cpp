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

#include "rppdefs.h"
#include "rppi_validate.hpp"
#include "rppt_tensor_audio.h"
#include "cpu/host_tensor_audio.hpp"


RppStatus rppt_non_silent_region_detection_host(RppPtr_t srcPtr,
                                                Rpp32u srcSize,
                                                Rpp32u detectedIndex,
                                                Rpp32u detectionLength,
                                                Rpp32f cutOffDB,
                                                Rpp32u windowLength,
                                                Rpp32f referencePower,
                                                Rpp32u resetInterval,
                                                bool referenceMax,
                                                rppHandle_t rppHandle)
{
    non_silent_region_detection_host_tensor((Rpp32f*) srcPtr,
                                            srcSize,
                                            detectedIndex,
                                            detectionLength,
                                            cutOffDB,
                                            windowLength,
                                            referencePower,
                                            resetInterval,
                                            referenceMax);

    return RPP_SUCCESS;
}