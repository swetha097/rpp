#ifndef HOST_ADVANCED_AUGMENTATIONS_H
#define HOST_ADVANCED_AUGMENTATIONS_H

#include "cpu/rpp_cpu_simd.hpp"
#include <cpu/rpp_cpu_common.hpp>

/**************** water ***************/

template <typename T>
RppStatus water_host_batch(T* srcPtr, RppiSize *batch_srcSize, RppiSize *batch_srcSizeMax, T* dstPtr,
                         Rpp32f *batch_ampl_x, Rpp32f *batch_ampl_y,
                         Rpp32f *batch_freq_x, Rpp32f *batch_freq_y,
                         Rpp32f *batch_phase_x, Rpp32f *batch_phase_y,
                         Rpp32u outputFormatToggle, Rpp32u nbatchSize,
                         RppiChnFormat chnFormat, Rpp32u channel)
{
    if(chnFormat == RPPI_CHN_PLANAR)
    {
        omp_set_dynamic(0);
#pragma omp parallel for num_threads(nbatchSize)
        for(int batchCount = 0; batchCount < nbatchSize; batchCount ++)
        {
            Rpp32u imageDimMax = batch_srcSizeMax[batchCount].height * batch_srcSizeMax[batchCount].width;

            Rpp32f ampl_x = batch_ampl_x[batchCount];
            Rpp32f ampl_y = batch_ampl_y[batchCount];
            Rpp32f freq_x = batch_freq_x[batchCount];
            Rpp32f freq_y = batch_freq_y[batchCount];
            Rpp32f phase_x = batch_phase_x[batchCount];
            Rpp32f phase_y = batch_phase_y[batchCount];

            T *srcPtrImage, *dstPtrImage;
            Rpp32u loc = 0;
            compute_image_location_host(batch_srcSizeMax, batchCount, &loc, channel);
            srcPtrImage = srcPtr + loc;
            dstPtrImage = dstPtr + loc;

            T *srcPtrImageR, *srcPtrImageG, *srcPtrImageB;
            srcPtrImageR = srcPtrImage;
            srcPtrImageG = srcPtrImageR + imageDimMax;
            srcPtrImageB = srcPtrImageG + imageDimMax;

            Rpp32u elementsInRowMax = batch_srcSizeMax[batchCount].width;

            for(int i = 0; i < batch_srcSize[batchCount].height; i++)
            {
                T *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                dstPtrTempR = dstPtrImage + (i * elementsInRowMax);
                dstPtrTempG = dstPtrTempR + imageDimMax;
                dstPtrTempB = dstPtrTempG + imageDimMax;

                Rpp32u bufferLength = batch_srcSize[batchCount].width;
                Rpp32u alignedLength = bufferLength & ~3;

                __m128 pI = _mm_set1_ps((Rpp32f)i);
                __m128 pJ, pWaterI, pWaterJ;
                __m128 pAmplX = _mm_set1_ps(ampl_x);
                __m128 pAmplY = _mm_set1_ps(ampl_y);
                __m128 pFreqX = _mm_set1_ps(freq_x);
                __m128 pFreqY = _mm_set1_ps(freq_y);
                __m128 pPhaseX = _mm_set1_ps(phase_x);
                __m128 pPhaseY = _mm_set1_ps(phase_y);
                __m128 p0, p1, p2, p3;

                Rpp32f waterI[4], waterJ[4];

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount+=4)
                {
                    pJ = _mm_setr_ps((Rpp32f)(vectorLoopCount), (Rpp32f)(vectorLoopCount + 1), (Rpp32f)(vectorLoopCount + 2), (Rpp32f)(vectorLoopCount + 3));
                    p0 = _mm_mul_ps(pFreqX, pI);
                    p0 = _mm_add_ps(p0, pPhaseX);
                    sincos_ps(p0, &p1, &p2);
                    p1 = _mm_mul_ps(p1, pAmplX);
                    pWaterJ = _mm_add_ps(pJ, p1);
                    p0 = _mm_mul_ps(pFreqY, pJ);
                    p0 = _mm_add_ps(p0, pPhaseY);
                    sincos_ps(p0, &p1, &p2);
                    p2 = _mm_mul_ps(p2, pAmplY);
                    pWaterI = _mm_add_ps(pI, p2);

                    _mm_storeu_ps(waterI, pWaterI);
                    _mm_storeu_ps(waterJ, pWaterJ);

                    for (int count = 0; count < 4; count++)
                    {
                        Rpp32u waterIint, waterJint;
                        waterIint = (Rpp32u) RPPPRANGECHECK(waterI[count], 0, batch_srcSize[batchCount].height - 1);
                        waterJint = (Rpp32u) RPPPRANGECHECK(waterJ[count], 0, batch_srcSize[batchCount].width - 1);

                        *dstPtrTempR = *(srcPtrImageR + (waterIint * elementsInRowMax) + waterJint);
                        dstPtrTempR++;

                        if (channel == 3)
                        {
                            *dstPtrTempG = *(srcPtrImageG + (waterIint * elementsInRowMax) + waterJint);
                            *dstPtrTempB = *(srcPtrImageB + (waterIint * elementsInRowMax) + waterJint);
                            dstPtrTempG++;
                            dstPtrTempB++;
                        }
                    }
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    Rpp32f locI = (Rpp32f) i;
                    Rpp32f locJ = (Rpp32f) vectorLoopCount;

                    Rpp32f waterLocJ = locJ + ampl_x * sin((freq_x * locI) + phase_x);
                    Rpp32f waterLocI = locI + ampl_y * cos((freq_y * locJ) + phase_y);

                    Rpp32u waterLocIint, waterLocJint;
                    waterLocIint = (Rpp32u) RPPPRANGECHECK(waterLocI, 0, batch_srcSize[batchCount].height);
                    waterLocJint = (Rpp32u) RPPPRANGECHECK(waterLocJ, 0, batch_srcSize[batchCount].width);

                    *dstPtrTempR = *(srcPtrImageR + (waterLocIint * elementsInRowMax) + waterLocJint);
                    dstPtrTempR++;

                    if (channel == 3)
                    {
                        *dstPtrTempG = *(srcPtrImageG + (waterLocIint * elementsInRowMax) + waterLocJint);
                        *dstPtrTempB = *(srcPtrImageB + (waterLocIint * elementsInRowMax) + waterLocJint);
                        dstPtrTempG++;
                        dstPtrTempB++;
                    }
                }
            }
            if (outputFormatToggle == 1)
            {
                T *dstPtrImageUnpadded = (T*) calloc(channel * batch_srcSize[batchCount].height * batch_srcSize[batchCount].width, sizeof(T));
                T *dstPtrImageUnpaddedCopy = (T*) calloc(channel * batch_srcSize[batchCount].height * batch_srcSize[batchCount].width, sizeof(T));

                compute_unpadded_from_padded_host(dstPtrImage, batch_srcSize[batchCount], batch_srcSizeMax[batchCount], dstPtrImageUnpadded, chnFormat, channel);

                memcpy(dstPtrImageUnpaddedCopy, dstPtrImageUnpadded, channel * batch_srcSize[batchCount].height * batch_srcSize[batchCount].width * sizeof(T));

                compute_planar_to_packed_host(dstPtrImageUnpaddedCopy, batch_srcSize[batchCount], dstPtrImageUnpadded, channel);

                memset(dstPtrImage, (T) 0, imageDimMax * channel * sizeof(T));

                compute_padded_from_unpadded_host(dstPtrImageUnpadded, batch_srcSize[batchCount], batch_srcSizeMax[batchCount], dstPtrImage, RPPI_CHN_PACKED, channel);

                free(dstPtrImageUnpadded);
                free(dstPtrImageUnpaddedCopy);
            }
        }
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        omp_set_dynamic(0);
#pragma omp parallel for num_threads(nbatchSize)
        for(int batchCount = 0; batchCount < nbatchSize; batchCount ++)
        {
            Rpp32u imageDimMax = batch_srcSizeMax[batchCount].height * batch_srcSizeMax[batchCount].width;

            Rpp32f ampl_x = batch_ampl_x[batchCount];
            Rpp32f ampl_y = batch_ampl_y[batchCount];
            Rpp32f freq_x = batch_freq_x[batchCount];
            Rpp32f freq_y = batch_freq_y[batchCount];
            Rpp32f phase_x = batch_phase_x[batchCount];
            Rpp32f phase_y = batch_phase_y[batchCount];

            T *srcPtrImage, *dstPtrImage;
            Rpp32u loc = 0;
            compute_image_location_host(batch_srcSizeMax, batchCount, &loc, channel);
            srcPtrImage = srcPtr + loc;
            dstPtrImage = dstPtr + loc;

            Rpp32u elementsInRowMax = channel * batch_srcSizeMax[batchCount].width;

            for(int i = 0; i < batch_srcSize[batchCount].height; i++)
            {
                T *dstPtrTemp;
                dstPtrTemp = dstPtrImage + (i * elementsInRowMax);

                Rpp32u bufferLength = batch_srcSize[batchCount].width;
                Rpp32u alignedLength = bufferLength & ~3;

                __m128 pI = _mm_set1_ps((Rpp32f)i);
                __m128 pJ, pWaterI, pWaterJ;
                __m128 pAmplX = _mm_set1_ps(ampl_x);
                __m128 pAmplY = _mm_set1_ps(ampl_y);
                __m128 pFreqX = _mm_set1_ps(freq_x);
                __m128 pFreqY = _mm_set1_ps(freq_y);
                __m128 pPhaseX = _mm_set1_ps(phase_x);
                __m128 pPhaseY = _mm_set1_ps(phase_y);
                __m128 p0, p1, p2, p3;

                Rpp32f waterI[4], waterJ[4];

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount+=4)
                {
                    pJ = _mm_setr_ps((Rpp32f)(vectorLoopCount), (Rpp32f)(vectorLoopCount + 1), (Rpp32f)(vectorLoopCount + 2), (Rpp32f)(vectorLoopCount + 3));
                    p0 = _mm_mul_ps(pFreqX, pI);
                    p0 = _mm_add_ps(p0, pPhaseX);
                    sincos_ps(p0, &p1, &p2);
                    p1 = _mm_mul_ps(p1, pAmplX);
                    pWaterJ = _mm_add_ps(pJ, p1);
                    p0 = _mm_mul_ps(pFreqY, pJ);
                    p0 = _mm_add_ps(p0, pPhaseY);
                    sincos_ps(p0, &p1, &p2);
                    p2 = _mm_mul_ps(p2, pAmplY);
                    pWaterI = _mm_add_ps(pI, p2);

                    _mm_storeu_ps(waterI, pWaterI);
                    _mm_storeu_ps(waterJ, pWaterJ);

                    for (int count = 0; count < 4; count++)
                    {
                        Rpp32u waterIint, waterJint;
                        waterIint = (Rpp32u) RPPPRANGECHECK(waterI[count], 0, batch_srcSize[batchCount].height - 1);
                        waterJint = (Rpp32u) RPPPRANGECHECK(waterJ[count], 0, batch_srcSize[batchCount].width - 1);

                        for (int c = 0; c < channel; c++)
                        {
                            *dstPtrTemp = *(srcPtrImage + (waterIint * elementsInRowMax) + (waterJint * channel) + c);
                            dstPtrTemp++;
                        }
                    }
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    Rpp32f locI = (Rpp32f) i;
                    Rpp32f locJ = (Rpp32f) vectorLoopCount;

                    Rpp32f waterLocJ = locJ + ampl_x * sin((freq_x * locI) + phase_x);
                    Rpp32f waterLocI = locI + ampl_y * cos((freq_y * locJ) + phase_y);

                    Rpp32u waterLocIint, waterLocJint;
                    waterLocIint = (Rpp32u) RPPPRANGECHECK(waterLocI, 0, batch_srcSize[batchCount].height);
                    waterLocJint = (Rpp32u) RPPPRANGECHECK(waterLocJ, 0, batch_srcSize[batchCount].width);

                    for (int c = 0; c < channel; c++)
                    {
                        *dstPtrTemp = *(srcPtrImage + (waterLocIint * elementsInRowMax) + (waterLocJint * channel) + c);
                        dstPtrTemp++;
                    }
                }
            }
            if (outputFormatToggle == 1)
            {
                T *dstPtrImageUnpadded = (T*) calloc(channel * batch_srcSize[batchCount].height * batch_srcSize[batchCount].width, sizeof(T));
                T *dstPtrImageUnpaddedCopy = (T*) calloc(channel * batch_srcSize[batchCount].height * batch_srcSize[batchCount].width, sizeof(T));

                compute_unpadded_from_padded_host(dstPtrImage, batch_srcSize[batchCount], batch_srcSizeMax[batchCount], dstPtrImageUnpadded, chnFormat, channel);

                memcpy(dstPtrImageUnpaddedCopy, dstPtrImageUnpadded, channel * batch_srcSize[batchCount].height * batch_srcSize[batchCount].width * sizeof(T));

                compute_packed_to_planar_host(dstPtrImageUnpaddedCopy, batch_srcSize[batchCount], dstPtrImageUnpadded, channel);

                memset(dstPtrImage, (T) 0, imageDimMax * channel * sizeof(T));

                compute_padded_from_unpadded_host(dstPtrImageUnpadded, batch_srcSize[batchCount], batch_srcSizeMax[batchCount], dstPtrImage, RPPI_CHN_PLANAR, channel);

                free(dstPtrImageUnpadded);
                free(dstPtrImageUnpaddedCopy);
            }
        }
    }

    return RPP_SUCCESS;
}

/**************** non_linear_blend ***************/

template <typename T>
RppStatus non_linear_blend_host_batch(T* srcPtr1, T* srcPtr2, RppiSize *batch_srcSize, RppiSize *batch_srcSizeMax, T* dstPtr,
                         Rpp32f *batch_std_dev,
                         Rpp32u outputFormatToggle, Rpp32u nbatchSize,
                         RppiChnFormat chnFormat, Rpp32u channel)
{
    if(chnFormat == RPPI_CHN_PLANAR)
    {
        omp_set_dynamic(0);
#pragma omp parallel for num_threads(nbatchSize)
        for(int batchCount = 0; batchCount < nbatchSize; batchCount ++)
        {
            Rpp32u imageDimMax = batch_srcSizeMax[batchCount].height * batch_srcSizeMax[batchCount].width;

            Rpp32f std_dev = batch_std_dev[batchCount];
            Rpp32f multiplier = - 1.0 / (2.0 * std_dev * std_dev);

            T *srcPtr1Image, *srcPtr2Image, *dstPtrImage;
            Rpp32u loc = 0;
            compute_image_location_host(batch_srcSizeMax, batchCount, &loc, channel);
            srcPtr1Image = srcPtr1 + loc;
            srcPtr2Image = srcPtr2 + loc;
            dstPtrImage = dstPtr + loc;

            Rpp32u elementsInRowMax = batch_srcSizeMax[batchCount].width;
            Rpp32s subtrahendI = (Rpp32s) (batch_srcSize[batchCount].height >> 1);
            Rpp32s subtrahendJ = (Rpp32s) (batch_srcSize[batchCount].width >> 1);

            for(int i = 0; i < batch_srcSize[batchCount].height; i++)
            {
                T *srcPtr1Temp, *srcPtr2Temp, *dstPtrTemp;
                srcPtr1Temp = srcPtr1Image + (i * elementsInRowMax);
                srcPtr2Temp = srcPtr2Image + (i * elementsInRowMax);
                dstPtrTemp = dstPtrImage + (i * elementsInRowMax);

                Rpp32f locI = (Rpp32f) (i - subtrahendI);
                Rpp32f relativeGaussian[4];

                Rpp32u bufferLength = batch_srcSize[batchCount].width;
                Rpp32u alignedLength = (bufferLength & ~3) - 4;

                __m128i const zero = _mm_setzero_si128();
                __m128 pMultiplier = _mm_set1_ps(multiplier);
                __m128 pZero = _mm_set1_ps(0.0);
                __m128 pOne = _mm_set1_ps(1.0);
                __m128 pExpI = _mm_set1_ps(locI * locI * multiplier);
                __m128 pJ, pExpJ, pRelGaussian;
                __m128i px0, px1, px2, px3;
                __m128 p0, p1;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount+=4)
                {
                    pJ = _mm_setr_ps(
                        (Rpp32f)(vectorLoopCount - subtrahendJ),
                        (Rpp32f)(vectorLoopCount + 1 - subtrahendJ),
                        (Rpp32f)(vectorLoopCount + 2 - subtrahendJ),
                        (Rpp32f)(vectorLoopCount + 3 - subtrahendJ)
                    );
                    pExpJ = _mm_mul_ps(pJ, pJ);
                    pExpJ = _mm_mul_ps(pExpJ, pMultiplier);
                    pRelGaussian = _mm_add_ps(pExpI, pExpJ);
                    pRelGaussian = fast_exp_sse(pRelGaussian);

                    for (int c = 0; c < channel; c++)
                    {
                        T *srcPtr1TempChannel, *srcPtr2TempChannel, *dstPtrTempChannel;
                        srcPtr1TempChannel = srcPtr1Temp + (c * imageDimMax);
                        srcPtr2TempChannel = srcPtr2Temp + (c * imageDimMax);
                        dstPtrTempChannel = dstPtrTemp + (c * imageDimMax);

                        px0 =  _mm_loadu_si128((__m128i *)srcPtr1TempChannel);
                        px0 = _mm_unpacklo_epi8(px0, zero);    // pixels 0-7
                        p0 = _mm_cvtepi32_ps(_mm_unpacklo_epi16(px0, zero));    // pixels 0-3

                        px0 =  _mm_loadu_si128((__m128i *)srcPtr2TempChannel);
                        px0 = _mm_unpacklo_epi8(px0, zero);    // pixels 0-7
                        p1 = _mm_cvtepi32_ps(_mm_unpacklo_epi16(px0, zero));    // pixels 0-3

                        p0 = _mm_mul_ps(pRelGaussian, p0);
                        p1 = _mm_mul_ps(_mm_sub_ps(pOne, pRelGaussian), p1);
                        p0 = _mm_add_ps(p0, p1);

                        px0 = _mm_cvtps_epi32(p0);
                        px1 = _mm_cvtps_epi32(pZero);
                        px2 = _mm_cvtps_epi32(pZero);
                        px3 = _mm_cvtps_epi32(pZero);

                        px0 = _mm_packus_epi32(px0, px1);
                        px1 = _mm_packus_epi32(px2, px3);

                        px0 = _mm_packus_epi16(px0, px1);    // pixels 0-15

                        _mm_storeu_si128((__m128i *)dstPtrTempChannel, px0);
                    }

                    srcPtr1Temp += 4;
                    srcPtr2Temp += 4;
                    dstPtrTemp += 4;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    Rpp32s locI = i - subtrahendI;
                    Rpp32s locJ = vectorLoopCount - subtrahendJ;

                    Rpp32f gaussianValue = gaussian_2d_relative(locI, locJ, std_dev);

                    for (int c = 0; c < channel; c++)
                    {
                        T *srcPtr1TempChannel, *srcPtr2TempChannel, *dstPtrTempChannel;
                        srcPtr1TempChannel = srcPtr1Temp + (c * imageDimMax);
                        srcPtr2TempChannel = srcPtr2Temp + (c * imageDimMax);
                        dstPtrTempChannel = dstPtrTemp + (c * imageDimMax);

                        *dstPtrTempChannel = (T) RPPPIXELCHECK((gaussianValue * ((Rpp32f) *srcPtr1TempChannel)) + ((1 - gaussianValue) * ((Rpp32f) *srcPtr2TempChannel)));
                    }

                    srcPtr1Temp++;
                    srcPtr2Temp++;
                    dstPtrTemp++;
                }
            }
            if (outputFormatToggle == 1)
            {
                T *dstPtrImageUnpadded = (T*) calloc(channel * batch_srcSize[batchCount].height * batch_srcSize[batchCount].width, sizeof(T));
                T *dstPtrImageUnpaddedCopy = (T*) calloc(channel * batch_srcSize[batchCount].height * batch_srcSize[batchCount].width, sizeof(T));

                compute_unpadded_from_padded_host(dstPtrImage, batch_srcSize[batchCount], batch_srcSizeMax[batchCount], dstPtrImageUnpadded, chnFormat, channel);

                memcpy(dstPtrImageUnpaddedCopy, dstPtrImageUnpadded, channel * batch_srcSize[batchCount].height * batch_srcSize[batchCount].width * sizeof(T));

                compute_planar_to_packed_host(dstPtrImageUnpaddedCopy, batch_srcSize[batchCount], dstPtrImageUnpadded, channel);

                memset(dstPtrImage, (T) 0, imageDimMax * channel * sizeof(T));

                compute_padded_from_unpadded_host(dstPtrImageUnpadded, batch_srcSize[batchCount], batch_srcSizeMax[batchCount], dstPtrImage, RPPI_CHN_PACKED, channel);

                free(dstPtrImageUnpadded);
                free(dstPtrImageUnpaddedCopy);
            }
        }
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        omp_set_dynamic(0);
#pragma omp parallel for num_threads(nbatchSize)
        for(int batchCount = 0; batchCount < nbatchSize; batchCount ++)
        {
            Rpp32u imageDimMax = batch_srcSizeMax[batchCount].height * batch_srcSizeMax[batchCount].width;

            Rpp32f std_dev = batch_std_dev[batchCount];
            Rpp32f multiplier = - 1.0 / (2.0 * std_dev * std_dev);

            T *srcPtr1Image, *srcPtr2Image, *dstPtrImage;
            Rpp32u loc = 0;
            compute_image_location_host(batch_srcSizeMax, batchCount, &loc, channel);
            srcPtr1Image = srcPtr1 + loc;
            srcPtr2Image = srcPtr2 + loc;
            dstPtrImage = dstPtr + loc;

            Rpp32u elementsInRowMax = channel * batch_srcSizeMax[batchCount].width;
            Rpp32s subtrahendI = (Rpp32s) (batch_srcSize[batchCount].height >> 1);
            Rpp32s subtrahendJ = (Rpp32s) (batch_srcSize[batchCount].width >> 1);

            for(int i = 0; i < batch_srcSize[batchCount].height; i++)
            {
                T *srcPtr1Temp, *srcPtr2Temp, *dstPtrTemp;
                srcPtr1Temp = srcPtr1Image + (i * elementsInRowMax);
                srcPtr2Temp = srcPtr2Image + (i * elementsInRowMax);
                dstPtrTemp = dstPtrImage + (i * elementsInRowMax);

                Rpp32f locI = (Rpp32f) (i - subtrahendI);
                Rpp32f relativeGaussian[4];

                Rpp32u bufferLength = batch_srcSize[batchCount].width;
                Rpp32u alignedLength = bufferLength & ~3;

                __m128i const zero = _mm_setzero_si128();
                __m128 pMultiplier = _mm_set1_ps(multiplier);
                __m128 pZero = _mm_set1_ps(0.0);
                __m128 pOne = _mm_set1_ps(1.0);
                __m128 pExpI = _mm_set1_ps(locI * locI * multiplier);
                __m128 pJ, pExpJ, pRelGaussian;
                __m128i px0, px1, px2, px3;
                __m128 p0, p1, p2, p3, p4, p5, p6, p7;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount+=4)
                {
                    pJ = _mm_setr_ps(
                        (Rpp32f)(vectorLoopCount - subtrahendJ),
                        (Rpp32f)(vectorLoopCount + 1 - subtrahendJ),
                        (Rpp32f)(vectorLoopCount + 2 - subtrahendJ),
                        (Rpp32f)(vectorLoopCount + 3 - subtrahendJ)
                    );
                    pExpJ = _mm_mul_ps(pJ, pJ);
                    pExpJ = _mm_mul_ps(pExpJ, pMultiplier);
                    pRelGaussian = _mm_add_ps(pExpI, pExpJ);
                    pRelGaussian = fast_exp_sse(pRelGaussian);

                    _mm_storeu_ps(relativeGaussian, pRelGaussian);

                    __m128 pRelGaussian0 = _mm_setr_ps(relativeGaussian[0], relativeGaussian[0], relativeGaussian[0], relativeGaussian[1]);
                    __m128 pRelGaussian1 = _mm_setr_ps(relativeGaussian[1], relativeGaussian[1], relativeGaussian[2], relativeGaussian[2]);
                    __m128 pRelGaussian2 = _mm_setr_ps(relativeGaussian[2], relativeGaussian[3], relativeGaussian[3], relativeGaussian[3]);

                    px0 =  _mm_loadu_si128((__m128i *)srcPtr1Temp);
                    px1 = _mm_unpackhi_epi8(px0, zero);    // pixels 8-15
                    px0 = _mm_unpacklo_epi8(px0, zero);    // pixels 0-7
                    p0 = _mm_cvtepi32_ps(_mm_unpacklo_epi16(px0, zero));    // pixels 0-3
                    p1 = _mm_cvtepi32_ps(_mm_unpackhi_epi16(px0, zero));    // pixels 4-7
                    p2 = _mm_cvtepi32_ps(_mm_unpacklo_epi16(px1, zero));    // pixels 8-11
                    p3 = _mm_cvtepi32_ps(_mm_unpackhi_epi16(px1, zero));    // pixels 12-15

                    px0 =  _mm_loadu_si128((__m128i *)srcPtr2Temp);
                    px1 = _mm_unpackhi_epi8(px0, zero);    // pixels 8-15
                    px0 = _mm_unpacklo_epi8(px0, zero);    // pixels 0-7
                    p4 = _mm_cvtepi32_ps(_mm_unpacklo_epi16(px0, zero));    // pixels 0-3
                    p5 = _mm_cvtepi32_ps(_mm_unpackhi_epi16(px0, zero));    // pixels 4-7
                    p6 = _mm_cvtepi32_ps(_mm_unpacklo_epi16(px1, zero));    // pixels 8-11
                    p7 = _mm_cvtepi32_ps(_mm_unpackhi_epi16(px1, zero));    // pixels 12-15

                    p0 = _mm_mul_ps(pRelGaussian0, p0);
                    p4 = _mm_mul_ps(_mm_sub_ps(pOne, pRelGaussian0), p4);
                    p0 = _mm_add_ps(p0, p4);

                    p1 = _mm_mul_ps(pRelGaussian1, p1);
                    p5 = _mm_mul_ps(_mm_sub_ps(pOne, pRelGaussian1), p5);
                    p1 = _mm_add_ps(p1, p5);

                    p2 = _mm_mul_ps(pRelGaussian2, p2);
                    p6 = _mm_mul_ps(_mm_sub_ps(pOne, pRelGaussian2), p6);
                    p2 = _mm_add_ps(p2, p6);

                    px0 = _mm_cvtps_epi32(p0);
                    px1 = _mm_cvtps_epi32(p1);
                    px2 = _mm_cvtps_epi32(p2);
                    px3 = _mm_cvtps_epi32(pZero);

                    px0 = _mm_packus_epi32(px0, px1);
                    px1 = _mm_packus_epi32(px2, px3);

                    px0 = _mm_packus_epi16(px0, px1);    // pixels 0-15

                    _mm_storeu_si128((__m128i *)dstPtrTemp, px0);

                    srcPtr1Temp += 12;
                    srcPtr2Temp += 12;
                    dstPtrTemp += 12;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    Rpp32s locI = i - subtrahendI;
                    Rpp32s locJ = vectorLoopCount - subtrahendJ;

                    Rpp32f gaussianValue = gaussian_2d_relative(locI, locJ, std_dev);

                    for (int c = 0; c < channel; c++)
                    {
                        *dstPtrTemp = (T) RPPPIXELCHECK((gaussianValue * ((Rpp32f) *srcPtr1Temp)) + ((1 - gaussianValue) * ((Rpp32f) *srcPtr2Temp)));
                        srcPtr1Temp++;
                        srcPtr2Temp++;
                        dstPtrTemp++;
                    }
                }
            }
            if (outputFormatToggle == 1)
            {
                T *dstPtrImageUnpadded = (T*) calloc(channel * batch_srcSize[batchCount].height * batch_srcSize[batchCount].width, sizeof(T));
                T *dstPtrImageUnpaddedCopy = (T*) calloc(channel * batch_srcSize[batchCount].height * batch_srcSize[batchCount].width, sizeof(T));

                compute_unpadded_from_padded_host(dstPtrImage, batch_srcSize[batchCount], batch_srcSizeMax[batchCount], dstPtrImageUnpadded, chnFormat, channel);

                memcpy(dstPtrImageUnpaddedCopy, dstPtrImageUnpadded, channel * batch_srcSize[batchCount].height * batch_srcSize[batchCount].width * sizeof(T));

                compute_packed_to_planar_host(dstPtrImageUnpaddedCopy, batch_srcSize[batchCount], dstPtrImageUnpadded, channel);

                memset(dstPtrImage, (T) 0, imageDimMax * channel * sizeof(T));

                compute_padded_from_unpadded_host(dstPtrImageUnpadded, batch_srcSize[batchCount], batch_srcSizeMax[batchCount], dstPtrImage, RPPI_CHN_PLANAR, channel);

                free(dstPtrImageUnpadded);
                free(dstPtrImageUnpaddedCopy);
            }
        }
    }

    return RPP_SUCCESS;
}

RppStatus non_linear_blend_f32_host_batch(Rpp32f* srcPtr1, Rpp32f* srcPtr2, RppiSize *batch_srcSize, RppiSize *batch_srcSizeMax, Rpp32f* dstPtr,
                         Rpp32f *batch_std_dev,
                         Rpp32u outputFormatToggle, Rpp32u nbatchSize,
                         RppiChnFormat chnFormat, Rpp32u channel)
{
    if(chnFormat == RPPI_CHN_PLANAR)
    {
        omp_set_dynamic(0);
#pragma omp parallel for num_threads(nbatchSize)
        for(int batchCount = 0; batchCount < nbatchSize; batchCount ++)
        {
            Rpp32u imageDimMax = batch_srcSizeMax[batchCount].height * batch_srcSizeMax[batchCount].width;

            Rpp32f std_dev = batch_std_dev[batchCount];
            Rpp32f multiplier = - 1.0 / (2.0 * std_dev * std_dev);

            Rpp32f *srcPtr1Image, *srcPtr2Image, *dstPtrImage;
            Rpp32u loc = 0;
            compute_image_location_host(batch_srcSizeMax, batchCount, &loc, channel);
            srcPtr1Image = srcPtr1 + loc;
            srcPtr2Image = srcPtr2 + loc;
            dstPtrImage = dstPtr + loc;

            Rpp32u elementsInRowMax = batch_srcSizeMax[batchCount].width;
            Rpp32s subtrahendI = (Rpp32s) (batch_srcSize[batchCount].height >> 1);
            Rpp32s subtrahendJ = (Rpp32s) (batch_srcSize[batchCount].width >> 1);

            for(int i = 0; i < batch_srcSize[batchCount].height; i++)
            {
                Rpp32f *srcPtr1Temp, *srcPtr2Temp, *dstPtrTemp;
                srcPtr1Temp = srcPtr1Image + (i * elementsInRowMax);
                srcPtr2Temp = srcPtr2Image + (i * elementsInRowMax);
                dstPtrTemp = dstPtrImage + (i * elementsInRowMax);

                Rpp32f locI = (Rpp32f) (i - subtrahendI);
                Rpp32f relativeGaussian[4];

                Rpp32u bufferLength = batch_srcSize[batchCount].width;
                Rpp32u alignedLength = bufferLength & ~3;

                __m128 pMultiplier = _mm_set1_ps(multiplier);
                __m128 pOne = _mm_set1_ps(1.0);
                __m128 pExpI = _mm_set1_ps(locI * locI * multiplier);
                __m128 pJ, pExpJ, pRelGaussian;
                __m128 p0, p1;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount+=4)
                {
                    pJ = _mm_setr_ps(
                        (Rpp32f)(vectorLoopCount - subtrahendJ),
                        (Rpp32f)(vectorLoopCount + 1 - subtrahendJ),
                        (Rpp32f)(vectorLoopCount + 2 - subtrahendJ),
                        (Rpp32f)(vectorLoopCount + 3 - subtrahendJ)
                    );
                    pExpJ = _mm_mul_ps(pJ, pJ);
                    pExpJ = _mm_mul_ps(pExpJ, pMultiplier);
                    pRelGaussian = _mm_add_ps(pExpI, pExpJ);
                    pRelGaussian = fast_exp_sse(pRelGaussian);

                    for (int c = 0; c < channel; c++)
                    {
                        Rpp32f *srcPtr1TempChannel, *srcPtr2TempChannel, *dstPtrTempChannel;
                        srcPtr1TempChannel = srcPtr1Temp + (c * imageDimMax);
                        srcPtr2TempChannel = srcPtr2Temp + (c * imageDimMax);
                        dstPtrTempChannel = dstPtrTemp + (c * imageDimMax);

                        p0 = _mm_loadu_ps(srcPtr1TempChannel);
                        p1 = _mm_loadu_ps(srcPtr2TempChannel);

                        p0 = _mm_mul_ps(pRelGaussian, p0);
                        p1 = _mm_mul_ps(_mm_sub_ps(pOne, pRelGaussian), p1);
                        p0 = _mm_add_ps(p0, p1);

                        _mm_storeu_ps(dstPtrTempChannel, p0);
                    }

                    srcPtr1Temp += 4;
                    srcPtr2Temp += 4;
                    dstPtrTemp += 4;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    Rpp32s locI = i - subtrahendI;
                    Rpp32s locJ = vectorLoopCount - subtrahendJ;

                    Rpp32f gaussianValue = gaussian_2d_relative(locI, locJ, std_dev);

                    for (int c = 0; c < channel; c++)
                    {
                        Rpp32f *srcPtr1TempChannel, *srcPtr2TempChannel, *dstPtrTempChannel;
                        srcPtr1TempChannel = srcPtr1Temp + (c * imageDimMax);
                        srcPtr2TempChannel = srcPtr2Temp + (c * imageDimMax);
                        dstPtrTempChannel = dstPtrTemp + (c * imageDimMax);

                        *dstPtrTempChannel = (gaussianValue * *srcPtr1TempChannel) + ((1 - gaussianValue) * *srcPtr2TempChannel);
                    }

                    srcPtr1Temp++;
                    srcPtr2Temp++;
                    dstPtrTemp++;
                }
            }
            if (outputFormatToggle == 1)
            {
                Rpp32f *dstPtrImageUnpadded = (Rpp32f*) calloc(channel * batch_srcSize[batchCount].height * batch_srcSize[batchCount].width, sizeof(Rpp32f));
                Rpp32f *dstPtrImageUnpaddedCopy = (Rpp32f*) calloc(channel * batch_srcSize[batchCount].height * batch_srcSize[batchCount].width, sizeof(Rpp32f));

                compute_unpadded_from_padded_host(dstPtrImage, batch_srcSize[batchCount], batch_srcSizeMax[batchCount], dstPtrImageUnpadded, chnFormat, channel);

                memcpy(dstPtrImageUnpaddedCopy, dstPtrImageUnpadded, channel * batch_srcSize[batchCount].height * batch_srcSize[batchCount].width * sizeof(Rpp32f));

                compute_planar_to_packed_host(dstPtrImageUnpaddedCopy, batch_srcSize[batchCount], dstPtrImageUnpadded, channel);

                memset(dstPtrImage, (Rpp32f) 0, imageDimMax * channel * sizeof(Rpp32f));

                compute_padded_from_unpadded_host(dstPtrImageUnpadded, batch_srcSize[batchCount], batch_srcSizeMax[batchCount], dstPtrImage, RPPI_CHN_PACKED, channel);

                free(dstPtrImageUnpadded);
                free(dstPtrImageUnpaddedCopy);
            }
        }
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        omp_set_dynamic(0);
#pragma omp parallel for num_threads(nbatchSize)
        for(int batchCount = 0; batchCount < nbatchSize; batchCount ++)
        {
            Rpp32u imageDimMax = batch_srcSizeMax[batchCount].height * batch_srcSizeMax[batchCount].width;

            Rpp32f std_dev = batch_std_dev[batchCount];
            Rpp32f multiplier = - 1.0 / (2.0 * std_dev * std_dev);

            Rpp32f *srcPtr1Image, *srcPtr2Image, *dstPtrImage;
            Rpp32u loc = 0;
            compute_image_location_host(batch_srcSizeMax, batchCount, &loc, channel);
            srcPtr1Image = srcPtr1 + loc;
            srcPtr2Image = srcPtr2 + loc;
            dstPtrImage = dstPtr + loc;

            Rpp32u elementsInRowMax = channel * batch_srcSizeMax[batchCount].width;
            Rpp32s subtrahendI = (Rpp32s) (batch_srcSize[batchCount].height >> 1);
            Rpp32s subtrahendJ = (Rpp32s) (batch_srcSize[batchCount].width >> 1);

            for(int i = 0; i < batch_srcSize[batchCount].height; i++)
            {
                Rpp32f *srcPtr1Temp, *srcPtr2Temp, *dstPtrTemp;
                srcPtr1Temp = srcPtr1Image + (i * elementsInRowMax);
                srcPtr2Temp = srcPtr2Image + (i * elementsInRowMax);
                dstPtrTemp = dstPtrImage + (i * elementsInRowMax);

                Rpp32f locI = (Rpp32f) (i - subtrahendI);
                Rpp32f relativeGaussian[4];

                Rpp32u bufferLength = batch_srcSize[batchCount].width;
                Rpp32u alignedLength = bufferLength & ~3;

                __m128 pMultiplier = _mm_set1_ps(multiplier);
                __m128 pOne = _mm_set1_ps(1.0);
                __m128 pExpI = _mm_set1_ps(locI * locI * multiplier);
                __m128 pJ, pExpJ, pRelGaussian;
                __m128 p0, p1, p2, p4, p5, p6;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount+=4)
                {
                    pJ = _mm_setr_ps(
                        (Rpp32f)(vectorLoopCount - subtrahendJ),
                        (Rpp32f)(vectorLoopCount + 1 - subtrahendJ),
                        (Rpp32f)(vectorLoopCount + 2 - subtrahendJ),
                        (Rpp32f)(vectorLoopCount + 3 - subtrahendJ)
                    );
                    pExpJ = _mm_mul_ps(pJ, pJ);
                    pExpJ = _mm_mul_ps(pExpJ, pMultiplier);
                    pRelGaussian = _mm_add_ps(pExpI, pExpJ);
                    pRelGaussian = fast_exp_sse(pRelGaussian);

                    _mm_storeu_ps(relativeGaussian, pRelGaussian);

                    __m128 pRelGaussian0 = _mm_setr_ps(relativeGaussian[0], relativeGaussian[0], relativeGaussian[0], relativeGaussian[1]);
                    __m128 pRelGaussian1 = _mm_setr_ps(relativeGaussian[1], relativeGaussian[1], relativeGaussian[2], relativeGaussian[2]);
                    __m128 pRelGaussian2 = _mm_setr_ps(relativeGaussian[2], relativeGaussian[3], relativeGaussian[3], relativeGaussian[3]);

                    p0 = _mm_loadu_ps(srcPtr1Temp);
                    p1 = _mm_loadu_ps(srcPtr1Temp + 4);
                    p2 = _mm_loadu_ps(srcPtr1Temp + 8);

                    p4 = _mm_loadu_ps(srcPtr2Temp);
                    p5 = _mm_loadu_ps(srcPtr2Temp + 4);
                    p6 = _mm_loadu_ps(srcPtr2Temp + 8);

                    p0 = _mm_mul_ps(pRelGaussian0, p0);
                    p4 = _mm_mul_ps(_mm_sub_ps(pOne, pRelGaussian0), p4);
                    p0 = _mm_add_ps(p0, p4);

                    p1 = _mm_mul_ps(pRelGaussian1, p1);
                    p5 = _mm_mul_ps(_mm_sub_ps(pOne, pRelGaussian1), p5);
                    p1 = _mm_add_ps(p1, p5);

                    p2 = _mm_mul_ps(pRelGaussian2, p2);
                    p6 = _mm_mul_ps(_mm_sub_ps(pOne, pRelGaussian2), p6);
                    p2 = _mm_add_ps(p2, p6);

                    _mm_storeu_ps(dstPtrTemp, p0);
                    _mm_storeu_ps(dstPtrTemp + 4, p1);
                    _mm_storeu_ps(dstPtrTemp + 8, p2);

                    srcPtr1Temp += 12;
                    srcPtr2Temp += 12;
                    dstPtrTemp += 12;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    Rpp32s locI = i - subtrahendI;
                    Rpp32s locJ = vectorLoopCount - subtrahendJ;

                    Rpp32f gaussianValue = gaussian_2d_relative(locI, locJ, std_dev);

                    for (int c = 0; c < channel; c++)
                    {
                        *dstPtrTemp = (gaussianValue * *srcPtr1Temp) + ((1 - gaussianValue) * *srcPtr2Temp);
                        srcPtr1Temp++;
                        srcPtr2Temp++;
                        dstPtrTemp++;
                    }
                }
            }
            if (outputFormatToggle == 1)
            {
                Rpp32f *dstPtrImageUnpadded = (Rpp32f*) calloc(channel * batch_srcSize[batchCount].height * batch_srcSize[batchCount].width, sizeof(Rpp32f));
                Rpp32f *dstPtrImageUnpaddedCopy = (Rpp32f*) calloc(channel * batch_srcSize[batchCount].height * batch_srcSize[batchCount].width, sizeof(Rpp32f));

                compute_unpadded_from_padded_host(dstPtrImage, batch_srcSize[batchCount], batch_srcSizeMax[batchCount], dstPtrImageUnpadded, chnFormat, channel);

                memcpy(dstPtrImageUnpaddedCopy, dstPtrImageUnpadded, channel * batch_srcSize[batchCount].height * batch_srcSize[batchCount].width * sizeof(Rpp32f));

                compute_packed_to_planar_host(dstPtrImageUnpaddedCopy, batch_srcSize[batchCount], dstPtrImageUnpadded, channel);

                memset(dstPtrImage, (Rpp32f) 0, imageDimMax * channel * sizeof(Rpp32f));

                compute_padded_from_unpadded_host(dstPtrImageUnpadded, batch_srcSize[batchCount], batch_srcSizeMax[batchCount], dstPtrImage, RPPI_CHN_PLANAR, channel);

                free(dstPtrImageUnpadded);
                free(dstPtrImageUnpaddedCopy);
            }
        }
    }

    return RPP_SUCCESS;
}

RppStatus non_linear_blend_f16_host_batch(Rpp16f* srcPtr1, Rpp16f* srcPtr2, RppiSize *batch_srcSize, RppiSize *batch_srcSizeMax, Rpp16f* dstPtr,
                         Rpp32f *batch_std_dev,
                         Rpp32u outputFormatToggle, Rpp32u nbatchSize,
                         RppiChnFormat chnFormat, Rpp32u channel)
{
    if(chnFormat == RPPI_CHN_PLANAR)
    {
        omp_set_dynamic(0);
#pragma omp parallel for num_threads(nbatchSize)
        for(int batchCount = 0; batchCount < nbatchSize; batchCount ++)
        {
            Rpp32u imageDimMax = batch_srcSizeMax[batchCount].height * batch_srcSizeMax[batchCount].width;

            Rpp32f std_dev = batch_std_dev[batchCount];
            Rpp32f multiplier = - 1.0 / (2.0 * std_dev * std_dev);

            Rpp16f *srcPtr1Image, *srcPtr2Image, *dstPtrImage;
            Rpp32u loc = 0;
            compute_image_location_host(batch_srcSizeMax, batchCount, &loc, channel);
            srcPtr1Image = srcPtr1 + loc;
            srcPtr2Image = srcPtr2 + loc;
            dstPtrImage = dstPtr + loc;

            Rpp32u elementsInRowMax = batch_srcSizeMax[batchCount].width;
            Rpp32s subtrahendI = (Rpp32s) (batch_srcSize[batchCount].height >> 1);
            Rpp32s subtrahendJ = (Rpp32s) (batch_srcSize[batchCount].width >> 1);

            Rpp32f srcPtr1TempChannelps[4], srcPtr2TempChannelps[4], dstPtrTempChannelps[4];

            for(int i = 0; i < batch_srcSize[batchCount].height; i++)
            {
                Rpp16f *srcPtr1Temp, *srcPtr2Temp, *dstPtrTemp;
                srcPtr1Temp = srcPtr1Image + (i * elementsInRowMax);
                srcPtr2Temp = srcPtr2Image + (i * elementsInRowMax);
                dstPtrTemp = dstPtrImage + (i * elementsInRowMax);

                Rpp32f locI = (Rpp32f) (i - subtrahendI);
                Rpp32f relativeGaussian[4];

                Rpp32u bufferLength = batch_srcSize[batchCount].width;
                Rpp32u alignedLength = bufferLength & ~3;

                __m128 pMultiplier = _mm_set1_ps(multiplier);
                __m128 pOne = _mm_set1_ps(1.0);
                __m128 pExpI = _mm_set1_ps(locI * locI * multiplier);
                __m128 pJ, pExpJ, pRelGaussian;
                __m128 p0, p1;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount+=4)
                {
                    pJ = _mm_setr_ps(
                        (Rpp32f)(vectorLoopCount - subtrahendJ),
                        (Rpp32f)(vectorLoopCount + 1 - subtrahendJ),
                        (Rpp32f)(vectorLoopCount + 2 - subtrahendJ),
                        (Rpp32f)(vectorLoopCount + 3 - subtrahendJ)
                    );
                    pExpJ = _mm_mul_ps(pJ, pJ);
                    pExpJ = _mm_mul_ps(pExpJ, pMultiplier);
                    pRelGaussian = _mm_add_ps(pExpI, pExpJ);
                    pRelGaussian = fast_exp_sse(pRelGaussian);

                    for (int c = 0; c < channel; c++)
                    {
                        Rpp16f *srcPtr1TempChannel, *srcPtr2TempChannel, *dstPtrTempChannel;
                        srcPtr1TempChannel = srcPtr1Temp + (c * imageDimMax);
                        srcPtr2TempChannel = srcPtr2Temp + (c * imageDimMax);
                        dstPtrTempChannel = dstPtrTemp + (c * imageDimMax);

                        for(int cnt = 0; cnt < 4; cnt++)
                        {
                            *(srcPtr1TempChannelps + cnt) = (Rpp32f) (*(srcPtr1TempChannel + cnt));
                            *(srcPtr2TempChannelps + cnt) = (Rpp32f) (*(srcPtr2TempChannel + cnt));
                        }

                        p0 = _mm_loadu_ps(srcPtr1TempChannelps);
                        p1 = _mm_loadu_ps(srcPtr2TempChannelps);

                        p0 = _mm_mul_ps(pRelGaussian, p0);
                        p1 = _mm_mul_ps(_mm_sub_ps(pOne, pRelGaussian), p1);
                        p0 = _mm_add_ps(p0, p1);

                        _mm_storeu_ps(dstPtrTempChannelps, p0);

                        for(int cnt = 0; cnt < 4; cnt++)
                        {
                            *(dstPtrTempChannel + cnt) = (Rpp16f) (*(dstPtrTempChannelps + cnt));
                        }
                    }

                    srcPtr1Temp += 4;
                    srcPtr2Temp += 4;
                    dstPtrTemp += 4;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    Rpp32s locI = i - subtrahendI;
                    Rpp32s locJ = vectorLoopCount - subtrahendJ;

                    Rpp32f gaussianValue = gaussian_2d_relative(locI, locJ, std_dev);

                    for (int c = 0; c < channel; c++)
                    {
                        Rpp16f *srcPtr1TempChannel, *srcPtr2TempChannel, *dstPtrTempChannel;
                        srcPtr1TempChannel = srcPtr1Temp + (c * imageDimMax);
                        srcPtr2TempChannel = srcPtr2Temp + (c * imageDimMax);
                        dstPtrTempChannel = dstPtrTemp + (c * imageDimMax);

                        *dstPtrTempChannel = (Rpp16f) ((gaussianValue * *srcPtr1TempChannel) + ((1 - gaussianValue) * *srcPtr2TempChannel));
                    }

                    srcPtr1Temp++;
                    srcPtr2Temp++;
                    dstPtrTemp++;
                }
            }
            if (outputFormatToggle == 1)
            {
                Rpp16f *dstPtrImageUnpadded = (Rpp16f*) calloc(channel * batch_srcSize[batchCount].height * batch_srcSize[batchCount].width, sizeof(Rpp16f));
                Rpp16f *dstPtrImageUnpaddedCopy = (Rpp16f*) calloc(channel * batch_srcSize[batchCount].height * batch_srcSize[batchCount].width, sizeof(Rpp16f));

                compute_unpadded_from_padded_host(dstPtrImage, batch_srcSize[batchCount], batch_srcSizeMax[batchCount], dstPtrImageUnpadded, chnFormat, channel);

                memcpy(dstPtrImageUnpaddedCopy, dstPtrImageUnpadded, channel * batch_srcSize[batchCount].height * batch_srcSize[batchCount].width * sizeof(Rpp16f));

                compute_planar_to_packed_host(dstPtrImageUnpaddedCopy, batch_srcSize[batchCount], dstPtrImageUnpadded, channel);

                memset(dstPtrImage, (Rpp16f) 0, imageDimMax * channel * sizeof(Rpp16f));

                compute_padded_from_unpadded_host(dstPtrImageUnpadded, batch_srcSize[batchCount], batch_srcSizeMax[batchCount], dstPtrImage, RPPI_CHN_PACKED, channel);

                free(dstPtrImageUnpadded);
                free(dstPtrImageUnpaddedCopy);
            }
        }
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        omp_set_dynamic(0);
#pragma omp parallel for num_threads(nbatchSize)
        for(int batchCount = 0; batchCount < nbatchSize; batchCount ++)
        {
            Rpp32u imageDimMax = batch_srcSizeMax[batchCount].height * batch_srcSizeMax[batchCount].width;

            Rpp32f std_dev = batch_std_dev[batchCount];
            Rpp32f multiplier = - 1.0 / (2.0 * std_dev * std_dev);

            Rpp16f *srcPtr1Image, *srcPtr2Image, *dstPtrImage;
            Rpp32u loc = 0;
            compute_image_location_host(batch_srcSizeMax, batchCount, &loc, channel);
            srcPtr1Image = srcPtr1 + loc;
            srcPtr2Image = srcPtr2 + loc;
            dstPtrImage = dstPtr + loc;

            Rpp32u elementsInRowMax = channel * batch_srcSizeMax[batchCount].width;
            Rpp32s subtrahendI = (Rpp32s) (batch_srcSize[batchCount].height >> 1);
            Rpp32s subtrahendJ = (Rpp32s) (batch_srcSize[batchCount].width >> 1);

            Rpp32f srcPtr1Tempps[12], srcPtr2Tempps[12], dstPtrTempps[12];

            for(int i = 0; i < batch_srcSize[batchCount].height; i++)
            {
                Rpp16f *srcPtr1Temp, *srcPtr2Temp, *dstPtrTemp;
                srcPtr1Temp = srcPtr1Image + (i * elementsInRowMax);
                srcPtr2Temp = srcPtr2Image + (i * elementsInRowMax);
                dstPtrTemp = dstPtrImage + (i * elementsInRowMax);

                Rpp32f locI = (Rpp32f) (i - subtrahendI);
                Rpp32f relativeGaussian[4];

                Rpp32u bufferLength = batch_srcSize[batchCount].width;
                Rpp32u alignedLength = bufferLength & ~3;

                __m128 pMultiplier = _mm_set1_ps(multiplier);
                __m128 pOne = _mm_set1_ps(1.0);
                __m128 pExpI = _mm_set1_ps(locI * locI * multiplier);
                __m128 pJ, pExpJ, pRelGaussian;
                __m128 p0, p1, p2, p4, p5, p6;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount+=4)
                {
                    pJ = _mm_setr_ps(
                        (Rpp32f)(vectorLoopCount - subtrahendJ),
                        (Rpp32f)(vectorLoopCount + 1 - subtrahendJ),
                        (Rpp32f)(vectorLoopCount + 2 - subtrahendJ),
                        (Rpp16f)(vectorLoopCount + 3 - subtrahendJ)
                    );
                    pExpJ = _mm_mul_ps(pJ, pJ);
                    pExpJ = _mm_mul_ps(pExpJ, pMultiplier);
                    pRelGaussian = _mm_add_ps(pExpI, pExpJ);
                    pRelGaussian = fast_exp_sse(pRelGaussian);

                    _mm_storeu_ps(relativeGaussian, pRelGaussian);

                    __m128 pRelGaussian0 = _mm_setr_ps(relativeGaussian[0], relativeGaussian[0], relativeGaussian[0], relativeGaussian[1]);
                    __m128 pRelGaussian1 = _mm_setr_ps(relativeGaussian[1], relativeGaussian[1], relativeGaussian[2], relativeGaussian[2]);
                    __m128 pRelGaussian2 = _mm_setr_ps(relativeGaussian[2], relativeGaussian[3], relativeGaussian[3], relativeGaussian[3]);

                    for(int cnt = 0; cnt < 12; cnt++)
                    {
                        *(srcPtr1Tempps + cnt) = (Rpp32f) (*(srcPtr1Temp + cnt));
                        *(srcPtr2Tempps + cnt) = (Rpp32f) (*(srcPtr2Temp + cnt));
                    }

                    p0 = _mm_loadu_ps(srcPtr1Tempps);
                    p1 = _mm_loadu_ps(srcPtr1Tempps + 4);
                    p2 = _mm_loadu_ps(srcPtr1Tempps + 8);

                    p4 = _mm_loadu_ps(srcPtr2Tempps);
                    p5 = _mm_loadu_ps(srcPtr2Tempps + 4);
                    p6 = _mm_loadu_ps(srcPtr2Tempps + 8);

                    p0 = _mm_mul_ps(pRelGaussian0, p0);
                    p4 = _mm_mul_ps(_mm_sub_ps(pOne, pRelGaussian0), p4);
                    p0 = _mm_add_ps(p0, p4);

                    p1 = _mm_mul_ps(pRelGaussian1, p1);
                    p5 = _mm_mul_ps(_mm_sub_ps(pOne, pRelGaussian1), p5);
                    p1 = _mm_add_ps(p1, p5);

                    p2 = _mm_mul_ps(pRelGaussian2, p2);
                    p6 = _mm_mul_ps(_mm_sub_ps(pOne, pRelGaussian2), p6);
                    p2 = _mm_add_ps(p2, p6);

                    _mm_storeu_ps(dstPtrTempps, p0);
                    _mm_storeu_ps(dstPtrTempps + 4, p1);
                    _mm_storeu_ps(dstPtrTempps + 8, p2);

                    for(int cnt = 0; cnt < 12; cnt++)
                    {
                        *(dstPtrTemp + cnt) = (Rpp16f) (*(dstPtrTempps + cnt));
                    }

                    srcPtr1Temp += 12;
                    srcPtr2Temp += 12;
                    dstPtrTemp += 12;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    Rpp32s locI = i - subtrahendI;
                    Rpp32s locJ = vectorLoopCount - subtrahendJ;

                    Rpp32f gaussianValue = gaussian_2d_relative(locI, locJ, std_dev);

                    for (int c = 0; c < channel; c++)
                    {
                        *dstPtrTemp = (Rpp16f) ((gaussianValue * *srcPtr1Temp) + ((1 - gaussianValue) * *srcPtr2Temp));
                        srcPtr1Temp++;
                        srcPtr2Temp++;
                        dstPtrTemp++;
                    }
                }
            }
            if (outputFormatToggle == 1)
            {
                Rpp16f *dstPtrImageUnpadded = (Rpp16f*) calloc(channel * batch_srcSize[batchCount].height * batch_srcSize[batchCount].width, sizeof(Rpp16f));
                Rpp16f *dstPtrImageUnpaddedCopy = (Rpp16f*) calloc(channel * batch_srcSize[batchCount].height * batch_srcSize[batchCount].width, sizeof(Rpp16f));

                compute_unpadded_from_padded_host(dstPtrImage, batch_srcSize[batchCount], batch_srcSizeMax[batchCount], dstPtrImageUnpadded, chnFormat, channel);

                memcpy(dstPtrImageUnpaddedCopy, dstPtrImageUnpadded, channel * batch_srcSize[batchCount].height * batch_srcSize[batchCount].width * sizeof(Rpp16f));

                compute_packed_to_planar_host(dstPtrImageUnpaddedCopy, batch_srcSize[batchCount], dstPtrImageUnpadded, channel);

                memset(dstPtrImage, (Rpp16f) 0, imageDimMax * channel * sizeof(Rpp16f));

                compute_padded_from_unpadded_host(dstPtrImageUnpadded, batch_srcSize[batchCount], batch_srcSizeMax[batchCount], dstPtrImage, RPPI_CHN_PLANAR, channel);

                free(dstPtrImageUnpadded);
                free(dstPtrImageUnpaddedCopy);
            }
        }
    }

    return RPP_SUCCESS;
}

RppStatus non_linear_blend_i8_host_batch(Rpp8s* srcPtr1, Rpp8s* srcPtr2, RppiSize *batch_srcSize, RppiSize *batch_srcSizeMax, Rpp8s* dstPtr,
                         Rpp32f *batch_std_dev,
                         Rpp32u outputFormatToggle, Rpp32u nbatchSize,
                         RppiChnFormat chnFormat, Rpp32u channel)
{
    Rpp64u bufferLength = batch_srcSizeMax[0].height * batch_srcSizeMax[0].width * channel * nbatchSize;

    Rpp8u *srcPtr1_8u = (Rpp8u*) calloc(bufferLength, sizeof(Rpp8u));
    Rpp8u *srcPtr2_8u = (Rpp8u*) calloc(bufferLength, sizeof(Rpp8u));
    Rpp8u *dstPtr_8u = (Rpp8u*) calloc(bufferLength, sizeof(Rpp8u));

    Rpp8s *srcPtr1Temp, *srcPtr2Temp;
    srcPtr1Temp = srcPtr1;
    srcPtr2Temp = srcPtr2;

    Rpp8u *srcPtr1_8uTemp, *srcPtr2_8uTemp;
    srcPtr1_8uTemp = srcPtr1_8u;
    srcPtr2_8uTemp = srcPtr2_8u;

    for (int i = 0; i < bufferLength; i++)
    {
        *srcPtr1_8uTemp = (Rpp8u) RPPPIXELCHECK(((Rpp32s) *srcPtr1Temp) + 128);
        *srcPtr2_8uTemp = (Rpp8u) RPPPIXELCHECK(((Rpp32s) *srcPtr2Temp) + 128);
        srcPtr1Temp++;
        srcPtr2Temp++;
        srcPtr1_8uTemp++;
        srcPtr2_8uTemp++;
    }

    non_linear_blend_host_batch<Rpp8u>(srcPtr1_8u, srcPtr2_8u, batch_srcSize, batch_srcSizeMax, dstPtr_8u, batch_std_dev, outputFormatToggle, nbatchSize, chnFormat, channel);

    Rpp8s *dstPtrTemp;
    dstPtrTemp = dstPtr;

    Rpp8u *dstPtr_8uTemp;
    dstPtr_8uTemp = dstPtr_8u;

    for (int i = 0; i < bufferLength; i++)
    {
        *dstPtrTemp = (Rpp8s) (((Rpp32s) *dstPtr_8uTemp) - 128);
        dstPtrTemp++;
        dstPtr_8uTemp++;
    }

    free(srcPtr1_8u);
    free(srcPtr2_8u);
    free(dstPtr_8u);

    return RPP_SUCCESS;
}

/**************** color_cast ***************/

template <typename T>
RppStatus color_cast_host_batch(T* srcPtr, RppiSize *batch_srcSize, RppiSize *batch_srcSizeMax, T* dstPtr,
                         Rpp8u *batch_r, Rpp8u *batch_g, Rpp8u *batch_b, Rpp32f *batch_alpha,
                         Rpp32u outputFormatToggle, Rpp32u nbatchSize,
                         RppiChnFormat chnFormat, Rpp32u channel)
{
    if(chnFormat == RPPI_CHN_PLANAR)
    {
        omp_set_dynamic(0);
#pragma omp parallel for num_threads(nbatchSize)
        for(int batchCount = 0; batchCount < nbatchSize; batchCount ++)
        {
            Rpp32u imageDimMax = batch_srcSizeMax[batchCount].height * batch_srcSizeMax[batchCount].width;

            Rpp8u r = batch_r[batchCount];
            Rpp8u g = batch_g[batchCount];
            Rpp8u b = batch_b[batchCount];
            Rpp32f alpha = batch_alpha[batchCount];

            Rpp32f userPixel[3] = {(Rpp32f) b, (Rpp32f) g, (Rpp32f) r};

            T *srcPtrImage, *dstPtrImage;
            Rpp32u loc = 0;
            compute_image_location_host(batch_srcSizeMax, batchCount, &loc, channel);
            srcPtrImage = srcPtr + loc;
            dstPtrImage = dstPtr + loc;

            Rpp32u elementsInRowMax = batch_srcSizeMax[batchCount].width;

            for (int c = 0; c < channel; c++)
            {
                T *srcPtrChannel, *dstPtrChannel;
                srcPtrChannel = srcPtrImage + (c * imageDimMax);
                dstPtrChannel = dstPtrImage + (c * imageDimMax);

                __m128i const zero = _mm_setzero_si128();
                __m128 pZero = _mm_set1_ps(0.0);
                __m128 pOne = _mm_set1_ps(1.0);
                __m128 pUserPixel = _mm_set1_ps(userPixel[c]);
                __m128 pAlpha = _mm_set1_ps(alpha);

                for(int i = 0; i < batch_srcSize[batchCount].height; i++)
                {
                    T *srcPtrTemp, *dstPtrTemp;
                    srcPtrTemp = srcPtrChannel + (i * elementsInRowMax);
                    dstPtrTemp = dstPtrChannel + (i * elementsInRowMax);

                    Rpp32u bufferLength = batch_srcSize[batchCount].width;
                    Rpp32u alignedLength = (bufferLength & ~3) - 4;

                    __m128i px0, px1, px2, px3;
                    __m128 p0, p1;

                    int vectorLoopCount = 0;
                    for (; vectorLoopCount < alignedLength; vectorLoopCount+=4)
                    {
                        px0 =  _mm_loadu_si128((__m128i *)srcPtrTemp);
                        px0 = _mm_unpacklo_epi8(px0, zero);    // pixels 0-7
                        p0 = _mm_cvtepi32_ps(_mm_unpacklo_epi16(px0, zero));    // pixels 0-3

                        p0 = _mm_mul_ps(pAlpha, p0);
                        p1 = _mm_mul_ps(_mm_sub_ps(pOne, pAlpha), pUserPixel);
                        p0 = _mm_add_ps(p0, p1);

                        px0 = _mm_cvtps_epi32(p0);
                        px1 = _mm_cvtps_epi32(pZero);
                        px2 = _mm_cvtps_epi32(pZero);
                        px3 = _mm_cvtps_epi32(pZero);

                        px0 = _mm_packus_epi32(px0, px1);
                        px1 = _mm_packus_epi32(px2, px3);

                        px0 = _mm_packus_epi16(px0, px1);    // pixels 0-15

                        _mm_storeu_si128((__m128i *)dstPtrTemp, px0);

                        srcPtrTemp += 4;
                        dstPtrTemp += 4;
                    }
                    for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                    {
                        *dstPtrTemp = (T) RPPPIXELCHECK((alpha * ((Rpp32f) *srcPtrTemp)) + ((1 - alpha) * userPixel[c]));

                        srcPtrTemp++;
                        dstPtrTemp++;
                    }
                }
            }
            if (outputFormatToggle == 1)
            {
                T *dstPtrImageUnpadded = (T*) calloc(channel * batch_srcSize[batchCount].height * batch_srcSize[batchCount].width, sizeof(T));
                T *dstPtrImageUnpaddedCopy = (T*) calloc(channel * batch_srcSize[batchCount].height * batch_srcSize[batchCount].width, sizeof(T));

                compute_unpadded_from_padded_host(dstPtrImage, batch_srcSize[batchCount], batch_srcSizeMax[batchCount], dstPtrImageUnpadded, chnFormat, channel);

                memcpy(dstPtrImageUnpaddedCopy, dstPtrImageUnpadded, channel * batch_srcSize[batchCount].height * batch_srcSize[batchCount].width * sizeof(T));

                compute_planar_to_packed_host(dstPtrImageUnpaddedCopy, batch_srcSize[batchCount], dstPtrImageUnpadded, channel);

                memset(dstPtrImage, (T) 0, imageDimMax * channel * sizeof(T));

                compute_padded_from_unpadded_host(dstPtrImageUnpadded, batch_srcSize[batchCount], batch_srcSizeMax[batchCount], dstPtrImage, RPPI_CHN_PACKED, channel);

                free(dstPtrImageUnpadded);
                free(dstPtrImageUnpaddedCopy);
            }
        }
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        omp_set_dynamic(0);
#pragma omp parallel for num_threads(nbatchSize)
        for(int batchCount = 0; batchCount < nbatchSize; batchCount ++)
        {
            Rpp32u imageDimMax = batch_srcSizeMax[batchCount].height * batch_srcSizeMax[batchCount].width;

            Rpp8u r = batch_r[batchCount];
            Rpp8u g = batch_g[batchCount];
            Rpp8u b = batch_b[batchCount];
            Rpp32f alpha = batch_alpha[batchCount];

            Rpp32f userPixel[3] = {(Rpp32f) b, (Rpp32f) g, (Rpp32f) r};

            T *srcPtrImage, *dstPtrImage;
            Rpp32u loc = 0;
            compute_image_location_host(batch_srcSizeMax, batchCount, &loc, channel);
            srcPtrImage = srcPtr + loc;
            dstPtrImage = dstPtr + loc;

            Rpp32u elementsInRowMax = channel * batch_srcSizeMax[batchCount].width;

            for(int i = 0; i < batch_srcSize[batchCount].height; i++)
            {
                T *srcPtrTemp, *dstPtrTemp;
                srcPtrTemp = srcPtrImage + (i * elementsInRowMax);
                dstPtrTemp = dstPtrImage + (i * elementsInRowMax);

                Rpp32u bufferLength = batch_srcSize[batchCount].width;
                Rpp32u alignedLength = bufferLength & ~3;

                __m128i const zero = _mm_setzero_si128();
                __m128 pZero = _mm_set1_ps(0.0);
                __m128 pOne = _mm_set1_ps(1.0);
                __m128 pAlpha = _mm_set1_ps(alpha);
                __m128 pUserPixelPartialVector1 = _mm_setr_ps(userPixel[0], userPixel[1], userPixel[2], userPixel[0]);
                __m128 pUserPixelPartialVector2 = _mm_setr_ps(userPixel[1], userPixel[2], userPixel[0], userPixel[1]);
                __m128 pUserPixelPartialVector3 = _mm_setr_ps(userPixel[2], userPixel[0], userPixel[1], userPixel[2]);
                __m128i px0, px1, px2, px3;
                __m128 p0, p1, p2, p3, p4, p5, p6, p7;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount+=4)
                {
                    px0 =  _mm_loadu_si128((__m128i *)srcPtrTemp);
                    px1 = _mm_unpackhi_epi8(px0, zero);    // pixels 8-15
                    px0 = _mm_unpacklo_epi8(px0, zero);    // pixels 0-7
                    p0 = _mm_cvtepi32_ps(_mm_unpacklo_epi16(px0, zero));    // pixels 0-3
                    p1 = _mm_cvtepi32_ps(_mm_unpackhi_epi16(px0, zero));    // pixels 4-7
                    p2 = _mm_cvtepi32_ps(_mm_unpacklo_epi16(px1, zero));    // pixels 8-11
                    p3 = _mm_cvtepi32_ps(_mm_unpackhi_epi16(px1, zero));    // pixels 12-15

                    p0 = _mm_mul_ps(pAlpha, p0);
                    p4 = _mm_mul_ps(_mm_sub_ps(pOne, pAlpha), pUserPixelPartialVector1);
                    p0 = _mm_add_ps(p0, p4);

                    p1 = _mm_mul_ps(pAlpha, p1);
                    p5 = _mm_mul_ps(_mm_sub_ps(pOne, pAlpha), pUserPixelPartialVector2);
                    p1 = _mm_add_ps(p1, p5);

                    p2 = _mm_mul_ps(pAlpha, p2);
                    p6 = _mm_mul_ps(_mm_sub_ps(pOne, pAlpha), pUserPixelPartialVector3);
                    p2 = _mm_add_ps(p2, p6);

                    px0 = _mm_cvtps_epi32(p0);
                    px1 = _mm_cvtps_epi32(p1);
                    px2 = _mm_cvtps_epi32(p2);
                    px3 = _mm_cvtps_epi32(pZero);

                    px0 = _mm_packus_epi32(px0, px1);
                    px1 = _mm_packus_epi32(px2, px3);

                    px0 = _mm_packus_epi16(px0, px1);    // pixels 0-15

                    _mm_storeu_si128((__m128i *)dstPtrTemp, px0);

                    srcPtrTemp += 12;
                    dstPtrTemp += 12;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    for (int c = 0; c < channel; c++)
                    {
                        *dstPtrTemp = (T) RPPPIXELCHECK((alpha * ((Rpp32f) *srcPtrTemp)) + ((1 - alpha) * userPixel[c]));
                        srcPtrTemp++;
                        dstPtrTemp++;
                    }
                }
            }
            if (outputFormatToggle == 1)
            {
                T *dstPtrImageUnpadded = (T*) calloc(channel * batch_srcSize[batchCount].height * batch_srcSize[batchCount].width, sizeof(T));
                T *dstPtrImageUnpaddedCopy = (T*) calloc(channel * batch_srcSize[batchCount].height * batch_srcSize[batchCount].width, sizeof(T));

                compute_unpadded_from_padded_host(dstPtrImage, batch_srcSize[batchCount], batch_srcSizeMax[batchCount], dstPtrImageUnpadded, chnFormat, channel);

                memcpy(dstPtrImageUnpaddedCopy, dstPtrImageUnpadded, channel * batch_srcSize[batchCount].height * batch_srcSize[batchCount].width * sizeof(T));

                compute_packed_to_planar_host(dstPtrImageUnpaddedCopy, batch_srcSize[batchCount], dstPtrImageUnpadded, channel);

                memset(dstPtrImage, (T) 0, imageDimMax * channel * sizeof(T));

                compute_padded_from_unpadded_host(dstPtrImageUnpadded, batch_srcSize[batchCount], batch_srcSizeMax[batchCount], dstPtrImage, RPPI_CHN_PLANAR, channel);

                free(dstPtrImageUnpadded);
                free(dstPtrImageUnpaddedCopy);
            }
        }
    }

    return RPP_SUCCESS;
}

RppStatus color_cast_f32_host_batch(Rpp32f* srcPtr, RppiSize *batch_srcSize, RppiSize *batch_srcSizeMax, Rpp32f* dstPtr,
                         Rpp8u *batch_r, Rpp8u *batch_g, Rpp8u *batch_b, Rpp32f *batch_alpha,
                         Rpp32u outputFormatToggle, Rpp32u nbatchSize,
                         RppiChnFormat chnFormat, Rpp32u channel)
{
    if(chnFormat == RPPI_CHN_PLANAR)
    {
        omp_set_dynamic(0);
#pragma omp parallel for num_threads(nbatchSize)
        for(int batchCount = 0; batchCount < nbatchSize; batchCount ++)
        {
            Rpp32u imageDimMax = batch_srcSizeMax[batchCount].height * batch_srcSizeMax[batchCount].width;

            Rpp8u r = batch_r[batchCount];
            Rpp8u g = batch_g[batchCount];
            Rpp8u b = batch_b[batchCount];
            Rpp32f alpha = batch_alpha[batchCount];

            Rpp32f userPixel[3] = {((Rpp32f) b) / 255, ((Rpp32f) g) / 255, ((Rpp32f) r) / 255};

            Rpp32f *srcPtrImage, *dstPtrImage;
            Rpp32u loc = 0;
            compute_image_location_host(batch_srcSizeMax, batchCount, &loc, channel);
            srcPtrImage = srcPtr + loc;
            dstPtrImage = dstPtr + loc;

            Rpp32u elementsInRowMax = batch_srcSizeMax[batchCount].width;

            for (int c = 0; c < channel; c++)
            {
                Rpp32f *srcPtrChannel, *dstPtrChannel;
                srcPtrChannel = srcPtrImage + (c * imageDimMax);
                dstPtrChannel = dstPtrImage + (c * imageDimMax);

                __m128 pOne = _mm_set1_ps(1.0);
                __m128 pUserPixel = _mm_set1_ps(userPixel[c]);
                __m128 pAlpha = _mm_set1_ps(alpha);

                for(int i = 0; i < batch_srcSize[batchCount].height; i++)
                {
                    Rpp32f *srcPtrTemp, *dstPtrTemp;
                    srcPtrTemp = srcPtrChannel + (i * elementsInRowMax);
                    dstPtrTemp = dstPtrChannel + (i * elementsInRowMax);

                    Rpp32u bufferLength = batch_srcSize[batchCount].width;
                    Rpp32u alignedLength = (bufferLength & ~3) - 4;

                    __m128i px0, px1, px2, px3;
                    __m128 p0, p1;

                    int vectorLoopCount = 0;
                    for (; vectorLoopCount < alignedLength; vectorLoopCount+=4)
                    {
                        p0 = _mm_loadu_ps(srcPtrTemp);

                        p0 = _mm_mul_ps(pAlpha, p0);
                        p1 = _mm_mul_ps(_mm_sub_ps(pOne, pAlpha), pUserPixel);
                        p0 = _mm_add_ps(p0, p1);

                        _mm_storeu_ps(dstPtrTemp, p0);

                        srcPtrTemp += 4;
                        dstPtrTemp += 4;
                    }
                    for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                    {
                        *dstPtrTemp = (Rpp32f) ((alpha * ((Rpp32f) *srcPtrTemp)) + ((1 - alpha) * userPixel[c]));

                        srcPtrTemp++;
                        dstPtrTemp++;
                    }
                }
            }
            if (outputFormatToggle == 1)
            {
                Rpp32f *dstPtrImageUnpadded = (Rpp32f*) calloc(channel * batch_srcSize[batchCount].height * batch_srcSize[batchCount].width, sizeof(Rpp32f));
                Rpp32f *dstPtrImageUnpaddedCopy = (Rpp32f*) calloc(channel * batch_srcSize[batchCount].height * batch_srcSize[batchCount].width, sizeof(Rpp32f));

                compute_unpadded_from_padded_host(dstPtrImage, batch_srcSize[batchCount], batch_srcSizeMax[batchCount], dstPtrImageUnpadded, chnFormat, channel);

                memcpy(dstPtrImageUnpaddedCopy, dstPtrImageUnpadded, channel * batch_srcSize[batchCount].height * batch_srcSize[batchCount].width * sizeof(Rpp32f));

                compute_planar_to_packed_host(dstPtrImageUnpaddedCopy, batch_srcSize[batchCount], dstPtrImageUnpadded, channel);

                memset(dstPtrImage, (Rpp32f) 0, imageDimMax * channel * sizeof(Rpp32f));

                compute_padded_from_unpadded_host(dstPtrImageUnpadded, batch_srcSize[batchCount], batch_srcSizeMax[batchCount], dstPtrImage, RPPI_CHN_PACKED, channel);

                free(dstPtrImageUnpadded);
                free(dstPtrImageUnpaddedCopy);
            }
        }
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        omp_set_dynamic(0);
#pragma omp parallel for num_threads(nbatchSize)
        for(int batchCount = 0; batchCount < nbatchSize; batchCount ++)
        {
            Rpp32u imageDimMax = batch_srcSizeMax[batchCount].height * batch_srcSizeMax[batchCount].width;

            Rpp8u r = batch_r[batchCount];
            Rpp8u g = batch_g[batchCount];
            Rpp8u b = batch_b[batchCount];
            Rpp32f alpha = batch_alpha[batchCount];

            Rpp32f userPixel[3] = {((Rpp32f) b) / 255, ((Rpp32f) g) / 255, ((Rpp32f) r) / 255};

            Rpp32f *srcPtrImage, *dstPtrImage;
            Rpp32u loc = 0;
            compute_image_location_host(batch_srcSizeMax, batchCount, &loc, channel);
            srcPtrImage = srcPtr + loc;
            dstPtrImage = dstPtr + loc;

            Rpp32u elementsInRowMax = channel * batch_srcSizeMax[batchCount].width;

            for(int i = 0; i < batch_srcSize[batchCount].height; i++)
            {
                Rpp32f *srcPtrTemp, *dstPtrTemp;
                srcPtrTemp = srcPtrImage + (i * elementsInRowMax);
                dstPtrTemp = dstPtrImage + (i * elementsInRowMax);

                Rpp32u bufferLength = batch_srcSize[batchCount].width;
                Rpp32u alignedLength = bufferLength & ~3;

                __m128 pOne = _mm_set1_ps(1.0);
                __m128 pAlpha = _mm_set1_ps(alpha);
                __m128 pUserPixelPartialVector1 = _mm_setr_ps(userPixel[0], userPixel[1], userPixel[2], userPixel[0]);
                __m128 pUserPixelPartialVector2 = _mm_setr_ps(userPixel[1], userPixel[2], userPixel[0], userPixel[1]);
                __m128 pUserPixelPartialVector3 = _mm_setr_ps(userPixel[2], userPixel[0], userPixel[1], userPixel[2]);
                __m128i px0, px1, px2, px3;
                __m128 p0, p1, p2, p4, p5, p6;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount+=4)
                {
                    p0 = _mm_loadu_ps(srcPtrTemp);
                    p1 = _mm_loadu_ps(srcPtrTemp + 4);
                    p2 = _mm_loadu_ps(srcPtrTemp + 8);

                    p0 = _mm_mul_ps(pAlpha, p0);
                    p4 = _mm_mul_ps(_mm_sub_ps(pOne, pAlpha), pUserPixelPartialVector1);
                    p0 = _mm_add_ps(p0, p4);

                    p1 = _mm_mul_ps(pAlpha, p1);
                    p5 = _mm_mul_ps(_mm_sub_ps(pOne, pAlpha), pUserPixelPartialVector2);
                    p1 = _mm_add_ps(p1, p5);

                    p2 = _mm_mul_ps(pAlpha, p2);
                    p6 = _mm_mul_ps(_mm_sub_ps(pOne, pAlpha), pUserPixelPartialVector3);
                    p2 = _mm_add_ps(p2, p6);

                    _mm_storeu_ps(dstPtrTemp, p0);
                    _mm_storeu_ps(dstPtrTemp + 4, p1);
                    _mm_storeu_ps(dstPtrTemp + 8, p2);

                    srcPtrTemp += 12;
                    dstPtrTemp += 12;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    for (int c = 0; c < channel; c++)
                    {
                        *dstPtrTemp = (Rpp32f) ((alpha * ((Rpp32f) *srcPtrTemp)) + ((1 - alpha) * userPixel[c]));
                        srcPtrTemp++;
                        dstPtrTemp++;
                    }
                }
            }
            if (outputFormatToggle == 1)
            {
                Rpp32f *dstPtrImageUnpadded = (Rpp32f*) calloc(channel * batch_srcSize[batchCount].height * batch_srcSize[batchCount].width, sizeof(Rpp32f));
                Rpp32f *dstPtrImageUnpaddedCopy = (Rpp32f*) calloc(channel * batch_srcSize[batchCount].height * batch_srcSize[batchCount].width, sizeof(Rpp32f));

                compute_unpadded_from_padded_host(dstPtrImage, batch_srcSize[batchCount], batch_srcSizeMax[batchCount], dstPtrImageUnpadded, chnFormat, channel);

                memcpy(dstPtrImageUnpaddedCopy, dstPtrImageUnpadded, channel * batch_srcSize[batchCount].height * batch_srcSize[batchCount].width * sizeof(Rpp32f));

                compute_packed_to_planar_host(dstPtrImageUnpaddedCopy, batch_srcSize[batchCount], dstPtrImageUnpadded, channel);

                memset(dstPtrImage, (Rpp32f) 0, imageDimMax * channel * sizeof(Rpp32f));

                compute_padded_from_unpadded_host(dstPtrImageUnpadded, batch_srcSize[batchCount], batch_srcSizeMax[batchCount], dstPtrImage, RPPI_CHN_PLANAR, channel);

                free(dstPtrImageUnpadded);
                free(dstPtrImageUnpaddedCopy);
            }
        }
    }

    return RPP_SUCCESS;
}

RppStatus color_cast_f16_host_batch(Rpp16f* srcPtr, RppiSize *batch_srcSize, RppiSize *batch_srcSizeMax, Rpp16f* dstPtr,
                         Rpp8u *batch_r, Rpp8u *batch_g, Rpp8u *batch_b, Rpp32f *batch_alpha,
                         Rpp32u outputFormatToggle, Rpp32u nbatchSize,
                         RppiChnFormat chnFormat, Rpp32u channel)
{
    if(chnFormat == RPPI_CHN_PLANAR)
    {
        omp_set_dynamic(0);
#pragma omp parallel for num_threads(nbatchSize)
        for(int batchCount = 0; batchCount < nbatchSize; batchCount ++)
        {
            Rpp32u imageDimMax = batch_srcSizeMax[batchCount].height * batch_srcSizeMax[batchCount].width;

            Rpp8u r = batch_r[batchCount];
            Rpp8u g = batch_g[batchCount];
            Rpp8u b = batch_b[batchCount];
            Rpp32f alpha = batch_alpha[batchCount];

            Rpp32f userPixel[3] = {((Rpp32f) b) / 255, ((Rpp32f) g) / 255, ((Rpp32f) r) / 255};

            Rpp16f *srcPtrImage, *dstPtrImage;
            Rpp32u loc = 0;
            compute_image_location_host(batch_srcSizeMax, batchCount, &loc, channel);
            srcPtrImage = srcPtr + loc;
            dstPtrImage = dstPtr + loc;

            Rpp32u elementsInRowMax = batch_srcSizeMax[batchCount].width;

            Rpp32f srcPtrTempps[4], dstPtrTempps[4];

            for (int c = 0; c < channel; c++)
            {
                Rpp16f *srcPtrChannel, *dstPtrChannel;
                srcPtrChannel = srcPtrImage + (c * imageDimMax);
                dstPtrChannel = dstPtrImage + (c * imageDimMax);

                __m128 pOne = _mm_set1_ps(1.0);
                __m128 pUserPixel = _mm_set1_ps(userPixel[c]);
                __m128 pAlpha = _mm_set1_ps(alpha);

                for(int i = 0; i < batch_srcSize[batchCount].height; i++)
                {
                    Rpp16f *srcPtrTemp, *dstPtrTemp;
                    srcPtrTemp = srcPtrChannel + (i * elementsInRowMax);
                    dstPtrTemp = dstPtrChannel + (i * elementsInRowMax);

                    Rpp32u bufferLength = batch_srcSize[batchCount].width;
                    Rpp32u alignedLength = (bufferLength & ~3) - 4;

                    __m128i px0, px1, px2, px3;
                    __m128 p0, p1;

                    int vectorLoopCount = 0;
                    for (; vectorLoopCount < alignedLength; vectorLoopCount+=4)
                    {
                        for(int cnt = 0; cnt < 4; cnt++)
                        {
                            *(srcPtrTempps + cnt) = (Rpp32f) (*(srcPtrTemp + cnt));
                        }

                        p0 = _mm_loadu_ps(srcPtrTempps);

                        p0 = _mm_mul_ps(pAlpha, p0);
                        p1 = _mm_mul_ps(_mm_sub_ps(pOne, pAlpha), pUserPixel);
                        p0 = _mm_add_ps(p0, p1);

                        _mm_storeu_ps(dstPtrTempps, p0);

                        for(int cnt = 0; cnt < 4; cnt++)
                        {
                            *(dstPtrTemp + cnt) = (Rpp16f) (*(dstPtrTempps + cnt));
                        }

                        srcPtrTemp += 4;
                        dstPtrTemp += 4;
                    }
                    for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                    {
                        *dstPtrTemp = (Rpp16f) ((alpha * ((Rpp32f) *srcPtrTemp)) + ((1 - alpha) * userPixel[c]));

                        srcPtrTemp++;
                        dstPtrTemp++;
                    }
                }
            }
            if (outputFormatToggle == 1)
            {
                Rpp16f *dstPtrImageUnpadded = (Rpp16f*) calloc(channel * batch_srcSize[batchCount].height * batch_srcSize[batchCount].width, sizeof(Rpp16f));
                Rpp16f *dstPtrImageUnpaddedCopy = (Rpp16f*) calloc(channel * batch_srcSize[batchCount].height * batch_srcSize[batchCount].width, sizeof(Rpp16f));

                compute_unpadded_from_padded_host(dstPtrImage, batch_srcSize[batchCount], batch_srcSizeMax[batchCount], dstPtrImageUnpadded, chnFormat, channel);

                memcpy(dstPtrImageUnpaddedCopy, dstPtrImageUnpadded, channel * batch_srcSize[batchCount].height * batch_srcSize[batchCount].width * sizeof(Rpp16f));

                compute_planar_to_packed_host(dstPtrImageUnpaddedCopy, batch_srcSize[batchCount], dstPtrImageUnpadded, channel);

                memset(dstPtrImage, (Rpp16f) 0, imageDimMax * channel * sizeof(Rpp16f));

                compute_padded_from_unpadded_host(dstPtrImageUnpadded, batch_srcSize[batchCount], batch_srcSizeMax[batchCount], dstPtrImage, RPPI_CHN_PACKED, channel);

                free(dstPtrImageUnpadded);
                free(dstPtrImageUnpaddedCopy);
            }
        }
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        omp_set_dynamic(0);
#pragma omp parallel for num_threads(nbatchSize)
        for(int batchCount = 0; batchCount < nbatchSize; batchCount ++)
        {
            Rpp32u imageDimMax = batch_srcSizeMax[batchCount].height * batch_srcSizeMax[batchCount].width;

            Rpp8u r = batch_r[batchCount];
            Rpp8u g = batch_g[batchCount];
            Rpp8u b = batch_b[batchCount];
            Rpp32f alpha = batch_alpha[batchCount];

            Rpp32f userPixel[3] = {((Rpp32f) b) / 255, ((Rpp32f) g) / 255, ((Rpp32f) r) / 255};

            Rpp16f *srcPtrImage, *dstPtrImage;
            Rpp32u loc = 0;
            compute_image_location_host(batch_srcSizeMax, batchCount, &loc, channel);
            srcPtrImage = srcPtr + loc;
            dstPtrImage = dstPtr + loc;

            Rpp32u elementsInRowMax = channel * batch_srcSizeMax[batchCount].width;

            Rpp32f srcPtrTempps[12], dstPtrTempps[12];

            for(int i = 0; i < batch_srcSize[batchCount].height; i++)
            {
                Rpp16f *srcPtrTemp, *dstPtrTemp;
                srcPtrTemp = srcPtrImage + (i * elementsInRowMax);
                dstPtrTemp = dstPtrImage + (i * elementsInRowMax);

                Rpp32u bufferLength = batch_srcSize[batchCount].width;
                Rpp32u alignedLength = bufferLength & ~3;

                __m128 pOne = _mm_set1_ps(1.0);
                __m128 pAlpha = _mm_set1_ps(alpha);
                __m128 pUserPixelPartialVector1 = _mm_setr_ps(userPixel[0], userPixel[1], userPixel[2], userPixel[0]);
                __m128 pUserPixelPartialVector2 = _mm_setr_ps(userPixel[1], userPixel[2], userPixel[0], userPixel[1]);
                __m128 pUserPixelPartialVector3 = _mm_setr_ps(userPixel[2], userPixel[0], userPixel[1], userPixel[2]);
                __m128i px0, px1, px2, px3;
                __m128 p0, p1, p2, p4, p5, p6;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount+=4)
                {
                    for(int cnt = 0; cnt < 12; cnt++)
                    {
                        *(srcPtrTempps + cnt) = (Rpp32f) (*(srcPtrTemp + cnt));
                    }

                    p0 = _mm_loadu_ps(srcPtrTempps);
                    p1 = _mm_loadu_ps(srcPtrTempps + 4);
                    p2 = _mm_loadu_ps(srcPtrTempps + 8);

                    p0 = _mm_mul_ps(pAlpha, p0);
                    p4 = _mm_mul_ps(_mm_sub_ps(pOne, pAlpha), pUserPixelPartialVector1);
                    p0 = _mm_add_ps(p0, p4);

                    p1 = _mm_mul_ps(pAlpha, p1);
                    p5 = _mm_mul_ps(_mm_sub_ps(pOne, pAlpha), pUserPixelPartialVector2);
                    p1 = _mm_add_ps(p1, p5);

                    p2 = _mm_mul_ps(pAlpha, p2);
                    p6 = _mm_mul_ps(_mm_sub_ps(pOne, pAlpha), pUserPixelPartialVector3);
                    p2 = _mm_add_ps(p2, p6);

                    _mm_storeu_ps(dstPtrTempps, p0);
                    _mm_storeu_ps(dstPtrTempps + 4, p1);
                    _mm_storeu_ps(dstPtrTempps + 8, p2);

                    for(int cnt = 0; cnt < 12; cnt++)
                    {
                        *(dstPtrTemp + cnt) = (Rpp16f) (*(dstPtrTempps + cnt));
                    }

                    srcPtrTemp += 12;
                    dstPtrTemp += 12;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    for (int c = 0; c < channel; c++)
                    {
                        *dstPtrTemp = (Rpp16f) ((alpha * ((Rpp32f) *srcPtrTemp)) + ((1 - alpha) * userPixel[c]));
                        srcPtrTemp++;
                        dstPtrTemp++;
                    }
                }
            }
            if (outputFormatToggle == 1)
            {
                Rpp16f *dstPtrImageUnpadded = (Rpp16f*) calloc(channel * batch_srcSize[batchCount].height * batch_srcSize[batchCount].width, sizeof(Rpp16f));
                Rpp16f *dstPtrImageUnpaddedCopy = (Rpp16f*) calloc(channel * batch_srcSize[batchCount].height * batch_srcSize[batchCount].width, sizeof(Rpp16f));

                compute_unpadded_from_padded_host(dstPtrImage, batch_srcSize[batchCount], batch_srcSizeMax[batchCount], dstPtrImageUnpadded, chnFormat, channel);

                memcpy(dstPtrImageUnpaddedCopy, dstPtrImageUnpadded, channel * batch_srcSize[batchCount].height * batch_srcSize[batchCount].width * sizeof(Rpp16f));

                compute_packed_to_planar_host(dstPtrImageUnpaddedCopy, batch_srcSize[batchCount], dstPtrImageUnpadded, channel);

                memset(dstPtrImage, (Rpp16f) 0, imageDimMax * channel * sizeof(Rpp16f));

                compute_padded_from_unpadded_host(dstPtrImageUnpadded, batch_srcSize[batchCount], batch_srcSizeMax[batchCount], dstPtrImage, RPPI_CHN_PLANAR, channel);

                free(dstPtrImageUnpadded);
                free(dstPtrImageUnpaddedCopy);
            }
        }
    }

    return RPP_SUCCESS;
}

RppStatus color_cast_i8_host_batch(Rpp8s* srcPtr, RppiSize *batch_srcSize, RppiSize *batch_srcSizeMax, Rpp8s* dstPtr,
                         Rpp8u *batch_r, Rpp8u *batch_g, Rpp8u *batch_b, Rpp32f *batch_alpha,
                         Rpp32u outputFormatToggle, Rpp32u nbatchSize,
                         RppiChnFormat chnFormat, Rpp32u channel)
{
    Rpp64u bufferLength = batch_srcSizeMax[0].height * batch_srcSizeMax[0].width * channel * nbatchSize;

    Rpp8u *srcPtr_8u = (Rpp8u*) calloc(bufferLength, sizeof(Rpp8u));
    Rpp8u *dstPtr_8u = (Rpp8u*) calloc(bufferLength, sizeof(Rpp8u));

    Rpp8s *srcPtrTemp;
    srcPtrTemp = srcPtr;

    Rpp8u *srcPtr_8uTemp;
    srcPtr_8uTemp = srcPtr_8u;

    for (int i = 0; i < bufferLength; i++)
    {
        *srcPtr_8uTemp = (Rpp8u) RPPPIXELCHECK(((Rpp32s) *srcPtrTemp) + 128);
        srcPtrTemp++;
        srcPtr_8uTemp++;
    }

    color_cast_host_batch<Rpp8u>(srcPtr_8u, batch_srcSize, batch_srcSizeMax, dstPtr_8u, batch_r, batch_g, batch_b, batch_alpha, outputFormatToggle, nbatchSize, chnFormat, channel);

    Rpp8s *dstPtrTemp;
    dstPtrTemp = dstPtr;

    Rpp8u *dstPtr_8uTemp;
    dstPtr_8uTemp = dstPtr_8u;

    for (int i = 0; i < bufferLength; i++)
    {
        *dstPtrTemp = (Rpp8s) (((Rpp32s) *dstPtr_8uTemp) - 128);
        dstPtrTemp++;
        dstPtr_8uTemp++;
    }

    free(srcPtr_8u);
    free(dstPtr_8u);

    return RPP_SUCCESS;
}

/**************** erase ***************/

template <typename T>
RppStatus erase_host_batch(T* srcPtr, RppiSize *batch_srcSize, RppiSize *batch_srcSizeMax, T* dstPtr,
                         Rpp32u *batch_anchor_box_info, T *batch_colors, Rpp32u *batch_box_offset, Rpp32u *batch_num_of_boxes,
                         Rpp32u outputFormatToggle, Rpp32u nbatchSize,
                         RppiChnFormat chnFormat, Rpp32u channel)
{
    if(chnFormat == RPPI_CHN_PLANAR)
    {
        omp_set_dynamic(0);
#pragma omp parallel for num_threads(nbatchSize)
        for(int batchCount = 0; batchCount < nbatchSize; batchCount ++)
        {
            Rpp32u imageDimMax = batch_srcSizeMax[batchCount].height * batch_srcSizeMax[batchCount].width;

            Rpp32u num_of_boxes = batch_num_of_boxes[batchCount];
            Rpp32u box_offset = batch_box_offset[batchCount];

            T *srcPtrImage, *dstPtrImage;
            Rpp32u loc = 0;
            compute_image_location_host(batch_srcSizeMax, batchCount, &loc, channel);
            srcPtrImage = srcPtr + loc;
            dstPtrImage = dstPtr + loc;

            memcpy(dstPtrImage, srcPtrImage, imageDimMax * channel * sizeof(T));

            Rpp32u elementsInRowMax = batch_srcSizeMax[batchCount].width;

            for(int count = 0; count < num_of_boxes; count++)
            {
                Rpp32u boxLoc = (box_offset * 4) + (count * 4);

                Rpp32u x1 = (Rpp32u) RPPPRANGECHECK(batch_anchor_box_info[boxLoc], 0, batch_srcSize[batchCount].width);
                Rpp32u y1 = (Rpp32u) RPPPRANGECHECK(batch_anchor_box_info[boxLoc + 1], 0, batch_srcSize[batchCount].height);
                Rpp32u x2 = (Rpp32u) RPPPRANGECHECK(batch_anchor_box_info[boxLoc + 2], 0, batch_srcSize[batchCount].width);
                Rpp32u y2 = (Rpp32u) RPPPRANGECHECK(batch_anchor_box_info[boxLoc + 3], 0, batch_srcSize[batchCount].height);

                Rpp32u locationInEachChannel = (y1 * elementsInRowMax) + x1;
                Rpp32u vtBufferLength = y2 - y1 + 1;
                Rpp32u hrBufferLength = x2 - x1 + 1;

                for (int c = 0; c < channel; c++)
                {
                    T userPixel = batch_colors[(box_offset * channel) + (count * channel) + c];

                    T *dstPtrTemp;
                    dstPtrTemp = dstPtrImage + (c * imageDimMax) + locationInEachChannel;

                    for (int i = 0; i < vtBufferLength; i++)
                    {
                        T * dstPtrTemp2;
                        dstPtrTemp2 = dstPtrTemp;

                        for(int j = 0; j < hrBufferLength; j++)
                        {
                            *dstPtrTemp2 = userPixel;
                            dstPtrTemp2++;
                        }

                        // memset(dstPtrTemp, userPixel, hrBufferLength * sizeof(T));
                        dstPtrTemp += elementsInRowMax;
                    }
                }
            }
            if (outputFormatToggle == 1)
            {
                T *dstPtrImageUnpadded = (T*) calloc(channel * batch_srcSize[batchCount].height * batch_srcSize[batchCount].width, sizeof(T));
                T *dstPtrImageUnpaddedCopy = (T*) calloc(channel * batch_srcSize[batchCount].height * batch_srcSize[batchCount].width, sizeof(T));

                compute_unpadded_from_padded_host(dstPtrImage, batch_srcSize[batchCount], batch_srcSizeMax[batchCount], dstPtrImageUnpadded, chnFormat, channel);

                memcpy(dstPtrImageUnpaddedCopy, dstPtrImageUnpadded, channel * batch_srcSize[batchCount].height * batch_srcSize[batchCount].width * sizeof(T));

                compute_planar_to_packed_host(dstPtrImageUnpaddedCopy, batch_srcSize[batchCount], dstPtrImageUnpadded, channel);

                memset(dstPtrImage, (T) 0, imageDimMax * channel * sizeof(T));

                compute_padded_from_unpadded_host(dstPtrImageUnpadded, batch_srcSize[batchCount], batch_srcSizeMax[batchCount], dstPtrImage, RPPI_CHN_PACKED, channel);

                free(dstPtrImageUnpadded);
                free(dstPtrImageUnpaddedCopy);
            }
        }
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        omp_set_dynamic(0);
#pragma omp parallel for num_threads(nbatchSize)
        for(int batchCount = 0; batchCount < nbatchSize; batchCount ++)
        {
            Rpp32u imageDimMax = batch_srcSizeMax[batchCount].height * batch_srcSizeMax[batchCount].width;

            Rpp32u num_of_boxes = batch_num_of_boxes[batchCount];
            Rpp32u box_offset = batch_box_offset[batchCount];

            T *srcPtrImage, *dstPtrImage;
            Rpp32u loc = 0;
            compute_image_location_host(batch_srcSizeMax, batchCount, &loc, channel);
            srcPtrImage = srcPtr + loc;
            dstPtrImage = dstPtr + loc;

            memcpy(dstPtrImage, srcPtrImage, imageDimMax * channel * sizeof(T));

            Rpp32u elementsInRowMax = channel * batch_srcSizeMax[batchCount].width;

            for(int count = 0; count < num_of_boxes; count++)
            {
                Rpp32u boxLoc = (box_offset * 4) + (count * 4);

                Rpp32u x1 = (Rpp32u) RPPPRANGECHECK(batch_anchor_box_info[boxLoc], 0, batch_srcSize[batchCount].width);
                Rpp32u y1 = (Rpp32u) RPPPRANGECHECK(batch_anchor_box_info[boxLoc + 1], 0, batch_srcSize[batchCount].height);
                Rpp32u x2 = (Rpp32u) RPPPRANGECHECK(batch_anchor_box_info[boxLoc + 2], 0, batch_srcSize[batchCount].width);
                Rpp32u y2 = (Rpp32u) RPPPRANGECHECK(batch_anchor_box_info[boxLoc + 3], 0, batch_srcSize[batchCount].height);

                Rpp32u locationInImage = (y1 * elementsInRowMax) + (x1 * channel);
                Rpp32u vtBufferLength = y2 - y1 + 1;
                Rpp32u hrBufferLength = x2 - x1 + 1;

                for (int c = 0; c < channel; c++)
                {
                    T userPixel = batch_colors[(box_offset * channel) + (count * channel) + c];

                    T *dstPtrTemp;
                    dstPtrTemp = dstPtrImage + locationInImage + c;

                    for (int i = 0; i < vtBufferLength; i++)
                    {
                        T * dstPtrTemp2;
                        dstPtrTemp2 = dstPtrTemp;

                        for(int j = 0; j < hrBufferLength; j++)
                        {
                            *dstPtrTemp2 = userPixel;
                            dstPtrTemp2 += channel;
                        }

                        // memset(dstPtrTemp, userPixel, hrBufferLength * sizeof(T));
                        dstPtrTemp += elementsInRowMax;
                    }
                }
            }
            if (outputFormatToggle == 1)
            {
                T *dstPtrImageUnpadded = (T*) calloc(channel * batch_srcSize[batchCount].height * batch_srcSize[batchCount].width, sizeof(T));
                T *dstPtrImageUnpaddedCopy = (T*) calloc(channel * batch_srcSize[batchCount].height * batch_srcSize[batchCount].width, sizeof(T));

                compute_unpadded_from_padded_host(dstPtrImage, batch_srcSize[batchCount], batch_srcSizeMax[batchCount], dstPtrImageUnpadded, chnFormat, channel);

                memcpy(dstPtrImageUnpaddedCopy, dstPtrImageUnpadded, channel * batch_srcSize[batchCount].height * batch_srcSize[batchCount].width * sizeof(T));

                compute_packed_to_planar_host(dstPtrImageUnpaddedCopy, batch_srcSize[batchCount], dstPtrImageUnpadded, channel);

                memset(dstPtrImage, (T) 0, imageDimMax * channel * sizeof(T));

                compute_padded_from_unpadded_host(dstPtrImageUnpadded, batch_srcSize[batchCount], batch_srcSizeMax[batchCount], dstPtrImage, RPPI_CHN_PLANAR, channel);

                free(dstPtrImageUnpadded);
                free(dstPtrImageUnpaddedCopy);
            }
        }
    }

    return RPP_SUCCESS;
}

template <typename T>
RppStatus crop_and_patch_host_batch(T* srcPtr1, RppiSize *batch_srcSize1, RppiSize *batch_srcSizeMax1,
                                    T* srcPtr2, RppiSize *batch_srcSize2, RppiSize *batch_srcSizeMax2, T* dstPtr,
                                    Rpp32u *batch_src1x1, Rpp32u *batch_src1y1, Rpp32u *batch_src1x2, Rpp32u *batch_src1y2,
                                    Rpp32u *batch_src2x1, Rpp32u *batch_src2y1, Rpp32u *batch_src2x2, Rpp32u *batch_src2y2,
                                    Rpp32u outputFormatToggle, Rpp32u nbatchSize,
                                    RppiChnFormat chnFormat, Rpp32u channel)
{
    if(chnFormat == RPPI_CHN_PLANAR)
    {
        omp_set_dynamic(0);
#pragma omp parallel for num_threads(nbatchSize)
        for(int batchCount = 0; batchCount < nbatchSize; batchCount ++)
        {
            Rpp32u dstImageDimMax = batch_srcSizeMax1[batchCount].height * batch_srcSizeMax1[batchCount].width;
            Rpp32u src2ImageDimMax = batch_srcSizeMax2[batchCount].height * batch_srcSizeMax2[batchCount].width;

            Rpp32u src1x1 = batch_src1x1[batchCount];
            Rpp32u src1y1 = batch_src1y1[batchCount];
            Rpp32u src1x2 = batch_src1x2[batchCount];
            Rpp32u src1y2 = batch_src1y2[batchCount];
            Rpp32u src2x1 = batch_src2x1[batchCount];
            Rpp32u src2y1 = batch_src2y1[batchCount];
            Rpp32u src2x2 = batch_src2x2[batchCount];
            Rpp32u src2y2 = batch_src2y2[batchCount];

            T *src1PtrImage, *src2PtrImage, *dstPtrImage;
            Rpp32u loc1 = 0;
            Rpp32u loc2 = 0;
            compute_image_location_host(batch_srcSizeMax1, batchCount, &loc1, channel);
            compute_image_location_host(batch_srcSizeMax2, batchCount, &loc2, channel);
            src1PtrImage = srcPtr1 + loc1;
            dstPtrImage = dstPtr + loc1;
            src2PtrImage = srcPtr2 + loc2;

            Rpp32u elementsInRow1 = batch_srcSize1[batchCount].width;
            Rpp32u elementsInRowMax1 = batch_srcSizeMax1[batchCount].width;

            Rpp32u elementsInRow2 = batch_srcSize2[batchCount].width;
            Rpp32u elementsInRowMax2 = batch_srcSizeMax2[batchCount].width;

            for (int c = 0; c < channel; c++)
            {
                T *srcPtrChannel, *dstPtrChannel;
                srcPtrChannel = src1PtrImage + (c * dstImageDimMax);
                dstPtrChannel = dstPtrImage + (c * dstImageDimMax);

                for(int i = 0; i < batch_srcSize1[batchCount].height; i++)
                {
                    T *srcPtrTemp, *dstPtrTemp;
                    srcPtrTemp = srcPtrChannel + (i * elementsInRowMax1);
                    dstPtrTemp = dstPtrChannel + (i * elementsInRowMax1);
                    memcpy(dstPtrTemp, srcPtrTemp, elementsInRow1 * sizeof(T));
                }
            }

            RppiSize srcSize2SubImage, dstSizeSubImage;
            T *srcPtr2SubImage, *dstPtrSubImage;
            srcSize2SubImage.height = RPPABS(src2y2 - src2y1) + 1;
            srcSize2SubImage.width = RPPABS(src2x2 - src2x1) + 1;
            srcPtr2SubImage = src2PtrImage + (src2y1 * elementsInRowMax2) + (src2x1);
            dstSizeSubImage.height = RPPABS(src1y2 - src1y1) + 1;
            dstSizeSubImage.width = RPPABS(src1x2 - src1x1) + 1;
            dstPtrSubImage = dstPtrImage + (src1y1 * elementsInRowMax1) + (src1x1);

            Rpp32f hRatio = (((Rpp32f) (dstSizeSubImage.height - 1)) / ((Rpp32f) (srcSize2SubImage.height - 1)));
            Rpp32f wRatio = (((Rpp32f) (dstSizeSubImage.width - 1)) / ((Rpp32f) (srcSize2SubImage.width - 1)));
            Rpp32f srcLocationRow, srcLocationColumn, pixel;
            Rpp32s srcLocationRowFloor, srcLocationColumnFloor;
            T *srcPtrTemp, *dstPtrTemp, *srcPtrTopRow, *srcPtrBottomRow;
            Rpp32u remainingElementsInRowDst = (batch_srcSizeMax1[batchCount].width - dstSizeSubImage.width);
            for (int c = 0; c < channel; c++)
            {
                srcPtrTemp = srcPtr2SubImage + (c * src2ImageDimMax);
                dstPtrTemp = dstPtrSubImage + (c * dstImageDimMax);

                for (int i = 0; i < dstSizeSubImage.height; i++)
                {
                    srcLocationRow = ((Rpp32f) i) / hRatio;
                    srcLocationRowFloor = (Rpp32s) RPPFLOOR(srcLocationRow);
                    Rpp32f weightedHeight = srcLocationRow - srcLocationRowFloor;

                    if (srcLocationRowFloor > (srcSize2SubImage.height - 2))
                    {
                        srcLocationRowFloor = srcSize2SubImage.height - 2;
                    }

                    srcPtrTopRow = srcPtrTemp + srcLocationRowFloor * elementsInRowMax1;
                    srcPtrBottomRow  = srcPtrTopRow + elementsInRowMax1;

                    for (int j = 0; j < dstSizeSubImage.width; j++)
                    {
                        srcLocationColumn = ((Rpp32f) j) / wRatio;
                        srcLocationColumnFloor = (Rpp32s) RPPFLOOR(srcLocationColumn);
                        Rpp32f weightedWidth = srcLocationColumn - srcLocationColumnFloor;

                        if (srcLocationColumnFloor > (srcSize2SubImage.width - 2))
                        {
                            srcLocationColumnFloor = srcSize2SubImage.width - 2;
                        }

                        pixel = ((*(srcPtrTopRow + srcLocationColumnFloor)) * (1 - weightedHeight) * (1 - weightedWidth))
                                + ((*(srcPtrTopRow + srcLocationColumnFloor + 1)) * (1 - weightedHeight) * (weightedWidth))
                                + ((*(srcPtrBottomRow + srcLocationColumnFloor)) * (weightedHeight) * (1 - weightedWidth))
                                + ((*(srcPtrBottomRow + srcLocationColumnFloor + 1)) * (weightedHeight) * (weightedWidth));

                        *dstPtrTemp = (T) pixel;
                        dstPtrTemp ++;
                    }
                    dstPtrTemp = dstPtrTemp + remainingElementsInRowDst;
                }
            }
            if (outputFormatToggle == 1)
            {
                T *dstPtrImageUnpadded = (T*) calloc(channel * batch_srcSize1[batchCount].height * batch_srcSize1[batchCount].width, sizeof(T));
                T *dstPtrImageUnpaddedCopy = (T*) calloc(channel * batch_srcSize1[batchCount].height * batch_srcSize1[batchCount].width, sizeof(T));

                compute_unpadded_from_padded_host(dstPtrImage, batch_srcSize1[batchCount], batch_srcSizeMax1[batchCount], dstPtrImageUnpadded, chnFormat, channel);

                memcpy(dstPtrImageUnpaddedCopy, dstPtrImageUnpadded, channel * batch_srcSize1[batchCount].height * batch_srcSize1[batchCount].width * sizeof(T));

                compute_planar_to_packed_host(dstPtrImageUnpaddedCopy, batch_srcSize1[batchCount], dstPtrImageUnpadded, channel);

                memset(dstPtrImage, (T) 0, dstImageDimMax * channel * sizeof(T));

                compute_padded_from_unpadded_host(dstPtrImageUnpadded, batch_srcSize1[batchCount], batch_srcSizeMax1[batchCount], dstPtrImage, RPPI_CHN_PACKED, channel);

                free(dstPtrImageUnpadded);
                free(dstPtrImageUnpaddedCopy);
            }
        }
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        omp_set_dynamic(0);
#pragma omp parallel for num_threads(nbatchSize)
        for(int batchCount = 0; batchCount < nbatchSize; batchCount ++)
        {
            Rpp32u dstImageDimMax = batch_srcSizeMax1[batchCount].height * batch_srcSizeMax1[batchCount].width;

            Rpp32u src1x1 = batch_src1x1[batchCount];
            Rpp32u src1y1 = batch_src1y1[batchCount];
            Rpp32u src1x2 = batch_src1x2[batchCount];
            Rpp32u src1y2 = batch_src1y2[batchCount];
            Rpp32u src2x1 = batch_src2x1[batchCount];
            Rpp32u src2y1 = batch_src2y1[batchCount];
            Rpp32u src2x2 = batch_src2x2[batchCount];
            Rpp32u src2y2 = batch_src2y2[batchCount];

            T *src1PtrImage, *src2PtrImage, *dstPtrImage;
            Rpp32u loc1 = 0;
            Rpp32u loc2 = 0;
            compute_image_location_host(batch_srcSizeMax1, batchCount, &loc1, channel);
            compute_image_location_host(batch_srcSizeMax2, batchCount, &loc2, channel);
            src1PtrImage = srcPtr1 + loc1;
            dstPtrImage = dstPtr + loc1;
            src2PtrImage = srcPtr2 + loc2;

            Rpp32u elementsInRow1 = channel * batch_srcSize1[batchCount].width;
            Rpp32u elementsInRowMax1 = channel * batch_srcSizeMax1[batchCount].width;

            Rpp32u elementsInRow2 = channel * batch_srcSize2[batchCount].width;
            Rpp32u elementsInRowMax2 = channel * batch_srcSizeMax2[batchCount].width;

            for(int i = 0; i < batch_srcSize1[batchCount].height; i++)
            {
                T *srcPtrTemp, *dstPtrTemp;
                srcPtrTemp = src1PtrImage + (i * elementsInRowMax1);
                dstPtrTemp = dstPtrImage + (i * elementsInRowMax1);
                memcpy(dstPtrTemp, srcPtrTemp, elementsInRow1 * sizeof(T));
            }

            RppiSize srcSize2SubImage, dstSizeSubImage;
            T *srcPtr2SubImage, *dstPtrSubImage;
            srcSize2SubImage.height = RPPABS(src2y2 - src2y1) + 1;
            srcSize2SubImage.width = RPPABS(src2x2 - src2x1) + 1;
            srcPtr2SubImage = src2PtrImage + (src2y1 * elementsInRowMax2) + (src2x1 * channel);
            dstSizeSubImage.height = RPPABS(src1y2 - src1y1) + 1;
            dstSizeSubImage.width = RPPABS(src1x2 - src1x1) + 1;
            dstPtrSubImage = dstPtrImage + (src1y1 * elementsInRowMax1) + (src1x1 * channel);

            Rpp32f hRatio = (((Rpp32f) (dstSizeSubImage.height - 1)) / ((Rpp32f) (srcSize2SubImage.height - 1)));
            Rpp32f wRatio = (((Rpp32f) (dstSizeSubImage.width - 1)) / ((Rpp32f) (srcSize2SubImage.width - 1)));
            Rpp32f srcLocationRow, srcLocationColumn, pixel;
            Rpp32s srcLocationRowFloor, srcLocationColumnFloor;
            T *srcPtrTemp, *dstPtrTemp;
            srcPtrTemp = srcPtr2SubImage;
            dstPtrTemp = dstPtrSubImage;
            Rpp32u remainingElementsInRowDst = (batch_srcSizeMax1[batchCount].width - dstSizeSubImage.width) * channel;
            for (int i = 0; i < dstSizeSubImage.height; i++)
            {
                srcLocationRow = ((Rpp32f) i) / hRatio;
                srcLocationRowFloor = (Rpp32s) RPPFLOOR(srcLocationRow);
                Rpp32f weightedHeight = srcLocationRow - srcLocationRowFloor;

                if (srcLocationRowFloor > (srcSize2SubImage.height - 2))
                {
                    srcLocationRowFloor = srcSize2SubImage.height - 2;
                }

                T *srcPtrTopRow, *srcPtrBottomRow;
                srcPtrTopRow = srcPtrTemp + srcLocationRowFloor * elementsInRowMax1;
                srcPtrBottomRow  = srcPtrTopRow + elementsInRowMax1;

                for (int j = 0; j < dstSizeSubImage.width; j++)
                {
                    srcLocationColumn = ((Rpp32f) j) / wRatio;
                    srcLocationColumnFloor = (Rpp32s) RPPFLOOR(srcLocationColumn);
                    Rpp32f weightedWidth = srcLocationColumn - srcLocationColumnFloor;

                    if (srcLocationColumnFloor > (srcSize2SubImage.width - 2))
                    {
                        srcLocationColumnFloor = srcSize2SubImage.width - 2;
                    }

                    Rpp32s srcLocColFloorChanneled = channel * srcLocationColumnFloor;

                    for (int c = 0; c < channel; c++)
                    {
                        pixel = ((*(srcPtrTopRow + c + srcLocColFloorChanneled)) * (1 - weightedHeight) * (1 - weightedWidth))
                                + ((*(srcPtrTopRow + c + srcLocColFloorChanneled + channel)) * (1 - weightedHeight) * (weightedWidth))
                                + ((*(srcPtrBottomRow + c + srcLocColFloorChanneled)) * (weightedHeight) * (1 - weightedWidth))
                                + ((*(srcPtrBottomRow + c + srcLocColFloorChanneled + channel)) * (weightedHeight) * (weightedWidth));

                        *dstPtrTemp = (T) pixel;
                        dstPtrTemp ++;
                    }
                }
                dstPtrTemp = dstPtrTemp + remainingElementsInRowDst;
            }
            if (outputFormatToggle == 1)
            {
                T *dstPtrImageUnpadded = (T*) calloc(channel * batch_srcSize1[batchCount].height * batch_srcSize1[batchCount].width, sizeof(T));
                T *dstPtrImageUnpaddedCopy = (T*) calloc(channel * batch_srcSize1[batchCount].height * batch_srcSize1[batchCount].width, sizeof(T));

                compute_unpadded_from_padded_host(dstPtrImage, batch_srcSize1[batchCount], batch_srcSizeMax1[batchCount], dstPtrImageUnpadded, chnFormat, channel);

                memcpy(dstPtrImageUnpaddedCopy, dstPtrImageUnpadded, channel * batch_srcSize1[batchCount].height * batch_srcSize1[batchCount].width * sizeof(T));

                compute_packed_to_planar_host(dstPtrImageUnpaddedCopy, batch_srcSize1[batchCount], dstPtrImageUnpadded, channel);

                memset(dstPtrImage, (T) 0, dstImageDimMax * channel * sizeof(T));

                compute_padded_from_unpadded_host(dstPtrImageUnpadded, batch_srcSize1[batchCount], batch_srcSizeMax1[batchCount], dstPtrImage, RPPI_CHN_PLANAR, channel);

                free(dstPtrImageUnpadded);
                free(dstPtrImageUnpaddedCopy);
            }
        }
    }

    return RPP_SUCCESS;
}

template <typename T>
RppStatus lut_host_batch(T* srcPtr, RppiSize *batch_srcSize, RppiSize *batch_srcSizeMax, T* dstPtr,
                        T *batch_lutPtr,
                        Rpp32u outputFormatToggle, Rpp32u nbatchSize,
                        RppiChnFormat chnFormat, Rpp32u channel)
{
    Rpp32u lutSize = 256;

    if(chnFormat == RPPI_CHN_PLANAR)
    {
        omp_set_dynamic(0);
#pragma omp parallel for num_threads(nbatchSize)
        for(int batchCount = 0; batchCount < nbatchSize; batchCount ++)
        {
            Rpp32u imageDimMax = batch_srcSizeMax[batchCount].height * batch_srcSizeMax[batchCount].width;

            T* lutPtr = (T*) calloc(lutSize, sizeof(T));
            memcpy(lutPtr, (batch_lutPtr + (batchCount * lutSize)), lutSize * sizeof(T));

            T *srcPtrImage, *dstPtrImage;
            Rpp32u loc = 0;
            compute_image_location_host(batch_srcSizeMax, batchCount, &loc, channel);
            srcPtrImage = srcPtr + loc;
            dstPtrImage = dstPtr + loc;

            for(int c = 0; c < channel; c++)
            {
                T *srcPtrChannel, *dstPtrChannel;
                srcPtrChannel = srcPtrImage + (c * imageDimMax);
                dstPtrChannel = dstPtrImage + (c * imageDimMax);

                for(int i = 0; i < batch_srcSize[batchCount].height; i++)
                {
                    T *srcPtrTemp, *dstPtrTemp;
                    srcPtrTemp = srcPtrChannel + (i * batch_srcSizeMax[batchCount].width);
                    dstPtrTemp = dstPtrChannel + (i * batch_srcSizeMax[batchCount].width);

                    if (typeid(Rpp8u) == typeid(T))
                    {
                        for(int j = 0; j < batch_srcSize[batchCount].width; j++)
                        {
                            *dstPtrTemp = *(lutPtr + (Rpp32s)(*srcPtrTemp));

                            srcPtrTemp++;
                            dstPtrTemp++;
                        }
                    }
                    else if (typeid(Rpp8s) == typeid(T))
                    {
                        for(int j = 0; j < batch_srcSize[batchCount].width; j++)
                        {
                            *dstPtrTemp = *(lutPtr + (((Rpp32s) (*srcPtrTemp)) + 128));

                            srcPtrTemp++;
                            dstPtrTemp++;
                        }
                    }
                }
            }
            free(lutPtr);
            if (outputFormatToggle == 1)
            {
                T *dstPtrImageUnpadded = (T*) calloc(channel * batch_srcSize[batchCount].height * batch_srcSize[batchCount].width, sizeof(T));
                T *dstPtrImageUnpaddedCopy = (T*) calloc(channel * batch_srcSize[batchCount].height * batch_srcSize[batchCount].width, sizeof(T));

                compute_unpadded_from_padded_host(dstPtrImage, batch_srcSize[batchCount], batch_srcSizeMax[batchCount], dstPtrImageUnpadded, chnFormat, channel);

                memcpy(dstPtrImageUnpaddedCopy, dstPtrImageUnpadded, channel * batch_srcSize[batchCount].height * batch_srcSize[batchCount].width * sizeof(T));

                compute_planar_to_packed_host(dstPtrImageUnpaddedCopy, batch_srcSize[batchCount], dstPtrImageUnpadded, channel);

                memset(dstPtrImage, (T) 0, imageDimMax * channel * sizeof(T));

                compute_padded_from_unpadded_host(dstPtrImageUnpadded, batch_srcSize[batchCount], batch_srcSizeMax[batchCount], dstPtrImage, RPPI_CHN_PACKED, channel);

                free(dstPtrImageUnpadded);
                free(dstPtrImageUnpaddedCopy);
            }
        }
    }
    else if(chnFormat == RPPI_CHN_PACKED)
    {
        omp_set_dynamic(0);
#pragma omp parallel for num_threads(nbatchSize)
        for(int batchCount = 0; batchCount < nbatchSize; batchCount ++)
        {
            Rpp32u imageDimMax = batch_srcSizeMax[batchCount].height * batch_srcSizeMax[batchCount].width;

            T* lutPtr = (T*) calloc(lutSize, sizeof(T));
            memcpy(lutPtr, (batch_lutPtr + (batchCount * lutSize)), lutSize * sizeof(T));

            T *srcPtrImage, *dstPtrImage;
            Rpp32u loc = 0;
            compute_image_location_host(batch_srcSizeMax, batchCount, &loc, channel);
            srcPtrImage = srcPtr + loc;
            dstPtrImage = dstPtr + loc;

            Rpp32u elementsInRowMax = channel * batch_srcSizeMax[batchCount].width;
            Rpp32u elementsInRow = channel * batch_srcSize[batchCount].width;

            if (typeid(Rpp8u) == typeid(T))
            {
                for(int i = 0; i < batch_srcSize[batchCount].height; i++)
                {
                    T *srcPtrTemp, *dstPtrTemp;
                    srcPtrTemp = srcPtrImage + (i * elementsInRowMax);
                    dstPtrTemp = dstPtrImage + (i * elementsInRowMax);

                    for(int j = 0; j < batch_srcSize[batchCount].width; j++)
                    {
                        for(int c = 0; c < channel; c++)
                        {
                            *dstPtrTemp = *(lutPtr + (Rpp32s) *srcPtrTemp);

                            srcPtrTemp++;
                            dstPtrTemp++;
                        }
                    }

                    srcPtrTemp += (elementsInRowMax - elementsInRow);
                    dstPtrTemp += (elementsInRowMax - elementsInRow);
                }
            }
            else if (typeid(Rpp8s) == typeid(T))
            {
                for(int i = 0; i < batch_srcSize[batchCount].height; i++)
                {
                    T *srcPtrTemp, *dstPtrTemp;
                    srcPtrTemp = srcPtrImage + (i * elementsInRowMax);
                    dstPtrTemp = dstPtrImage + (i * elementsInRowMax);

                    for(int j = 0; j < batch_srcSize[batchCount].width; j++)
                    {
                        for(int c = 0; c < channel; c++)
                        {
                            *dstPtrTemp = *(lutPtr + (Rpp32s) *srcPtrTemp + 128);

                            srcPtrTemp++;
                            dstPtrTemp++;
                        }
                    }

                    srcPtrTemp += (elementsInRowMax - elementsInRow);
                    dstPtrTemp += (elementsInRowMax - elementsInRow);
                }
            }
            free(lutPtr);
            if (outputFormatToggle == 1)
            {
                T *dstPtrImageUnpadded = (T*) calloc(channel * batch_srcSize[batchCount].height * batch_srcSize[batchCount].width, sizeof(T));
                T *dstPtrImageUnpaddedCopy = (T*) calloc(channel * batch_srcSize[batchCount].height * batch_srcSize[batchCount].width, sizeof(T));

                compute_unpadded_from_padded_host(dstPtrImage, batch_srcSize[batchCount], batch_srcSizeMax[batchCount], dstPtrImageUnpadded, chnFormat, channel);

                memcpy(dstPtrImageUnpaddedCopy, dstPtrImageUnpadded, channel * batch_srcSize[batchCount].height * batch_srcSize[batchCount].width * sizeof(T));

                compute_packed_to_planar_host(dstPtrImageUnpaddedCopy, batch_srcSize[batchCount], dstPtrImageUnpadded, channel);

                memset(dstPtrImage, (T) 0, imageDimMax * channel * sizeof(T));

                compute_padded_from_unpadded_host(dstPtrImageUnpadded, batch_srcSize[batchCount], batch_srcSizeMax[batchCount], dstPtrImage, RPPI_CHN_PLANAR, channel);

                free(dstPtrImageUnpadded);
                free(dstPtrImageUnpaddedCopy);
            }
        }
    }

    return RPP_SUCCESS;
}

/**************** glitch ***************/

template <typename T>
RppStatus glitch_host_batch(T* srcPtr, RppiSize *batch_srcSize, RppiSize *batch_srcSizeMax, T* dstPtr,
                         Rpp32u *batch_x_offset_r, Rpp32u *batch_y_offset_r,
                         Rpp32u *batch_x_offset_g, Rpp32u *batch_y_offset_g,
                         Rpp32u *batch_x_offset_b, Rpp32u *batch_y_offset_b,
                         Rpp32u outputFormatToggle, Rpp32u nbatchSize,
                         RppiChnFormat chnFormat, Rpp32u channel)
{
    if(chnFormat == RPPI_CHN_PLANAR)
    {
        omp_set_dynamic(0);
#pragma omp parallel for num_threads(nbatchSize)
        for(int batchCount = 0; batchCount < nbatchSize; batchCount ++)
        {
            Rpp32u imageDimMax = batch_srcSizeMax[batchCount].height * batch_srcSizeMax[batchCount].width;

            Rpp32u x_offset_r = batch_x_offset_r[batchCount];
            Rpp32u y_offset_r = batch_y_offset_r[batchCount];
            Rpp32u x_offset_g = batch_x_offset_g[batchCount];
            Rpp32u y_offset_g = batch_y_offset_g[batchCount];
            Rpp32u x_offset_b = batch_x_offset_b[batchCount];
            Rpp32u y_offset_b = batch_y_offset_b[batchCount];

            Rpp32u elementsInRowMax = batch_srcSizeMax[batchCount].width;
            Rpp32u elementsInRow = batch_srcSize[batchCount].width;
            Rpp32u increment = elementsInRowMax - elementsInRow;

            Rpp32u xOffsets[3] = {
                x_offset_r,
                x_offset_g,
                x_offset_b
            };

            Rpp32u yOffsets[3] = {
                y_offset_r,
                y_offset_g,
                y_offset_b
            };

            Rpp32u xOffsetsLoc[3] = {
                x_offset_r,
                x_offset_g,
                x_offset_b
            };

            Rpp32u yOffsetsLoc[3] = {
                y_offset_r * elementsInRowMax,
                y_offset_g * elementsInRowMax,
                y_offset_b * elementsInRowMax
            };

            T *srcPtrImage, *dstPtrImage;
            Rpp32u loc = 0;
            compute_image_location_host(batch_srcSizeMax, batchCount, &loc, channel);
            srcPtrImage = srcPtr + loc;
            dstPtrImage = dstPtr + loc;

            memcpy(dstPtrImage, srcPtrImage, imageDimMax * channel * sizeof(T));

            Rpp32u currentRow, currentCol;

            for (int c = 0; c < channel; c++)
            {
                T *srcPtrImageTemp, *dstPtrImageTemp;
                srcPtrImageTemp = srcPtrImage + (c * imageDimMax) + yOffsetsLoc[c] + xOffsetsLoc[c];
                dstPtrImageTemp = dstPtrImage + (c * imageDimMax);

                currentRow = yOffsets[c];
                for (int i = 0; i < batch_srcSize[batchCount].height; i++)
                {
                    currentCol = xOffsets[c];
                    for (int j = 0; j < batch_srcSize[batchCount].width; j++)
                    {
                        if (
                            ((currentRow >= 0) && (currentRow < batch_srcSize[batchCount].height)) &&
                            ((currentCol >= 0) && (currentCol < batch_srcSize[batchCount].width))
                        )
                        {
                            *dstPtrImageTemp = *srcPtrImageTemp;
                        }
                        dstPtrImageTemp++;
                        srcPtrImageTemp++;
                        currentCol++;
                    }
                    dstPtrImageTemp += increment;
                    srcPtrImageTemp += increment;
                    currentRow++;
                }
            }
            if (outputFormatToggle == 1)
            {
                T *dstPtrImageUnpadded = (T*) calloc(channel * batch_srcSize[batchCount].height * batch_srcSize[batchCount].width, sizeof(T));
                T *dstPtrImageUnpaddedCopy = (T*) calloc(channel * batch_srcSize[batchCount].height * batch_srcSize[batchCount].width, sizeof(T));

                compute_unpadded_from_padded_host(dstPtrImage, batch_srcSize[batchCount], batch_srcSizeMax[batchCount], dstPtrImageUnpadded, chnFormat, channel);

                memcpy(dstPtrImageUnpaddedCopy, dstPtrImageUnpadded, channel * batch_srcSize[batchCount].height * batch_srcSize[batchCount].width * sizeof(T));

                compute_planar_to_packed_host(dstPtrImageUnpaddedCopy, batch_srcSize[batchCount], dstPtrImageUnpadded, channel);

                memset(dstPtrImage, (T) 0, imageDimMax * channel * sizeof(T));

                compute_padded_from_unpadded_host(dstPtrImageUnpadded, batch_srcSize[batchCount], batch_srcSizeMax[batchCount], dstPtrImage, RPPI_CHN_PACKED, channel);

                free(dstPtrImageUnpadded);
                free(dstPtrImageUnpaddedCopy);
            }
        }
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        omp_set_dynamic(0);
#pragma omp parallel for num_threads(nbatchSize)
        for(int batchCount = 0; batchCount < nbatchSize; batchCount ++)
        {
            Rpp32u imageDimMax = batch_srcSizeMax[batchCount].height * batch_srcSizeMax[batchCount].width;

            Rpp32u x_offset_r = batch_x_offset_r[batchCount];
            Rpp32u y_offset_r = batch_y_offset_r[batchCount];
            Rpp32u x_offset_g = batch_x_offset_g[batchCount];
            Rpp32u y_offset_g = batch_y_offset_g[batchCount];
            Rpp32u x_offset_b = batch_x_offset_b[batchCount];
            Rpp32u y_offset_b = batch_y_offset_b[batchCount];

            Rpp32u elementsInRowMax = channel * batch_srcSizeMax[batchCount].width;
            Rpp32u elementsInRow = channel * batch_srcSize[batchCount].width;
            Rpp32u increment = elementsInRowMax - elementsInRow;

            Rpp32u xOffsets[3] = {
                x_offset_r,
                x_offset_g,
                x_offset_b
            };

            Rpp32u yOffsets[3] = {
                y_offset_r,
                y_offset_g,
                y_offset_b
            };

            Rpp32u xOffsetsLoc[3] = {
                x_offset_r * channel,
                x_offset_g * channel,
                x_offset_b * channel
            };

            Rpp32u yOffsetsLoc[3] = {
                y_offset_r * elementsInRowMax,
                y_offset_g * elementsInRowMax,
                y_offset_b * elementsInRowMax
            };

            T *srcPtrImage, *dstPtrImage;
            Rpp32u loc = 0;
            compute_image_location_host(batch_srcSizeMax, batchCount, &loc, channel);
            srcPtrImage = srcPtr + loc;
            dstPtrImage = dstPtr + loc;

            memcpy(dstPtrImage, srcPtrImage, imageDimMax * channel * sizeof(T));

            Rpp32u currentRow, currentCol;

            for (int c = 0; c < channel; c++)
            {
                T *srcPtrImageTemp, *dstPtrImageTemp;
                srcPtrImageTemp = srcPtrImage + c + yOffsetsLoc[c] + xOffsetsLoc[c];
                dstPtrImageTemp = dstPtrImage + c;

                currentRow = yOffsets[c];
                for (int i = 0; i < batch_srcSize[batchCount].height; i++)
                {
                    currentCol = xOffsets[c];
                    for (int j = 0; j < batch_srcSize[batchCount].width; j++)
                    {

                        if (
                            ((currentRow >= 0) && (currentRow < batch_srcSize[batchCount].height)) &&
                            ((currentCol >= 0) && (currentCol < batch_srcSize[batchCount].width))
                        )
                        {
                            *dstPtrImageTemp = *srcPtrImageTemp;
                        }
                        dstPtrImageTemp += channel;
                        srcPtrImageTemp += channel;
                        currentCol++;
                    }
                    dstPtrImageTemp += increment;
                    srcPtrImageTemp += increment;
                    currentRow++;
                }
            }
            if (outputFormatToggle == 1)
            {
                T *dstPtrImageUnpadded = (T*) calloc(channel * batch_srcSize[batchCount].height * batch_srcSize[batchCount].width, sizeof(T));
                T *dstPtrImageUnpaddedCopy = (T*) calloc(channel * batch_srcSize[batchCount].height * batch_srcSize[batchCount].width, sizeof(T));

                compute_unpadded_from_padded_host(dstPtrImage, batch_srcSize[batchCount], batch_srcSizeMax[batchCount], dstPtrImageUnpadded, chnFormat, channel);

                memcpy(dstPtrImageUnpaddedCopy, dstPtrImageUnpadded, channel * batch_srcSize[batchCount].height * batch_srcSize[batchCount].width * sizeof(T));

                compute_packed_to_planar_host(dstPtrImageUnpaddedCopy, batch_srcSize[batchCount], dstPtrImageUnpadded, channel);

                memset(dstPtrImage, (T) 0, imageDimMax * channel * sizeof(T));

                compute_padded_from_unpadded_host(dstPtrImageUnpadded, batch_srcSize[batchCount], batch_srcSizeMax[batchCount], dstPtrImage, RPPI_CHN_PLANAR, channel);

                free(dstPtrImageUnpadded);
                free(dstPtrImageUnpaddedCopy);
            }
        }
    }

    return RPP_SUCCESS;
}

/**************** ricap - random image crop and patch ***************/

template <typename T>
RppStatus ricap_host_batch(T *srcPtr, RppiSize *batch_srcSize, RppiSize *batch_srcSizeMax, T *dstPtr,
                           Rpp32u *permutedIndices1, Rpp32u *permutedIndices2, Rpp32u *permutedIndices3, Rpp32u *permutedIndices4,
                           Rpp32u *cropRegion1, Rpp32u *cropRegion2, Rpp32u *cropRegion3, Rpp32u *cropRegion4,
                           Rpp32u outputFormatToggle, Rpp32u nbatchSize,
                           RppiChnFormat chnFormat, Rpp32u channel)
{

    if (chnFormat == RPPI_CHN_PLANAR)
    {
        Rpp32u cropBufferInBytes[4];
        cropBufferInBytes[0] = cropRegion1[2] * sizeof(T);
        cropBufferInBytes[1] = cropRegion2[2] * sizeof(T);
        cropBufferInBytes[2] = cropRegion3[2] * sizeof(T);
        cropBufferInBytes[3] = cropRegion4[2] * sizeof(T);

        omp_set_dynamic(0);
#pragma omp parallel for num_threads(nbatchSize)
        for (int batchCount = 0; batchCount < nbatchSize; batchCount++)
        {
            Rpp32u srcAndDstImageDimMax = batch_srcSizeMax[batchCount].height * batch_srcSizeMax[batchCount].width;
            T *srcPtrImage1, *srcPtrImage2, *srcPtrImage3, *srcPtrImage4, *dstPtrImage;
            Rpp32u srcLoc1 = 0, srcLoc2 = 0, srcLoc3 = 0, srcLoc4 = 0, dstLoc = 0;
            compute_image_location_host(batch_srcSizeMax, permutedIndices1[batchCount], &srcLoc1, channel);
            compute_image_location_host(batch_srcSizeMax, permutedIndices2[batchCount], &srcLoc2, channel);
            compute_image_location_host(batch_srcSizeMax, permutedIndices3[batchCount], &srcLoc3, channel);
            compute_image_location_host(batch_srcSizeMax, permutedIndices4[batchCount], &srcLoc4, channel);
            compute_image_location_host(batch_srcSizeMax, batchCount, &dstLoc, channel);
            srcPtrImage1 = srcPtr + srcLoc1;
            srcPtrImage2 = srcPtr + srcLoc2;
            srcPtrImage3 = srcPtr + srcLoc3;
            srcPtrImage4 = srcPtr + srcLoc4;
            dstPtrImage = dstPtr + dstLoc;
            Rpp32u width_srcSizeMax = batch_srcSizeMax[permutedIndices1[0]].width;

            if (outputFormatToggle == 1)
            {
                T *srcPtrImageRoiR1, *srcPtrImageRoiG1, *srcPtrImageRoiB1, *srcPtrImageRoiR2, *srcPtrImageRoiG2, *srcPtrImageRoiB2, *srcPtrImageRoiR3, *srcPtrImageRoiG3, *srcPtrImageRoiB3, *srcPtrImageRoiR4, *srcPtrImageRoiG4, *srcPtrImageRoiB4;
                T *dstPtrImageTemp;

                srcPtrImageRoiR1 = srcPtrImage1 + (cropRegion1[1] * width_srcSizeMax) + cropRegion1[0];
                srcPtrImageRoiG1 = srcPtrImageRoiR1 + srcAndDstImageDimMax;
                srcPtrImageRoiB1 = srcPtrImageRoiG1 + srcAndDstImageDimMax;

                srcPtrImageRoiR2 = srcPtrImage2 + (cropRegion2[1] * width_srcSizeMax) + cropRegion2[0];
                srcPtrImageRoiG2 = srcPtrImageRoiR2 + srcAndDstImageDimMax;
                srcPtrImageRoiB2 = srcPtrImageRoiG2 + srcAndDstImageDimMax;

                srcPtrImageRoiR3 = srcPtrImage3 + (cropRegion3[1] * width_srcSizeMax) + cropRegion3[0];
                srcPtrImageRoiG3 = srcPtrImageRoiR3 + srcAndDstImageDimMax;
                srcPtrImageRoiB3 = srcPtrImageRoiG3 + srcAndDstImageDimMax;

                srcPtrImageRoiR4 = srcPtrImage4 + (cropRegion4[1] * width_srcSizeMax) + cropRegion4[0];
                srcPtrImageRoiG4 = srcPtrImageRoiR4 + srcAndDstImageDimMax;
                srcPtrImageRoiB4 = srcPtrImageRoiG4 + srcAndDstImageDimMax;

                dstPtrImageTemp = dstPtrImage;
                Rpp32u incrementDst = (batch_srcSizeMax[batchCount].width - batch_srcSize[batchCount].width) * channel;

                for (int i = 0; i < cropRegion1[3]; i++)
                {
                    for (int j = 0; j < cropRegion1[2]; j++)
                    {
                        *dstPtrImageTemp = *srcPtrImageRoiR1;
                        srcPtrImageRoiR1++;
                        dstPtrImageTemp++;

                        *dstPtrImageTemp = *srcPtrImageRoiG1;
                        srcPtrImageRoiG1++;
                        dstPtrImageTemp++;

                        *dstPtrImageTemp = *srcPtrImageRoiB1;
                        srcPtrImageRoiB1++;
                        dstPtrImageTemp++;
                    }
                    srcPtrImageRoiR1 += (width_srcSizeMax - cropRegion1[2]);
                    srcPtrImageRoiG1 += (width_srcSizeMax - cropRegion1[2]);
                    srcPtrImageRoiB1 += (width_srcSizeMax - cropRegion1[2]);

                    for (int j = 0; j < cropRegion2[2]; j++)
                    {
                        *dstPtrImageTemp = *srcPtrImageRoiR2;
                        srcPtrImageRoiR2++;
                        dstPtrImageTemp++;

                        *dstPtrImageTemp = *srcPtrImageRoiG2;
                        srcPtrImageRoiG2++;
                        dstPtrImageTemp++;

                        *dstPtrImageTemp = *srcPtrImageRoiB2;
                        srcPtrImageRoiB2++;
                        dstPtrImageTemp++;
                    }

                    srcPtrImageRoiR2 += (width_srcSizeMax - cropRegion2[2]);
                    srcPtrImageRoiG2 += (width_srcSizeMax - cropRegion2[2]);
                    srcPtrImageRoiB2 += (width_srcSizeMax - cropRegion2[2]);
                    dstPtrImageTemp += incrementDst;
                }

                for (int i = 0; i < cropRegion3[3]; i++)
                {
                    for (int j = 0; j < cropRegion3[2]; j++)
                    {
                        *dstPtrImageTemp = *srcPtrImageRoiR3;
                        srcPtrImageRoiR3++;
                        dstPtrImageTemp++;

                        *dstPtrImageTemp = *srcPtrImageRoiG3;
                        srcPtrImageRoiG3++;
                        dstPtrImageTemp++;

                        *dstPtrImageTemp = *srcPtrImageRoiB3;
                        srcPtrImageRoiB3++;
                        dstPtrImageTemp++;
                    }
                    srcPtrImageRoiR3 += (width_srcSizeMax - cropRegion3[2]);
                    srcPtrImageRoiG3 += (width_srcSizeMax - cropRegion3[2]);
                    srcPtrImageRoiB3 += (width_srcSizeMax - cropRegion3[2]);

                    for (int j = 0; j < cropRegion4[2]; j++)
                    {
                        *dstPtrImageTemp = *srcPtrImageRoiR4;
                        srcPtrImageRoiR4++;
                        dstPtrImageTemp++;

                        *dstPtrImageTemp = *srcPtrImageRoiG4;
                        srcPtrImageRoiG4++;
                        dstPtrImageTemp++;

                        *dstPtrImageTemp = *srcPtrImageRoiB4;
                        srcPtrImageRoiB4++;
                        dstPtrImageTemp++;
                    }

                    srcPtrImageRoiR4 += (width_srcSizeMax - cropRegion4[2]);
                    srcPtrImageRoiG4 += (width_srcSizeMax - cropRegion4[2]);
                    srcPtrImageRoiB4 += (width_srcSizeMax - cropRegion4[2]);
                    dstPtrImageTemp += incrementDst;
                }
            }

            else
            {
                T *dstPtrChannel, *srcPtrChannelROI1, *srcPtrChannelROI2, *srcPtrChannelROI3, *srcPtrChannelROI4;
                dstPtrChannel = dstPtrImage;
                srcPtrChannelROI1 = srcPtrImage1 + (cropRegion1[1] * width_srcSizeMax) + cropRegion1[0];
                srcPtrChannelROI2 = srcPtrImage2 + (cropRegion2[1] * width_srcSizeMax) + cropRegion2[0];
                srcPtrChannelROI3 = srcPtrImage3 + (cropRegion3[1] * width_srcSizeMax) + cropRegion3[0];
                srcPtrChannelROI4 = srcPtrImage4 + (cropRegion4[1] * width_srcSizeMax) + cropRegion4[0];

                T *tempDstPtrChannel, *tempSrcPtrChannelROI1, *tempSrcPtrChannelROI2, *tempSrcPtrChannelROI3, *tempSrcPtrChannelROI4;
                tempDstPtrChannel = dstPtrChannel;

                for (int c = 0; c < channel; c++)
                {
                    tempDstPtrChannel = dstPtrChannel;
                    tempSrcPtrChannelROI1 = srcPtrChannelROI1;
                    tempSrcPtrChannelROI2 = srcPtrChannelROI2;
                    tempSrcPtrChannelROI3 = srcPtrChannelROI3;
                    tempSrcPtrChannelROI4 = srcPtrChannelROI4;

                    for (int i = 0; i < cropRegion1[3]; i++)
                    {
                        memcpy(tempDstPtrChannel, tempSrcPtrChannelROI1, cropBufferInBytes[0]);
                        tempDstPtrChannel += cropRegion1[2];
                        tempSrcPtrChannelROI1 += width_srcSizeMax;
                        memcpy(tempDstPtrChannel, tempSrcPtrChannelROI2, cropBufferInBytes[1]);
                        tempDstPtrChannel += cropRegion2[2];
                        tempSrcPtrChannelROI2 += width_srcSizeMax;
                    }

                    for (int i = 0; i < cropRegion3[3]; i++)
                    {
                        memcpy(tempDstPtrChannel, tempSrcPtrChannelROI3, cropBufferInBytes[2]);
                        tempDstPtrChannel += cropRegion3[2];
                        tempSrcPtrChannelROI3 += width_srcSizeMax;
                        memcpy(tempDstPtrChannel, tempSrcPtrChannelROI4, cropBufferInBytes[3]);
                        tempDstPtrChannel += cropRegion4[2];
                        tempSrcPtrChannelROI4 += width_srcSizeMax;
                    }
                    dstPtrChannel += srcAndDstImageDimMax;
                    srcPtrChannelROI1 += srcAndDstImageDimMax;
                    srcPtrChannelROI2 += srcAndDstImageDimMax;
                    srcPtrChannelROI3 += srcAndDstImageDimMax;
                    srcPtrChannelROI4 += srcAndDstImageDimMax;
                }
            }
        }
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        Rpp32u cropBufferInBytes[4], srcElementsinRow[4], cropRegionStartPtr[4];
        Rpp32u width_srcSizeMax_channel = batch_srcSizeMax[permutedIndices1[0]].width * channel;

        cropRegionStartPtr[0] = (cropRegion1[1] * width_srcSizeMax_channel) + (cropRegion1[0] * channel);
        cropRegionStartPtr[1] = (cropRegion2[1] * width_srcSizeMax_channel) + (cropRegion2[0] * channel);
        cropRegionStartPtr[2] = (cropRegion3[1] * width_srcSizeMax_channel) + (cropRegion3[0] * channel);
        cropRegionStartPtr[3] = (cropRegion4[1] * width_srcSizeMax_channel) + (cropRegion4[0] * channel);

        srcElementsinRow[0] = cropRegion1[2] * channel;
        srcElementsinRow[1] = cropRegion2[2] * channel;
        srcElementsinRow[2] = cropRegion3[2] * channel;
        srcElementsinRow[3] = cropRegion4[2] * channel;

        cropBufferInBytes[0] = srcElementsinRow[0] * sizeof(T);
        cropBufferInBytes[1] = srcElementsinRow[1] * sizeof(T);
        cropBufferInBytes[2] = srcElementsinRow[2] * sizeof(T);
        cropBufferInBytes[3] = srcElementsinRow[3] * sizeof(T);

        omp_set_dynamic(0);
#pragma omp parallel for num_threads(nbatchSize)
        for (int batchCount = 0; batchCount < nbatchSize; batchCount++)
        {
            Rpp32u srcAndDstImageDimMax = batch_srcSizeMax[batchCount].height * batch_srcSizeMax[batchCount].width;

            T *srcPtrImage1, *srcPtrImage2, *srcPtrImage3, *srcPtrImage4, *dstPtrImage;
            Rpp32u srcLoc1 = 0, srcLoc2 = 0, srcLoc3 = 0, srcLoc4 = 0, dstLoc = 0;
            compute_image_location_host(batch_srcSizeMax, permutedIndices1[batchCount], &srcLoc1, channel);
            compute_image_location_host(batch_srcSizeMax, permutedIndices2[batchCount], &srcLoc2, channel);
            compute_image_location_host(batch_srcSizeMax, permutedIndices3[batchCount], &srcLoc3, channel);
            compute_image_location_host(batch_srcSizeMax, permutedIndices4[batchCount], &srcLoc4, channel);
            compute_image_location_host(batch_srcSizeMax, batchCount, &dstLoc, channel);
            srcPtrImage1 = srcPtr + srcLoc1;
            srcPtrImage2 = srcPtr + srcLoc2;
            srcPtrImage3 = srcPtr + srcLoc3;
            srcPtrImage4 = srcPtr + srcLoc4;
            dstPtrImage = dstPtr + dstLoc;
            if (outputFormatToggle == 1)
            {
                T *srcPtrImageROI1, *srcPtrImageROI2, *srcPtrImageROI3, *srcPtrImageROI4;
                srcPtrImageROI1 = srcPtrImage1 + cropRegionStartPtr[0];
                srcPtrImageROI2 = srcPtrImage2 + cropRegionStartPtr[1];
                srcPtrImageROI3 = srcPtrImage3 + cropRegionStartPtr[2];
                srcPtrImageROI4 = srcPtrImage4 + cropRegionStartPtr[3];

                T *dstPtrImageTempG, *dstPtrImageTempB, *dstPtrImageTempR;
                dstPtrImageTempR = dstPtrImage;
                dstPtrImageTempG = dstPtrImageTempR + (srcAndDstImageDimMax);
                dstPtrImageTempB = dstPtrImageTempG + (srcAndDstImageDimMax);

                Rpp32u increment = batch_srcSizeMax[batchCount].width - batch_srcSize[batchCount].width;

                for (int i = 0; i < cropRegion1[3]; i++)
                {
                    T *srcPtrImageROITemp1, *srcPtrImageROITemp2;
                    srcPtrImageROITemp1 = srcPtrImageROI1;
                    srcPtrImageROITemp2 = srcPtrImageROI2;
                    for (int j = 0; j < cropRegion1[2]; j++)
                    {
                        *dstPtrImageTempR = *srcPtrImageROITemp1;
                        srcPtrImageROITemp1++;
                        dstPtrImageTempR++;

                        *dstPtrImageTempG = *srcPtrImageROITemp1;
                        srcPtrImageROITemp1++;
                        dstPtrImageTempG++;

                        *dstPtrImageTempB = *srcPtrImageROITemp1;
                        srcPtrImageROITemp1++;
                        dstPtrImageTempB++;
                    }
                    for (int j = 0; j < cropRegion2[2]; j++)
                    {
                        *dstPtrImageTempR = *srcPtrImageROITemp2;
                        srcPtrImageROITemp2++;
                        dstPtrImageTempR++;

                        *dstPtrImageTempG = *srcPtrImageROITemp2;
                        srcPtrImageROITemp2++;
                        dstPtrImageTempG++;

                        *dstPtrImageTempB = *srcPtrImageROITemp2;
                        srcPtrImageROITemp2++;
                        dstPtrImageTempB++;
                    }
                    srcPtrImageROI1 += width_srcSizeMax_channel;
                    srcPtrImageROI2 += width_srcSizeMax_channel;
                }

                for (int i = 0; i < cropRegion3[3]; i++)
                {
                    T *srcPtrImageROITemp3, *srcPtrImageROITemp4;
                    srcPtrImageROITemp3 = srcPtrImageROI3;
                    srcPtrImageROITemp4 = srcPtrImageROI4;
                    for (int j = 0; j < cropRegion3[2]; j++)
                    {
                        *dstPtrImageTempR = *srcPtrImageROITemp3;
                        srcPtrImageROITemp3++;
                        dstPtrImageTempR++;

                        *dstPtrImageTempG = *srcPtrImageROITemp3;
                        srcPtrImageROITemp3++;
                        dstPtrImageTempG++;

                        *dstPtrImageTempB = *srcPtrImageROITemp3;
                        srcPtrImageROITemp3++;
                        dstPtrImageTempB++;
                    }
                    for (int j = 0; j < cropRegion4[2]; j++)
                    {
                        *dstPtrImageTempR = *srcPtrImageROITemp4;
                        srcPtrImageROITemp4++;
                        dstPtrImageTempR++;

                        *dstPtrImageTempG = *srcPtrImageROITemp4;
                        srcPtrImageROITemp4++;
                        dstPtrImageTempG++;

                        *dstPtrImageTempB = *srcPtrImageROITemp4;
                        srcPtrImageROITemp4++;
                        dstPtrImageTempB++;
                    }
                    srcPtrImageROI3 += width_srcSizeMax_channel;
                    srcPtrImageROI4 += width_srcSizeMax_channel;
                }
            }
            else
            {

                T *dstPtrImageTemp, *srcPtrImageROI1, *srcPtrImageROI2, *srcPtrImageROI3, *srcPtrImageROI4;
                dstPtrImageTemp = dstPtrImage;
                srcPtrImageROI1 = srcPtrImage1 + cropRegionStartPtr[0];
                srcPtrImageROI2 = srcPtrImage2 + cropRegionStartPtr[1];
                srcPtrImageROI3 = srcPtrImage3 + cropRegionStartPtr[2];
                srcPtrImageROI4 = srcPtrImage4 + cropRegionStartPtr[3];

                for (int i = 0; i < cropRegion1[3]; i++)
                {
                    memcpy(dstPtrImageTemp, srcPtrImageROI1, cropBufferInBytes[0]);
                    dstPtrImageTemp += srcElementsinRow[0];
                    srcPtrImageROI1 += width_srcSizeMax_channel;
                    memcpy(dstPtrImageTemp, srcPtrImageROI2, cropBufferInBytes[1]);
                    dstPtrImageTemp += srcElementsinRow[1];
                    srcPtrImageROI2 += width_srcSizeMax_channel;
                }

                for (int i = 0; i < cropRegion3[3]; i++)
                {
                    memcpy(dstPtrImageTemp, srcPtrImageROI3, cropBufferInBytes[2]);
                    dstPtrImageTemp += srcElementsinRow[2];
                    srcPtrImageROI3 += width_srcSizeMax_channel;
                    memcpy(dstPtrImageTemp, srcPtrImageROI4, cropBufferInBytes[3]);
                    dstPtrImageTemp += srcElementsinRow[3];
                    srcPtrImageROI4 += width_srcSizeMax_channel;
                }
            }
        }
    }
    return RPP_SUCCESS;
}
#endif