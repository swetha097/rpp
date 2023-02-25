#include "rppdefs.h"
#include "rpp_cpu_simd.hpp"
#include "rpp_cpu_common.hpp"
#include "../../../../third_party/ffts/include/ffts.h"
#include "../../../..//third_party/ffts/include/ffts_attributes.h"

inline void hann_window(Rpp32f *output, Rpp32s windowSize)
{
    Rpp64f a = (2.0 * M_PI) / windowSize;
    for (Rpp32s t = 0; t < windowSize; t++)
    {
        Rpp64f phase = a * (t + 0.5);
        output[t] = (0.5 * (1.0 - std::cos(phase)));
    }
}

bool is_pow2(int64_t n) { return (n & (n-1)) == 0; }

inline bool can_use_real_impl(int64_t n) { return is_pow2(n); }

inline int64_t size_in_buf(int64_t n) {
  // Real impl input needs:    N real numbers    -> N floats
  // Complex impl input needs: N complex numbers -> 2*N floats
  return can_use_real_impl(n) ? n : 2*n;
}

inline int64_t size_out_buf(int64_t n) {
  // Real impl output needs:    (N/2)+1 complex numbers -> N+2 floats
  // Complex impl output needs: N complex numbers       -> 2*N floats
  return can_use_real_impl(n) ? n+2 : 2*n;
}

inline Rpp32s get_num_windows(Rpp32s length, Rpp32s windowLength, Rpp32s windowStep, bool centerWindows)
{
    if (!centerWindows)
        length -= windowLength;
    return ((length / windowStep) + 1);
}

inline Rpp32s get_idx_reflect(Rpp32s idx, Rpp32s lo, Rpp32s hi)
{
    if (hi - lo < 2)
        return hi - 1;
    for (;;)
    {
        if (idx < lo)
            idx = 2 * lo - idx;
        else if (idx >= hi)
            idx = 2 * hi - 2 - idx;
        else
            break;
    }
    return idx;
}

RppStatus spectrogram_host_tensor(Rpp32f *srcPtr,
                                  RpptDescPtr srcDescPtr,
                                  Rpp32f *dstPtr,
                                  RpptDescPtr dstDescPtr,
                                  Rpp32s *srcLengthTensor,
                                  bool centerWindows,
                                  bool reflectPadding,
                                  Rpp32f *windowFunction,
                                  Rpp32s nfft,
                                  Rpp32s power,
                                  Rpp32s windowLength,
                                  Rpp32s windowStep,
                                  RpptSpectrogramLayout layout)
{
    Rpp32s windowCenterOffset = 0;
    bool vertical = (layout == RpptSpectrogramLayout::FT);
    if (centerWindows)
        windowCenterOffset = windowLength / 2;
    if (nfft == 0)
        nfft = windowLength;
    Rpp32s numBins = nfft / 2 + 1;
    const Rpp32f mulFactor = (2.0 * M_PI) / nfft;
    Rpp32u hStride = dstDescPtr->strides.hStride;
    Rpp32s alignedNfftLength = nfft & ~7;
    Rpp32s alignedNbinsLength = numBins & ~7;
    Rpp32s alignedWindowLength = windowLength & ~7;

    std::vector<Rpp32f> windowFn;
    windowFn.resize(windowLength);
    // Generate hanning window
    if (windowFunction == NULL)
        hann_window(windowFn.data(), windowLength);
    else
        memcpy(windowFn.data(), windowFunction, windowLength * sizeof(Rpp32f));

    // Get windows output
    omp_set_dynamic(0);
#pragma omp parallel for num_threads(8)
    for (Rpp32s batchCount = 0; batchCount < srcDescPtr->n; batchCount++)
    {
        Rpp32f *srcPtrTemp = srcPtr + batchCount * srcDescPtr->strides.nStride;
        Rpp32f *dstPtrTemp = dstPtr + batchCount * dstDescPtr->strides.nStride;
        Rpp32s bufferLength = srcLengthTensor[batchCount];
        Rpp32s numWindows = get_num_windows(bufferLength, windowLength, windowStep, centerWindows);
        Rpp32f windowOutput[numWindows * nfft];
        std::fill_n (windowOutput, numWindows * nfft, 0);
        for (Rpp64s w = 0; w < numWindows; w++)
        {
            Rpp64s windowStart = w * windowStep - windowCenterOffset;
            Rpp32f *windowOutputTemp = windowOutput + (w * nfft);
            if (windowStart < 0 || (windowStart + windowLength) > bufferLength)
            {
                for (Rpp32s t = 0; t < windowLength; t++)
                {
                    Rpp64s inIdx = windowStart + t;
                    if (reflectPadding)
                    {
                        inIdx = get_idx_reflect(inIdx, 0, bufferLength);
                        *windowOutputTemp++ = windowFn[t] * srcPtrTemp[inIdx];
                    }
                    else
                    {
                        if (inIdx >= 0 && inIdx < bufferLength)
                            *windowOutputTemp++ = windowFn[t] * srcPtrTemp[inIdx];
                        else
                            *windowOutputTemp++ = 0;
                    }
                }
            }
            else
            {
                Rpp32f *srcPtrWindowTemp = srcPtrTemp + windowStart;
                Rpp32f *windowFnTemp = windowFn.data();
                Rpp32s t = 0;
                for (; t < alignedWindowLength; t += 8)
                {
                    __m256 pSrc, pWindowFn;
                    pSrc = _mm256_loadu_ps(srcPtrWindowTemp);
                    pWindowFn = _mm256_loadu_ps(windowFnTemp);
                    pSrc = _mm256_mul_ps(pSrc, pWindowFn);
                    _mm256_storeu_ps(windowOutputTemp, pSrc);
                    srcPtrWindowTemp += 8;
                    windowFnTemp += 8;
                    windowOutputTemp += 8;
                }
                for (; t < windowLength; t++)
                    *windowOutputTemp++ = (*windowFnTemp++) * (*srcPtrWindowTemp++);
            }
        }

        // Generate FFT output
        ffts_plan_t *p;
        bool useRealImpl = can_use_real_impl(nfft);
        if(useRealImpl)
            p = ffts_init_1d_real(nfft, FFTS_FORWARD);
        else
            p = ffts_init_1d(nfft, FFTS_FORWARD);

        if (!p) {
            printf("FFT Plan is unsupported. Exiting the code\n");
            exit(0);
        }

        auto fftInSize = size_in_buf(nfft);
        auto fftOutSize = size_out_buf(nfft);

        // Set temporary buffers to 0
        Rpp32f FFTS_ALIGN(32) *fftInBuf = (Rpp32f *)_mm_malloc(fftInSize * sizeof(Rpp32f), 32); // ffts requires 32-byte aligned memory
        Rpp32f FFTS_ALIGN(32) *fftOutBuf = (Rpp32f *)_mm_malloc(fftOutSize * sizeof(Rpp32f), 32); // ffts requires 32-byte aligned memory

        for (Rpp64s w = 0; w < numWindows; w++)
        {
            Rpp32f *dstPtrBinTemp = dstPtrTemp + (w * hStride);
            Rpp32f *windowOutputTemp = windowOutput + (w * nfft);
            for(int k = 0; k < fftInSize; k++)
                fftInBuf[k] = 0.0f;

            for(int k = 0; k < fftOutSize; k++)
                fftOutBuf[k] = 0.0f;

            // memset(fftInBuf, 0, fftInSize * sizeof(Rpp32f));
            // memset(fftOutBuf, 0, fftOutSize * sizeof(Rpp32f));
            Rpp32s inWindowStart = windowLength < nfft ? (nfft - windowLength) / 2 : 0;

            // Copy the window input to fftInBuf
            if (useRealImpl)
            {
                for (int i = 0; i < windowLength; i++)
                    fftInBuf[inWindowStart + i] = windowOutputTemp[i];
            }
            else
            {
                for (int i = 0; i < windowLength; i++)
                {
                    int64_t off = 2 * (inWindowStart + i);
                    fftInBuf[off] = windowOutputTemp[i];
                    fftInBuf[off + 1] = 0.0f;
                }
            }

            ffts_execute(p, fftInBuf, fftOutBuf);
            auto *complexFft = reinterpret_cast<std::complex<Rpp32f> *>(fftOutBuf);
            for (int i = 0; i < numBins; i++)
            {
                if (vertical)
                {
                    Rpp64s outIdx = (i * hStride + w);
                    if (power == 1)
                        dstPtrTemp[outIdx] = std::abs(complexFft[i]);
                    else
                        dstPtrTemp[outIdx] = std::norm(complexFft[i]);
                }
                else
                {
                    if (power == 1)
                        *dstPtrBinTemp++ = std::abs(complexFft[i]);
                    else
                        *dstPtrBinTemp++ = std::norm(complexFft[i]);
                }
            }
        }
        _mm_free(fftInBuf);
        _mm_free(fftOutBuf);
    }
    return RPP_SUCCESS;
}