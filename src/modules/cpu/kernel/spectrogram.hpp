#include "rppdefs.h"
#include "rpp_cpu_simd.hpp"
#include "rpp_cpu_common.hpp"
#include<complex>

void HannWindow(float *output, int nfft)
{
  int N = nfft;
  double a = (2 * M_PI / N);
  for (int t = 0; t < N; t++) {
    double phase = a * (t + 0.5);
    output[t] = (0.5 * (1.0 - std::cos(phase)));
  }
}

int getOutputSize(int length, int windowLength, int windowStep, bool centerWindows)
{
    if(!centerWindows)
        length -= windowLength;

    return ((length / windowStep) + 1);
}

int getIdxReflect(int idx, int lo, int hi)
{
  if (hi - lo < 2)
    return hi - 1;
  for (;;) {
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
	omp_set_dynamic(0);
#pragma omp parallel for num_threads(srcDescPtr->n)
	for(int batchCount = 0; batchCount < srcDescPtr->n; batchCount++)
	{
        Rpp32f *srcPtrTemp = srcPtr + batchCount * srcDescPtr->strides.nStride;
		Rpp32f *dstPtrTemp = dstPtr + batchCount * dstDescPtr->strides.nStride;
        Rpp32s bufferLength = srcLengthTensor[batchCount];
        bool vertical = (layout == RpptSpectrogramLayout::FT);

        // Generate hanning window
        std::vector<float> windowFn;
        windowFn.resize(nfft);
        HannWindow(windowFn.data(), nfft);

        Rpp32s numBins = nfft / 2 + 1;
        Rpp32s numWindows = getOutputSize(bufferLength, windowLength, windowStep, centerWindows);
        Rpp32s windowCenterOffset = 0;

        if(centerWindows)
            windowCenterOffset = windowLength / 2;

        std::vector<Rpp32f> windowOutput(nfft * numWindows);
        for (int64_t w = 0; w < numWindows; w++)
        {
            int64_t windowStart = w * windowStep - windowCenterOffset;
            if (windowStart < 0 || (windowStart + windowLength) > bufferLength)
            {
                for (int t = 0; t < windowLength; t++)
                {
                    int64_t outIdx = (vertical) ? (t * numWindows + w) : (w * windowLength + t);
                    int64_t inIdx = windowStart + t;
                    if (reflectPadding)
                    {
                        inIdx = getIdxReflect(inIdx, 0, bufferLength);
                        windowOutput[outIdx] = windowFn[t] * srcPtrTemp[inIdx];
                    }
                    else
                    {
                        if(inIdx >= 0 && inIdx < bufferLength)
                            windowOutput[outIdx] = windowFn[t] * srcPtrTemp[inIdx];
                        else
                        windowOutput[outIdx] = 0;
                    }
                }
            }
            else
            {
                for (int t = 0; t < windowLength; t++)
                {
                    int64_t outIdx = (vertical) ? (t * numWindows + w) : (w * windowLength + t);
                    int64_t inIdx = windowStart + t;
                    windowOutput[outIdx] = windowFn[t] * srcPtrTemp[inIdx];
                }
            }
        }

        std::vector<Rpp32f> windowOutputTemp(nfft);
        for (int w = 0; w < numWindows; w++)
        {
            for (int i = 0; i < nfft; i++)
                windowOutputTemp[i] = windowOutput[i * numWindows + w];

            // Allocate buffers for fft output
            std::vector<std::complex<Rpp32f>> fftOutput;
            fftOutput.clear();
            fftOutput.reserve(numBins);

            // Compute FFT
            for (int k = 0; k < numBins; k++)
            {
                Rpp32f real = 0.0f, imag = 0.0f;
                for (int i = 0; i < windowOutputTemp.size(); i++)
                {
                    auto x = windowOutputTemp[i];
                    real += x * cos(2.0f * M_PI * k * i / nfft);
                    imag += -x * sin(2.0f * M_PI * k * i / nfft);
                }
                fftOutput.push_back({real, imag});
            }

            if(power == 2)
            {
                // Compute power spectrum
                for (int i = 0; i < numBins; i++)
                    dstPtrTemp[i * numWindows + w] = std::norm(fftOutput[i]);
            }
            else
            {
                // Compute magnitude spectrum
                for (int i = 0; i < numBins; i++)
                    dstPtrTemp[i * numWindows + w] = std::abs(fftOutput[i]);
            }
        }
    }

	return RPP_SUCCESS;
}
