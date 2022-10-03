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

void NaiveDft(std::vector<std::complex<float>> &out,
              float *in,
              int srcWidth,
              int nfft,
              bool full_spectrum = false)
{
    int n = srcWidth;
    auto out_size = nfft/2 + 1;
    out.clear();
    out.reserve(out_size);

    // Loop through each sample in the frequency domain
    for (int64_t k = 0; k < out_size; k++)
    {
        float real = 0.0f, imag = 0.0f;
        // Loop through each sample in the time domain
        for (int64_t i = 0; i < n; i++)
        {
            auto x = in[i];
            real += x * cos(2.0f * M_PI * k * i / n);
            imag += -x * sin(2.0f * M_PI * k * i / n);
        }
        out.push_back({real, imag});
    }
}



RppStatus spectrogram_host_tensor(Rpp32f *srcPtr,
                                  RpptDescPtr srcDescPtr,
                                  Rpp32f *dstPtr,
								  RpptDescPtr dstDescPtr,
                                  Rpp32s *srcLengthTensor,
                                  RpptImagePatchPtr dstDims,
                                  bool centerWindows,
                                  bool reflectPadding,
                                  Rpp32f *windowFunction,
                                  Rpp32s nfft,
                                  Rpp32s power,
                                  Rpp32s windowLength,
                                  Rpp32s windowStep,
                                  std::string layout)
{
	omp_set_dynamic(0);
#pragma omp parallel for num_threads(srcDescPtr->n)
	for(int batchCount = 0; batchCount < srcDescPtr->n; batchCount++)
	{
        Rpp32f *srcPtrTemp = srcPtr + batchCount * srcDescPtr->strides.nStride;
		Rpp32f *dstPtrTemp = dstPtr + batchCount * dstDescPtr->strides.nStride;
        Rpp32s bufferLength = srcLengthTensor[batchCount];
        bool vertical = (layout == "ft");

        // Generate hanning window
        std::vector<float> windowFn;
        windowFn.resize(nfft);
        HannWindow(windowFn.data(), nfft);

        Rpp32s numBins = nfft / 2 + 1;
        Rpp32s numWindows = getOutputSize(bufferLength, windowLength, windowStep, centerWindows);
        Rpp32s windowCenterOffset = 0;

        if(centerWindows)
            windowCenterOffset = windowLength / 2;

        Rpp32f *windowOut = (Rpp32f *)malloc(nfft * numWindows * sizeof(float));
        for (int64_t w = 0; w < numWindows; w++)
        {
            int64_t windowStart = w * windowStep - windowCenterOffset;
            if (windowStart < 0 || (windowStart + windowLength) > bufferLength)
            {
                for (int t = 0; t < windowLength; t++)
                {
                    int64_t out_idx = (vertical) ? (t * numWindows + w) : (w * windowLength + t);
                    int64_t in_idx = windowStart + t;
                    if(in_idx >= 0 && in_idx < bufferLength)
                        windowOut[out_idx] = windowFn[t] * srcPtrTemp[in_idx];
                    else
                       windowOut[out_idx] = 0;
                }
            }
            else
            {
                for (int t = 0; t < windowLength; t++)
                {
                    int64_t out_idx = (vertical) ? (t * numWindows + w) : (w * windowLength + t);
                    int64_t in_idx = windowStart + t;
                    windowOut[out_idx] = windowFn[t] * srcPtrTemp[in_idx];
                }
            }
        }

        // std::vector<std::complex<float>> outFft;
        // NaiveDft(outFft, windowOut, numWindows, nfft, false);
        // // auto *complex_fft = reinterpret_cast<std::complex<float> *>(outFft);
        // int out_size = nfft / 2 + 1;
        // int out_stride = numWindows;
        // int in_stride = 1;

        // Rpp32f *finalOut = (Rpp32f *)malloc((nfft / 2 +1) * numWindows * sizeof(float));

        // for (int i = 0; i < out_size; i++)
        // {
        //     for(j = 0; j < numWindows; j++)
        //     {
        //         finalOut[j * out_size + i] = std::norm(outFft[i * in_stride]);
        //     }
        // }

        // MagnitudeSpectrumCalculator().Calculate(
        //     args.spectrum_type, out_data, complex_fft, out_size, out_stride, 1);
        // }

        free(windowOut);
        // free(finalOut);
    }
	return RPP_SUCCESS;
}
