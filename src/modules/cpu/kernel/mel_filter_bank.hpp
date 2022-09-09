#include "rppdefs.h"
#include "rpp_cpu_simd.hpp"
#include "rpp_cpu_common.hpp"

template <typename T>
struct HtkMelScale {
    T hz_to_mel(T hz) {
    // equivalent to `2595.0 * std::log10(1 + hz / 700.0)`
    return T(1127) * std::log(T(1) + hz / T(700));
  }

    T mel_to_hz(T mel) {
    // equivalent to `700.0 * (std::pow(10, mel / 2595.0) - 1.0)`
    return T(700) * (std::exp(mel / T(1127)) - T(1));
  }
};

template <typename T>
struct SlaneyMelScale {
  static constexpr T freq_low = 0;
  static constexpr T fsp = 200.0 / 3.0;

  static constexpr T min_log_hz = 1000.0;
  static constexpr T min_log_mel = (min_log_hz - freq_low) / fsp;
  static constexpr T step_log = 0.068751777;  // Equivalent to std::log(6.4) / 27.0;

  T hz_to_mel(T hz) {
    T mel = 0;
    if (hz >= min_log_hz) {
      // non-linear scale
      mel = min_log_mel + std::log(hz / min_log_hz) / step_log;
    } else {
      // linear scale
      mel = (hz - freq_low) / fsp;
    }

    return mel;
  }

  T mel_to_hz(T mel) {
    T hz = 0;
    if (mel >= min_log_mel) {
      // non linear scale
      hz = min_log_hz * std::exp(step_log * (mel - min_log_mel));
    } else {
      // linear scale
      hz = freq_low + mel * fsp;
    }
    return hz;
  }
};

RppStatus mel_filter_bank_host_tensor(Rpp32f *srcPtr,
                                      RpptDescPtr srcDescPtr,
                                      Rpp32f *dstPtr,
									  RpptDescPtr dstDescPtr,
                                      RpptImagePatchPtr srcDims,
                                      Rpp32f maxFreq,
                                      Rpp32f minFreq,
                                      std::string melFormula,
                                      Rpp32s numFilter,
                                      Rpp32f sampleRate,
                                      bool normalize)
{
	omp_set_dynamic(0);
#pragma omp parallel for num_threads(srcDescPtr->n)
	for(int batchCount = 0; batchCount < srcDescPtr->n; batchCount++)
	{
		Rpp32f *srcPtrTemp = srcPtr + batchCount * srcDescPtr->strides.nStride;
		Rpp32f *dstPtrTemp = dstPtr + batchCount * dstDescPtr->strides.nStride;

        int nfft = (srcDims[batchCount].width - 1) * 2;

        // Algorithm to generate traingular matrix
        HtkMelScale<float> mel_scale;

        std::vector<std::vector<float>> fbanks(numFilter);
        auto low_mel = mel_scale.hz_to_mel(minFreq);
        auto high_mel = mel_scale.hz_to_mel(maxFreq);
        float delta_mel = (high_mel - low_mel) / (numFilter + 1);
        std::vector<float> mel_points(numFilter + 2, 0.0f);
        mel_points[0] = low_mel;
        for (int i = 1; i < numFilter + 1; i++) {
            mel_points[i] = mel_points[i - 1] + delta_mel;
        }
        mel_points[numFilter + 1] = high_mel;

        std::vector<float> fftfreqs(nfft / 2 + 1, 0.0f);
        for (int i = 0; i < nfft / 2 + 1; i++) {
            fftfreqs[i] = i * sampleRate / nfft;
        }

        std::vector<float> freq_grid(mel_points.size(), 0.0f);
        freq_grid[0] = minFreq;
        for (int i = 1; i < numFilter+1; i++) {
            freq_grid[i] = mel_scale.mel_to_hz(mel_points[i]);
        }
        freq_grid[numFilter+1] = maxFreq;

        for (int j = 0; j < numFilter; j++)
        {
            auto &fbank = fbanks[j];
            fbank.resize(nfft / 2 + 1, 0.0f);
            for (int i = 0; i < nfft/2 + 1; i++)
            {
                auto f = fftfreqs[i];
                if (f < minFreq || f > maxFreq)
                {
                    fbank[i] = 0.0f;
                }
                else
                {
                    auto upper = (f - freq_grid[j]) / (freq_grid[j + 1] - freq_grid[j]);
                    auto lower = (freq_grid[j + 2] - f) / (freq_grid[j + 2] - freq_grid[j + 1]);
                    fbank[i] = std::max(0.0f, std::min(upper, lower));
                }
            }
        }

        // Matrix multiplication of fbank and srcPtrTemp
        int R2 = srcDims[batchCount].height;
        for (int i = 0; i < numFilter; i++)
        {
            for (int j = 0; j < nfft; j++)
            {
                dstPtrTemp[i] = 0;
                for (int k = 0; k < R2; k++)
                {
                    dstPtrTemp[i] += srcPtrTemp[i] * fbanks[k][j];
                }
            }
        }
    }

    return RPP_SUCCESS;
}
