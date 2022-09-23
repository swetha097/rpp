#include "rppdefs.h"
#include "rpp_cpu_simd.hpp"
#include "rpp_cpu_common.hpp"

struct BaseMelScale
{
    public:
        virtual float hz_to_mel(float hz) = 0;
        virtual float mel_to_hz(float mel) = 0;
};

struct HtkMelScale : public BaseMelScale
{
    float hz_to_mel(float hz) { return 1127.0f * std::log(1.0f + hz / 700.0f); }
    float mel_to_hz(float mel) { return 700.0f * (std::exp(mel / 1127.0f) - 1.0f); }
};

struct SlaneyMelScale : public BaseMelScale
{
	const float freq_low = 0;
	const float fsp = 200.0 / 3.0;
	const float min_log_hz = 1000.0;
	const float min_log_mel = (min_log_hz - freq_low) / fsp;
	const float step_log = 0.068751777;  // Equivalent to std::log(6.4) / 27.0;

	float hz_to_mel(float hz)
	{
		float mel = 0.0f;
		if (hz >= min_log_hz)
        {
		    mel = min_log_mel + std::log(hz / min_log_hz) / step_log;
		}
        else
        {
		    mel = (hz - freq_low) / fsp;
		}
		return mel;
	}

	float mel_to_hz(float mel)
	{
		float hz = 0;
		if (mel >= min_log_mel)
        {
			hz = min_log_hz * std::exp(step_log * (mel - min_log_mel));
		}
        else
        {
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
                                      RpptMelScaleFormula melFormula,
                                      Rpp32s numFilter,
                                      Rpp32f sampleRate,
                                      bool normalize)
{
    BaseMelScale *melScalePtr;
    switch(melFormula)
    {
        case RpptMelScaleFormula::HTK:
            melScalePtr = new HtkMelScale;
            break;
        case RpptMelScaleFormula::SLANEY:
        default:
            melScalePtr = new SlaneyMelScale;
            break;
    }

	omp_set_dynamic(0);
#pragma omp parallel for num_threads(srcDescPtr->n)
	for(int batchCount = 0; batchCount < srcDescPtr->n; batchCount++)
	{
		Rpp32f *srcPtrTemp = srcPtr + batchCount * srcDescPtr->strides.nStride;
		Rpp32f *dstPtrTemp = dstPtr + batchCount * dstDescPtr->strides.nStride;

        // Extract nfft, number of Frames, numBins
        int nfft = (srcDims[batchCount].height - 1) * 2;
        int numBins = nfft / 2 + 1;
        int numFrames = srcDims[batchCount].width;

        // Convert lower, higher freqeuncies to mel scale
        double melLow = melScalePtr->hz_to_mel(minFreq);
        double melHigh = melScalePtr->hz_to_mel(maxFreq);
        double melStep = (melHigh - melLow) / (numFilter + 1);
        double hzStep = static_cast<double>(sampleRate) / nfft;
        double invHzStep = 1.0 / hzStep;

        int fftbin_start_ = std::ceil(minFreq * invHzStep);
        int fftbin_end_ = std::floor(maxFreq * invHzStep);
        if (fftbin_end_ > numBins - 1)
            fftbin_end_ = numBins - 1;

        std::vector<float> weights_down_;
        weights_down_.resize(numBins);

        std::vector<float> norm_factors_;
        norm_factors_.resize(numFilter, float(1));

        std::vector<int> intervals_;
        intervals_.resize(numBins, -1);

        int last_interval = numFilter;
        int fftbin = fftbin_start_;
        double mel0 = melLow, mel1 = melLow + melStep;
        double f = fftbin * hzStep;
        for (int interval = 0; interval < numFilter + 1; interval++, mel0 = mel1, mel1 += melStep)
        {
            double f0 = melScalePtr->mel_to_hz(mel0);
            double f1 = melScalePtr->mel_to_hz(interval == numFilter ? melHigh : mel1);
            double slope = 1. / (f1 - f0);
            for (; fftbin <= fftbin_end_ && f < f1; fftbin++, f = fftbin * hzStep)
            {
                weights_down_[fftbin] = (f1 - f) * slope;
                intervals_[fftbin] = interval;
            }
        }

        for (int64_t m = 0; m < numFilter; m++)
        {
            float* out_row = dstPtrTemp + m * numFrames;
            for (int64_t t = 0; t < numFrames; t++)
                out_row[t] = 0.0f;
        }

        const float *in_row = srcPtrTemp + fftbin_start_ * numFrames;
        for (int64_t fftbin = fftbin_start_; fftbin <= fftbin_end_; fftbin++, in_row += numFrames)
        {
            auto filter_up = intervals_[fftbin];
            auto weight_up = float(1) - weights_down_[fftbin];
            auto filter_down = filter_up - 1;
            auto weight_down = weights_down_[fftbin];

            if (filter_down >= 0)
            {
                if (normalize)
                weight_down *= norm_factors_[filter_down];

                float *out_row = dstPtrTemp + filter_down * numFrames;
                for (int t = 0; t < numFrames; t++)
                {
                    out_row[t] += weight_down * in_row[t];
                }
            }

            if (filter_up >= 0 && filter_up < numFilter)
            {
                if (normalize)
                weight_up *= norm_factors_[filter_up];

                float *out_row = dstPtrTemp + filter_up * numFrames;
                for (int t = 0; t < numFrames; t++)
                {
                    out_row[t] += weight_up * in_row[t];
                }
            }
        }
    }

    return RPP_SUCCESS;
}
