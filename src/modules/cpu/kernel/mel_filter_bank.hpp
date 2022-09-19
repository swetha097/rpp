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
	static const float freq_low = 0;
	static const float fsp = 200.0 / 3.0;
	static const float min_log_hz = 1000.0;
	static const float min_log_mel = (min_log_hz - freq_low) / fsp;
	static const float step_log = 0.068751777;  // Equivalent to std::log(6.4) / 27.0;

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

        int nfft = (srcDims[batchCount].height - 1) * 2;
        int numBins = nfft / 2 + 1;
        int numFrames = srcDims[batchCount].width;
        std::vector<std::vector<float>> filterBanks(numFilter);

        // Algorithm to generate traingular matrix
        Rpp32f melLow = melScalePtr->hz_to_mel(minFreq);
        Rpp32f melHigh = melScalePtr->hz_to_mel(maxFreq);
        Rpp32f melStep = (melHigh - melLow) / (numFilter + 1);

        // Create mel scale points
        std::vector<Rpp32f> melPoints(numFilter + 2, 0.0f);
        melPoints[0] = melLow;
        melPoints[numFilter + 1] = melHigh;
        for (int i = 1; i < numFilter + 1; i++)
           melPoints[i] = melPoints[i - 1] + melStep;

        // Convert mel scale points to hz scale
        std::vector<Rpp32f> freqGrid(melPoints.size(), 0.0f);
        freqGrid[0] = minFreq;
        freqGrid[numFilter + 1] = maxFreq;
        for (int i = 1; i < numFilter + 1; i++)
            freqGrid[i] = melScalePtr->mel_to_hz(melPoints[i]);

        std::vector<float> fftFreqs(numBins, 0.0f);
        float invFactor = sampleRate / nfft;
        for (int i = 0; i < numBins; i++)
            fftFreqs[i] = i * invFactor;

        for (int j = 0; j < numFilter; j++)
        {
            auto &fbank = filterBanks[j];
            fbank.resize(numBins, 0.0f);
            for (int i = 0; i < nfft/2 + 1; i++)
            {
                auto f = fftFreqs[i];
                if (f < minFreq || f > maxFreq)
                {
                    fbank[i] = 0.0f;
                }
                else
                {
                    auto upper = (f - freqGrid[j]) / (freqGrid[j + 1] - freqGrid[j]);
                    auto lower = (freqGrid[j + 2] - f) / (freqGrid[j + 2] - freqGrid[j + 1]);
                    fbank[i] = std::max(0.0f, std::min(upper, lower));
                    if(normalize)
                        fbank[i] *= (2.0f / (freqGrid[j + 2] - freqGrid[j]));
                }
            }
        }

		// Matrix Multiplication of Input and Filter Bank
        for (int i = 0; i < numFilter; i++)
        {
            for (int j = 0; j < numFrames; j++)
            {
                dstPtrTemp[i * numFrames + j] = 0.0f;
				for(int k = 0; k < numBins; k++)
				{
					dstPtrTemp[i * numFrames + j] += filterBanks[i][k] * srcPtrTemp[k * numFrames + j];
				}
            }
        }
    }

    return RPP_SUCCESS;
}
