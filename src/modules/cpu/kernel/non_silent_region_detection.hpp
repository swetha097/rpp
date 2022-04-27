#include "rppdefs.h"
#include "cpu/rpp_cpu_simd.hpp"
#include "cpu/rpp_cpu_common.hpp"


Rpp64f Square(Rpp64f value) 
{
  return (value * value);
}

Rpp64f CalcSumSquared(Rpp64f *values, uint start, uint n) 
{
  Rpp64f sumsq = 0;
  for (uint i = start ; i < n ; i++) 
  {
    sumsq += Square(values[i]);
  }
  return sumsq;
}

Rpp64f max_value(std::vector<double> &values, uint length)
{
  Rpp64f max = values[0];
  for (uint i = 1; i < length; i++) 
  {
    max = std::max(max, values[i]);
  }
  return max;
}

RppStatus non_silent_region_detection_host_tensor(Rpp64f *srcPtr,
                                                  Rpp32u srcSize,
                                                  Rpp32u detectedIndex,
                                                  Rpp32u detectionLength,
                                                  Rpp64f cutOffDB,
                                                  Rpp32u windowLength,
                                                  Rpp64f referencePower,
                                                  Rpp32u resetInterval,
                                                  bool referenceMax)
{
    //set reset interval based on the user input 
    resetInterval = resetInterval == -1 ? srcSize : resetInterval;

    //Allocate intermediate buffer with given srcSize
    Rpp32u tempBufferSize = srcSize - windowLength + 1;
    std::vector<double> tempBuffer;
    tempBuffer.reserve(tempBufferSize);
    
    //Calculate moving mean square of input array and store in intermediate buffer
    Rpp64f sumsq = 0;
    Rpp64f meanFactor = (double)1.0 / windowLength;
    
    for (int window_begin = 0; window_begin <= srcSize - windowLength;) 
    {
        sumsq = CalcSumSquared(srcPtr, window_begin, windowLength);
        tempBuffer[window_begin] = sumsq * meanFactor;
        auto interval_end = std::min(window_begin + resetInterval, srcSize) - windowLength + 1;
        for (window_begin++; window_begin < interval_end; window_begin++) 
        {
            sumsq += Square(srcPtr[window_begin + windowLength - 1]) - Square(srcPtr[window_begin - 1]);
            tempBuffer[window_begin] = sumsq * meanFactor;
        }
    }
 
    //Convert cutOff from DB to magnitude    
    Rpp64f base = (referenceMax) ?  max_value(tempBuffer, tempBufferSize) : referencePower;
    Rpp64f cutOffMag = base * pow(10.0 , cutOffDB / 10.0);
    std::cout<<std::endl<<"cutOff magnitude: "<<cutOffMag;
      
    //Calculate beginning index, length of non silent region from the intermediate buffer
    int end = tempBufferSize;
    int begin = end;
    for (int i = 0; i < end; i++) 
    {
        if (tempBuffer[i] >= cutOffMag) 
        {
            begin = i;
            break;
        }
    }

    if (begin == end) 
    {
        detectedIndex = 0;
        detectionLength = 0;
        std::cout<<std::endl<<"Index, Length: "<<detectedIndex<<" "<<detectionLength;
        return RPP_SUCCESS;
    }

    for (int i = end - 1; i >= begin; i--) 
    {
        if (tempBuffer[i] >= cutOffMag) 
        {
            end = i;
            break;
        }
    }

    detectedIndex = begin;
    detectionLength = end - begin + 1;
    std::cout<<std::endl<<"Index, Length: "<<detectedIndex<<" "<<detectionLength;
    
    return RPP_SUCCESS;
}
