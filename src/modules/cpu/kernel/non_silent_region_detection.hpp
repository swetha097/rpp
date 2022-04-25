#include "rppdefs.h"
#include "cpu/rpp_cpu_simd.hpp"
#include "cpu/rpp_cpu_common.hpp"


Rpp32f Square(Rpp32f value) 
{
  return (value * value);
}

Rpp32f CalcSumSquared(Rpp32f *values, uint start, uint n) 
{
  Rpp32f sumsq = 0;
  for (uint i = start ; i < n ; i++) 
  {
    sumsq += Square(values[i]);
  }
  return sumsq;
}

Rpp32f max_value(Rpp32f *values, uint size)
{
  Rpp32f max = values[0];
  for (uint i = 1; i < size; i++) 
  {
    max = std::max(max, values[i]);
  }
  return max;
}

RppStatus non_silent_region_detection_host_tensor(Rpp32f *srcPtr,
                                                  Rpp32u srcSize,
                                                  Rpp32u detectedIndex,
                                                  Rpp32u detectionLength,
                                                  Rpp32f cutOffDB,
                                                  Rpp32u windowLength,
                                                  Rpp32f referencePower,
                                                  Rpp32u resetInterval,
                                                  bool referenceMax)
{
    //set reset interval based on the user input    
    resetInterval = resetInterval == -1 ? srcSize : resetInterval;

    //Allocate intermediate buffer with given srcSize
    Rpp32u tempBufferSize = srcSize - windowLength + 1;
    Rpp32f tempBuffer[tempBufferSize];   
    
    //Calculate moving mean square of input array and store in intermediate buffer
    Rpp32u sumsq = 0;
    Rpp32f meanFactor = 1.f / windowLength;
    
    for (int window_begin = 0; window_begin <= srcSize - windowLength;) 
    {
        sumsq = CalcSumSquared(srcPtr , window_begin, windowLength);
        tempBuffer[window_begin] = sumsq * meanFactor;
        auto interval_end = std::min(window_begin + resetInterval, srcSize) - windowLength + 1;
        for (window_begin++; window_begin < interval_end; window_begin++) 
        {
            sumsq += Square(srcPtr[window_begin + windowLength - 1]) - Square(srcPtr[window_begin - 1]);
            tempBuffer[window_begin] = sumsq * meanFactor;
        }
    }

    // std::cout<<"Printing the moving mean square values:"<<std::endl;
    // for(int i = 0; i < tempBufferSize; i++)
    // {
    //     std::cout<<tempBuffer[i]<<" ";
    // }
    
    //Convert cutOff from DB to magnitude
    Rpp32f base = (referenceMax) ?  max_value(tempBuffer, tempBufferSize) : referencePower;
    Rpp32f cutOffMag = base * pow(10.0 , cutOffDB / 10.0);
    
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
