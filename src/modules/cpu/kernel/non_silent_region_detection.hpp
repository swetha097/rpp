#include "rppdefs.h"
#include "cpu/rpp_cpu_simd.hpp"
#include "cpu/rpp_cpu_common.hpp"


Rpp32f Square(Rpp32f &value) 
{
  Rpp32f res = value;
  return (res * res);
}

Rpp32f CalcSumSquared(Rpp32f *values, int start, int n) 
{
  Rpp32f sumOfSquares = 0.0;
  for (int i = start ; i < n ; i++) 
  {
    sumOfSquares += Square(values[i]);
  }
  return sumOfSquares;
}

Rpp32f max_value(std::vector<float> &values, int length)
{
  Rpp32f max = values[0];
  for (int i = 1; i < length; i++) 
  {
    max = std::max(max, values[i]);
  }
  return max;
}

void extend_non_silent_region(Rpp32s *detectionLength, Rpp32s *windowLength)
{
  if(*detectionLength != 0)
  {
   *detectionLength += *windowLength - 1;
  }
}

RppStatus non_silent_region_detection_host_tensor(Rpp32f *srcPtr,
                                                  RpptDescPtr srcDescPtr,
                                                  Rpp32s *srcSizeTensor,
                                                  Rpp32s *detectedIndexTensor,
                                                  Rpp32s *detectionLengthTensor,
                                                  Rpp32f *cutOffDBTensor,
                                                  Rpp32s *windowLengthTensor,
                                                  Rpp32f *referencePowerTensor,
                                                  Rpp32s *resetIntervalTensor,
                                                  bool *referenceMaxTensor)
{
#pragma omp parallel for num_threads(srcDescPtr->n)    
    for(int batchCount = 0; batchCount < srcDescPtr->n; batchCount++)
    {
      Rpp32s srcSize = srcSizeTensor[batchCount];
      Rpp32s windowLength = windowLengthTensor[batchCount];
      Rpp32f referencePower = referencePowerTensor[batchCount];
      Rpp32f cutOffDB = cutOffDBTensor[batchCount];
      bool referenceMax = referenceMaxTensor[batchCount];

      // for(int i = 0; i < srcSize; i++)
      // {
      //   std::cout<<srcPtr[i]<<" ";
      //   if (i % 1000 == 0)
      //   {
      //     std::cout<<std::endl;
      //   }
      // }

      //set reset interval based on the user input 
      Rpp32s resetInterval = resetIntervalTensor[batchCount];
      resetInterval = (resetInterval == -1) ? srcSize : resetInterval;

      //Calculate buffer size for mms array and allocate mms buffer
      Rpp32s mmsBufferSize = srcSize - windowLength + 1;
      std::vector<float> mmsBuffer;
      mmsBuffer.reserve(mmsBufferSize);
      
      //Calculate moving mean square of input array and store in mms buffer
      Rpp32f sumOfSquares = 0;
      Rpp32f meanFactor = 1.f / windowLength;
      for (int windowBegin = 0; windowBegin <= srcSize - windowLength;) 
      {
          sumOfSquares = CalcSumSquared(srcPtr, windowBegin, windowLength);
          mmsBuffer[windowBegin] = sumOfSquares * meanFactor;
          auto interval_endIdx = std::min(windowBegin + resetInterval, srcSize) - windowLength + 1;
          for (windowBegin++; windowBegin < interval_endIdx; windowBegin++) 
          {
              sumOfSquares += Square(srcPtr[windowBegin + windowLength - 1]) - Square(srcPtr[windowBegin - 1]);
              mmsBuffer[windowBegin] = sumOfSquares * meanFactor;
          }
      }
  
      //Convert cutOff from DB to magnitude    
      Rpp32f base = (referenceMax) ?  max_value(mmsBuffer, mmsBufferSize) : referencePower;
      Rpp32f cutOffMag = base * pow(float(10) , cutOffDB * float(1) / 10.f);
        
      //Calculate begining index, length of non silent region from the mms buffer
      int endIdx = mmsBufferSize;
      int beginIdx = endIdx;
      for (int i = 0; i < endIdx; i++) 
      {
          if (mmsBuffer[i] >= cutOffMag) 
          {
              beginIdx = i;
              break;
          }
      }

      if (beginIdx == endIdx) 
      {
          detectedIndexTensor[batchCount] = 0;
          detectionLengthTensor[batchCount] = 0;
      }
      else
      {        
        for (int i = endIdx - 1; i >= beginIdx; i--) 
        {
            if (mmsBuffer[i] >= cutOffMag) 
            {
                endIdx = i;
                break;
            }
        }
        detectedIndexTensor[batchCount] = beginIdx;
        detectionLengthTensor[batchCount] = endIdx - beginIdx + 1;
      }

      extend_non_silent_region(&detectionLengthTensor[batchCount], &windowLength);
    }
    return RPP_SUCCESS;
}
