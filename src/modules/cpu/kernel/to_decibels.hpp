#include "rppdefs.h"
#include "cpu/rpp_cpu_simd.hpp"
#include "cpu/rpp_cpu_common.hpp"

RppStatus to_decibels_host_tensor(Rpp32f *magnitudeTensor,
                                  Rpp32f *DBTensor,
                                  Rpp32u batchSize,
                                  Rpp32f cutOffDB,
                                  Rpp32f multiplier,
                                  Rpp32f referenceMagnitude)
{   
    bool referenceMax = (referenceMagnitude == 0.0) ? false : true;
    if(!referenceMax)
    {
        for(int i = 0; i < batchSize; i++)
        {
            if(magnitudeTensor[i] > referenceMagnitude)
                referenceMagnitude = magnitudeTensor[i];
        }

        //Avoid division by zero
        if(referenceMagnitude == 0.0)
            referenceMagnitude = 1.0;
    }

    for(int batchCount = 0; batchCount < batchSize; batchCount++)
    {
        Rpp32f magnitude = magnitudeTensor[batchCount];
        Rpp32f minRatio = pow(10, cutOffDB / multiplier);
        std::cout<<"minRatio: "<<minRatio<<std::endl;
        std::cout<<"magnitude: "<<magnitude<<std::endl;
        //Assert if minRatio < 0 - TODO
        DBTensor[batchCount] = multiplier * log10(std::max(minRatio, magnitude / referenceMagnitude));
    }
    return RPP_SUCCESS;
}