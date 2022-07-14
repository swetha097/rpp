#include "rppdefs.h"

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

        // Avoid division by zero
        if(referenceMagnitude == 0.0)
            referenceMagnitude = 1.0;
    }

    // Calculate the intermediate values needed for DB conversion
    Rpp32f minRatio = pow(10, cutOffDB / multiplier);
    if(minRatio == 0.0f)
        minRatio = std::nextafter(0.0f, 1.0f);
    Rpp32f invReferenceMagnitude = 1.f / referenceMagnitude;

    for(int batchCount = 0; batchCount < batchSize; batchCount++)
    {
        Rpp32f magnitude = magnitudeTensor[batchCount];
        DBTensor[batchCount] = multiplier * log10(std::max(minRatio, magnitude * invReferenceMagnitude));
    }

    return RPP_SUCCESS;
}
