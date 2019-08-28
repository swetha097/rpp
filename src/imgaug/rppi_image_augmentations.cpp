#include <rppi_image_augmentations.h>
#include <rppdefs.h>
#include "rppi_validate.hpp"

#ifdef HIP_COMPILE
#include <hip/rpp_hip_common.hpp>

#elif defined(OCL_COMPILE)
#include <cl/rpp_cl_common.hpp>
#include "cl/cl_declarations.hpp"
#endif //backend
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <chrono>
using namespace std::chrono;

#include "cpu/host_image_augmentations.hpp"

// ----------------------------------------
// Host blur functions calls
// ----------------------------------------


RppStatus
rppi_blur_u8_pln1_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f stdDev)
{

 	 validate_image_size(srcSize);
 	 validate_float_min(0, &stdDev);
	 unsigned int kernelSize = 3;
	 blur_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr),
			srcSize,
			static_cast<Rpp8u*>(dstPtr),
			stdDev,
			kernelSize,
			RPPI_CHN_PLANAR, 1);
	return RPP_SUCCESS;
}

RppStatus
rppi_blur_u8_pln3_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f stdDev)
{

 	 validate_image_size(srcSize);
 	 validate_float_min(0, &stdDev);
	 unsigned int kernelSize = 3;
	 blur_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr),
			srcSize,
			static_cast<Rpp8u*>(dstPtr),
			stdDev,
			kernelSize,
			RPPI_CHN_PLANAR, 3);
	return RPP_SUCCESS;
}

RppStatus
rppi_blur_u8_pkd3_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f stdDev)
{

 	 validate_image_size(srcSize);
 	 validate_float_min(0, &stdDev);
	 unsigned int kernelSize = 3;
	 blur_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr),
			srcSize,
			static_cast<Rpp8u*>(dstPtr),
			stdDev,
			kernelSize,
			RPPI_CHN_PACKED, 3);
	return RPP_SUCCESS;
}

// ----------------------------------------
// Host contrast functions calls
// ----------------------------------------


RppStatus
rppi_contrast_u8_pln1_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32u newMin,Rpp32u newMax)
{

 	 validate_image_size(srcSize);
 	 validate_unsigned_int_max(newMax, &newMin);
	 contrast_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr),
			srcSize,
			static_cast<Rpp8u*>(dstPtr),
			newMin,
			newMax,
			RPPI_CHN_PLANAR, 1);
	return RPP_SUCCESS;
}

RppStatus
rppi_contrast_u8_pln3_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32u newMin,Rpp32u newMax)
{

 	 validate_image_size(srcSize);
 	 validate_unsigned_int_max(newMax, &newMin);
	 contrast_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr),
			srcSize,
			static_cast<Rpp8u*>(dstPtr),
			newMin,
			newMax,
			RPPI_CHN_PLANAR, 3);
	return RPP_SUCCESS;
}

RppStatus
rppi_contrast_u8_pkd3_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32u newMin,Rpp32u newMax)
{

 	 validate_image_size(srcSize);
 	 validate_unsigned_int_max(newMax, &newMin);
	 contrast_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr),
			srcSize,
			static_cast<Rpp8u*>(dstPtr),
			newMin,
			newMax,
			RPPI_CHN_PACKED, 3);
	return RPP_SUCCESS;
}

// ----------------------------------------
// Host brightness functions calls
// ----------------------------------------


RppStatus
rppi_brightness_u8_pln1_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f alpha,Rpp32f beta)
{

 	 validate_image_size(srcSize);
 	 validate_float_range( 0, 2, &alpha);
 	 validate_float_range( 0, 255, &beta);
	 brightness_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr),
			srcSize,
			static_cast<Rpp8u*>(dstPtr),
			alpha,
			beta,
			RPPI_CHN_PLANAR, 1);
	return RPP_SUCCESS;
}

RppStatus
rppi_brightness_u8_pln3_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f alpha,Rpp32f beta)
{

 	 validate_image_size(srcSize);
 	 validate_float_range( 0, 2, &alpha);
 	 validate_float_range( 0, 255, &beta);
	 brightness_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr),
			srcSize,
			static_cast<Rpp8u*>(dstPtr),
			alpha,
			beta,
			RPPI_CHN_PLANAR, 3);
	return RPP_SUCCESS;
}

RppStatus
rppi_brightness_u8_pkd3_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f alpha,Rpp32f beta)
{

 	 validate_image_size(srcSize);
 	 validate_float_range( 0, 2, &alpha);
 	 validate_float_range( 0, 255, &beta);
	 brightness_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr),
			srcSize,
			static_cast<Rpp8u*>(dstPtr),
			alpha,
			beta,
			RPPI_CHN_PACKED, 3);
	return RPP_SUCCESS;
}

// ----------------------------------------
// Host gamma_correction functions calls
// ----------------------------------------


RppStatus
rppi_gamma_correction_u8_pln1_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f gamma)
{

 	 validate_image_size(srcSize);
 	 validate_float_min(0, &gamma);
	 gamma_correction_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr),
			srcSize,
			static_cast<Rpp8u*>(dstPtr),
			gamma,
			RPPI_CHN_PLANAR, 1);
	return RPP_SUCCESS;
}

RppStatus
rppi_gamma_correction_u8_pln3_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f gamma)
{

 	 validate_image_size(srcSize);
 	 validate_float_min(0, &gamma);
	 gamma_correction_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr),
			srcSize,
			static_cast<Rpp8u*>(dstPtr),
			gamma,
			RPPI_CHN_PLANAR, 3);
	return RPP_SUCCESS;
}

RppStatus
rppi_gamma_correction_u8_pkd3_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f gamma)
{

 	 validate_image_size(srcSize);
 	 validate_float_min(0, &gamma);
	 gamma_correction_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr),
			srcSize,
			static_cast<Rpp8u*>(dstPtr),
			gamma,
			RPPI_CHN_PACKED, 3);
	return RPP_SUCCESS;
}



// // ----------------------------------------
// Host jitter functions calls
// ----------------------------------------


RppStatus
rppi_jitter_u8_pln1_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32u kernelSize)
{
	validate_image_size(srcSize);
    jitter_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize, static_cast<Rpp8u*>(dstPtr),
                     kernelSize,
                     RPPI_CHN_PLANAR, 1);
    return RPP_SUCCESS;
}

RppStatus
rppi_jitter_u8_pln3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32u kernelSize)
{
	validate_image_size(srcSize);
    jitter_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize, static_cast<Rpp8u*>(dstPtr),
                     kernelSize,
                     RPPI_CHN_PLANAR, 3);
    return RPP_SUCCESS;
}

RppStatus
rppi_jitter_u8_pkd3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32u kernelSize)
{
	validate_image_size(srcSize);
    jitter_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize, static_cast<Rpp8u*>(dstPtr),
                     kernelSize,
                     RPPI_CHN_PACKED, 3);
    return RPP_SUCCESS;
}


RppStatus
rppi_snpNoise_u8_pln1_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32f noiseProbability)
{
	validate_image_size(srcSize);
 	validate_float_range( 0, 1, &noiseProbability);
    noise_snp_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize, static_cast<Rpp8u*>(dstPtr),
                        noiseProbability,
                        RPPI_CHN_PACKED, 1);
}

RppStatus
rppi_snpNoise_u8_pln3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32f noiseProbability)
{
	validate_image_size(srcSize);
 	validate_float_range( 0, 1, &noiseProbability);
    noise_snp_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize, static_cast<Rpp8u*>(dstPtr),
                        noiseProbability,
                        RPPI_CHN_PLANAR, 3);
}

RppStatus
rppi_snpNoise_u8_pkd3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32f noiseProbability)
{
	validate_image_size(srcSize);
 	validate_float_range( 0, 1, &noiseProbability);
    noise_snp_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize, static_cast<Rpp8u*>(dstPtr),
                        noiseProbability,
                        RPPI_CHN_PACKED, 3);
}

RppStatus
rppi_gaussianNoise_u8_pln1_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32f mean, Rpp32f sigma)
{
    noise_gaussian_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize, static_cast<Rpp8u*>(dstPtr),
                        mean, sigma,
                        RPPI_CHN_PLANAR, 1);
}

RppStatus
rppi_gaussianNoise_u8_pln3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32f mean, Rpp32f sigma)
{
    noise_gaussian_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize, static_cast<Rpp8u*>(dstPtr),
                        mean, sigma,
                        RPPI_CHN_PLANAR, 3);
}

RppStatus
rppi_gaussianNoise_u8_pkd3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32f mean, Rpp32f sigma)
{
    noise_gaussian_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize, static_cast<Rpp8u*>(dstPtr),
                        mean, sigma,
                        RPPI_CHN_PACKED, 3);
}

// ----------------------------------------
// Host fog functions call
// ----------------------------------------

RppStatus
rppi_fog_u8_pln1_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr, Rpp32f fogValue)
{
 	validate_float_range( 0, 1,&fogValue);
    validate_image_size(srcSize);
    Rpp32f stdDev=fogValue*50;
	unsigned int kernelSize = 5;
    if(fogValue!=0)
	blur_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr),
                srcSize,
                static_cast<Rpp8u*>(dstPtr),
                stdDev,
                kernelSize,
                RPPI_CHN_PLANAR, 1);

    fog_host<Rpp8u>(static_cast<Rpp8u*>(dstPtr),
			srcSize,
			fogValue,
			RPPI_CHN_PLANAR, 1, static_cast<Rpp8u*>(srcPtr) );

    return RPP_SUCCESS;
}

RppStatus
rppi_fog_u8_pln3_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr, Rpp32f fogValue)
{

    validate_float_range( 0, 1,&fogValue);
 	validate_image_size(srcSize);
    Rpp32f stdDev=fogValue*50;
	unsigned int kernelSize = 5;
    if(fogValue!=0)
	blur_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr),
			srcSize,
			static_cast<Rpp8u*>(dstPtr),
			stdDev,
			kernelSize,
			RPPI_CHN_PLANAR, 3);

    fog_host<Rpp8u>(static_cast<Rpp8u*>(dstPtr),
			srcSize,
			fogValue,
			RPPI_CHN_PLANAR, 3, static_cast<Rpp8u*>(srcPtr));
	return RPP_SUCCESS;
}

RppStatus
rppi_fog_u8_pkd3_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr, Rpp32f fogValue)
{
 	validate_float_range( 0, 1,&fogValue);
    validate_image_size(srcSize);
    Rpp32f stdDev=fogValue*10;
	unsigned int kernelSize = 5;
    if(fogValue!=0)
	blur_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr),
			srcSize,
			static_cast<Rpp8u*>(dstPtr),
			stdDev,
			kernelSize,
			RPPI_CHN_PACKED, 3);

    fog_host<Rpp8u>(static_cast<Rpp8u*>(dstPtr),
			srcSize,
			fogValue,
			RPPI_CHN_PACKED, 3, static_cast<Rpp8u*>(srcPtr));

    return RPP_SUCCESS;
}


// ----------------------------------------
// Host rain functions calls
// ----------------------------------------


RppStatus
rppi_rain_u8_pln1_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f rainPercentage,Rpp32u rainWidth,Rpp32u rainHeight, Rpp32f transparency)
{

 	 validate_image_size(srcSize);
 	 validate_float_range( 0, 1, &rainPercentage);
	 rain_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr),
			srcSize,
			static_cast<Rpp8u*>(dstPtr),
			rainPercentage,
			rainWidth,
			rainHeight, transparency,
			RPPI_CHN_PLANAR, 1);
	return RPP_SUCCESS;
}

RppStatus
rppi_rain_u8_pln3_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f rainPercentage,Rpp32u rainWidth,Rpp32u rainHeight, Rpp32f transparency)
{

 	 validate_image_size(srcSize);
 	 validate_float_range( 0, 1, &rainPercentage);
	 rain_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr),
			srcSize,
			static_cast<Rpp8u*>(dstPtr),
			rainPercentage,
			rainWidth,
			rainHeight, transparency,
			RPPI_CHN_PLANAR, 3);
	return RPP_SUCCESS;
}

RppStatus
rppi_rain_u8_pkd3_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f rainPercentage,Rpp32u rainWidth,Rpp32u rainHeight, Rpp32f transparency)
{

 	 validate_image_size(srcSize);
 	 validate_float_range( 0, 1, &rainPercentage);
	 rain_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr),
			srcSize,
			static_cast<Rpp8u*>(dstPtr),
			rainPercentage,
			rainWidth,
			rainHeight, transparency,
			RPPI_CHN_PACKED, 3);
	return RPP_SUCCESS;
}

// ----------------------------------------
// Host snow functions calls
// ----------------------------------------


RppStatus
rppi_snow_u8_pln1_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f snowValue)
{

 	 validate_image_size(srcSize);
 	 validate_float_range( 0, 1,&snowValue);
	 snow_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr),
			srcSize,
			static_cast<Rpp8u*>(dstPtr),
			snowValue,
			RPPI_CHN_PLANAR, 1);
	return RPP_SUCCESS;
}

RppStatus
rppi_snow_u8_pln3_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f snowValue)
{

 	 validate_image_size(srcSize);
 	 validate_float_range( 0, 1,&snowValue);
	 snow_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr),
			srcSize,
			static_cast<Rpp8u*>(dstPtr),
			snowValue,
			RPPI_CHN_PLANAR, 3);
	return RPP_SUCCESS;
}

RppStatus
rppi_snow_u8_pkd3_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f snowValue)
{

 	 validate_image_size(srcSize);
 	 validate_float_range( 0, 1,&snowValue);
	 snow_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr),
			srcSize,
			static_cast<Rpp8u*>(dstPtr),
			snowValue,
			RPPI_CHN_PACKED, 3);
	return RPP_SUCCESS;
}

// ----------------------------------------
// Host random_shadow functions calls
// ----------------------------------------


RppStatus
rppi_random_shadow_u8_pln1_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32u x1 ,Rpp32u y1,Rpp32u x2,Rpp32u y2,Rpp32u numberOfShadows,Rpp32u maxSizeX,Rpp32u maxSizeY)
{

 	 validate_image_size(srcSize);
 	 validate_unsigned_int_range( 0, srcSize.width - 1,& x1);
 	 validate_unsigned_int_range( 0, srcSize.height - 1, &y1);
 	 validate_unsigned_int_range( 0, srcSize.width - 1, &x2);
 	 validate_unsigned_int_range( 0, srcSize.height - 1, &y2);
 	 validate_unsigned_int_min(1, &numberOfShadows);
	 validate_unsigned_int_max(x2,&x1);
	 validate_unsigned_int_max(y2,&y1);
	 random_shadow_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr),
			srcSize,
			static_cast<Rpp8u*>(dstPtr),
			x1,
			y1,
			x2,
			y2,
			numberOfShadows,
			maxSizeX,
			maxSizeY,
			RPPI_CHN_PLANAR, 1);
	return RPP_SUCCESS;
}

RppStatus
rppi_random_shadow_u8_pln3_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32u x1 ,Rpp32u y1,Rpp32u x2,Rpp32u y2,Rpp32u numberOfShadows,Rpp32u maxSizeX,Rpp32u maxSizeY)
{

 	 validate_image_size(srcSize);
 	 validate_unsigned_int_range( 0, srcSize.width - 1,& x1);
 	 validate_unsigned_int_range( 0, srcSize.height - 1, &y1);
 	 validate_unsigned_int_range( 0, srcSize.width - 1, &x2);
 	 validate_unsigned_int_range( 0, srcSize.height - 1, &y2);
 	 validate_unsigned_int_min(1, &numberOfShadows);
	 validate_unsigned_int_max(x2,&x1);
	 validate_unsigned_int_max(y2,&y1);
	 random_shadow_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr),
			srcSize,
			static_cast<Rpp8u*>(dstPtr),
			x1,
			y1,
			x2,
			y2,
			numberOfShadows,
			maxSizeX,
			maxSizeY,
			RPPI_CHN_PLANAR, 3);
	return RPP_SUCCESS;
}

RppStatus
rppi_random_shadow_u8_pkd3_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32u x1,Rpp32u y1,Rpp32u x2,Rpp32u y2,Rpp32u numberOfShadows,Rpp32u maxSizeX,Rpp32u maxSizeY)
{

 	 validate_image_size(srcSize);
 	 validate_unsigned_int_range( 0, srcSize.width - 1,& x1);
 	 validate_unsigned_int_range( 0, srcSize.height - 1, &y1);
 	 validate_unsigned_int_range( 0, srcSize.width - 1, &x2);
 	 validate_unsigned_int_range( 0, srcSize.height - 1, &y2);
 	 validate_unsigned_int_min(1, &numberOfShadows);
	 validate_unsigned_int_max(x2,&x1);
	 validate_unsigned_int_max(y2,&y1);
	 random_shadow_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr),
			srcSize,
			static_cast<Rpp8u*>(dstPtr),
			x1,
			y1,
			x2,
			y2,
			numberOfShadows,
			maxSizeX,
			maxSizeY,
			RPPI_CHN_PACKED, 3);
	return RPP_SUCCESS;
}

// ----------------------------------------
// Host blend functions calls
// ----------------------------------------


RppStatus
rppi_blend_u8_pln1_host(RppPtr_t srcPtr1,RppPtr_t srcPtr2,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f alpha)
{

 	 validate_image_size(srcSize);
 	 validate_float_range( 0, 1, &alpha);
	 blend_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr1),
			static_cast<Rpp8u*>(srcPtr2),
			srcSize,
			static_cast<Rpp8u*>(dstPtr),
			alpha,
			RPPI_CHN_PLANAR, 1);
	return RPP_SUCCESS;
}

RppStatus
rppi_blend_u8_pln3_host(RppPtr_t srcPtr1,RppPtr_t srcPtr2,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f alpha)
{

 	 validate_image_size(srcSize);
 	 validate_float_range( 0, 1, &alpha);
	 blend_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr1),
			static_cast<Rpp8u*>(srcPtr2),
			srcSize,
			static_cast<Rpp8u*>(dstPtr),
			alpha,
			RPPI_CHN_PLANAR, 3);
	return RPP_SUCCESS;
}

RppStatus
rppi_blend_u8_pkd3_host(RppPtr_t srcPtr1,RppPtr_t srcPtr2,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f alpha)
{

 	 validate_image_size(srcSize);
 	 validate_float_range( 0, 1, &alpha);
	 blend_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr1),
			static_cast<Rpp8u*>(srcPtr2),
			srcSize,
			static_cast<Rpp8u*>(dstPtr),
			alpha,
			RPPI_CHN_PACKED, 3);
	return RPP_SUCCESS;
}

// ----------------------------------------
// Host pixelate functions calls
// ----------------------------------------


RppStatus
rppi_pixelate_u8_pln1_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr, Rpp32u x1,Rpp32u y1,Rpp32u x2,Rpp32u y2 )
{

 	 validate_image_size(srcSize);
 	 validate_unsigned_int_range( 0, srcSize.width - 1,& x1);
 	 validate_unsigned_int_range( 0, srcSize.height - 1, &y1);
 	 validate_unsigned_int_range( 0, srcSize.width - 1, &x2);
 	 validate_unsigned_int_range( 0, srcSize.height - 1, &y2);
	 Rpp32u kernelSize = 3;
	 validate_unsigned_int_max(x2,&x1);
	 validate_unsigned_int_max(y2,&y1);
	 pixelate_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr),
			srcSize,
			static_cast<Rpp8u*>(dstPtr),
			kernelSize,
			x1,
			y1,
			x2,
			y2,
			RPPI_CHN_PLANAR, 1);
	return RPP_SUCCESS;
}

RppStatus
rppi_pixelate_u8_pln3_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32u x1 ,Rpp32u y1,Rpp32u x2,Rpp32u y2 )
{

 	 validate_image_size(srcSize);
 	 validate_unsigned_int_range( 0, srcSize.width - 1,& x1);
 	 validate_unsigned_int_range( 0, srcSize.height - 1, &y1);
 	 validate_unsigned_int_range( 0, srcSize.width - 1, &x2);
 	 validate_unsigned_int_range( 0, srcSize.height - 1, &y2);
	 Rpp32u kernelSize = 3;
	 validate_unsigned_int_max(x2,&x1);
	 validate_unsigned_int_max(y2,&y1);
	 pixelate_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr),
			srcSize,
			static_cast<Rpp8u*>(dstPtr),
			kernelSize,
			x1,
			y1,
			x2,
			y2,
			RPPI_CHN_PLANAR, 3);
	return RPP_SUCCESS;
}

RppStatus
rppi_pixelate_u8_pkd3_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32u x1 ,Rpp32u y1,Rpp32u x2,Rpp32u y2 )
{

 	 validate_image_size(srcSize);
 	 validate_unsigned_int_range( 0, srcSize.width - 1,& x1);
 	 validate_unsigned_int_range( 0, srcSize.height - 1, &y1);
 	 validate_unsigned_int_range( 0, srcSize.width - 1, &x2);
 	 validate_unsigned_int_range( 0, srcSize.height - 1, &y2);
	 Rpp32u kernelSize = 3;
	 validate_unsigned_int_max(x2,&x1);
	 validate_unsigned_int_max(y2,&y1);
	 pixelate_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr),
			srcSize,
			static_cast<Rpp8u*>(dstPtr),
			kernelSize,
			x1,
			y1,
			x2,
			y2,
			RPPI_CHN_PACKED, 3);
	return RPP_SUCCESS;
}

// ----------------------------------------
// Host random_crop_letterbox functions calls
// ----------------------------------------


RppStatus
rppi_random_crop_letterbox_u8_pln1_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,RppiSize dstSize,Rpp32u x1,Rpp32u y1,Rpp32u x2,Rpp32u y2)
{

 	 validate_image_size(srcSize);
 	 validate_image_size(dstSize);
 	 validate_unsigned_int_range( 0, srcSize.width - 1,& x1);
 	 validate_unsigned_int_range( 0, srcSize.height - 1, &y1);
 	 validate_unsigned_int_range( 0, srcSize.width - 1, &x2);
 	 validate_unsigned_int_range( 0, srcSize.height - 1, &y2);
	 validate_unsigned_int_max(x2,&x1);
	 validate_unsigned_int_max(y2,&y1);
	 random_crop_letterbox_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr),
			srcSize,
			static_cast<Rpp8u*>(dstPtr),
			dstSize,
			x1,
			y1,
			x2,
			y2,
			RPPI_CHN_PLANAR, 1);
	return RPP_SUCCESS;
}

RppStatus
rppi_random_crop_letterbox_u8_pln3_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,RppiSize dstSize,Rpp32u x1,Rpp32u y1,Rpp32u x2,Rpp32u y2)
{

 	 validate_image_size(srcSize);
 	 validate_image_size(dstSize);
 	 validate_unsigned_int_range( 0, srcSize.width - 1,& x1);
 	 validate_unsigned_int_range( 0, srcSize.height - 1, &y1);
 	 validate_unsigned_int_range( 0, srcSize.width - 1, &x2);
 	 validate_unsigned_int_range( 0, srcSize.height - 1, &y2);
	 validate_unsigned_int_max(x2,&x1);
	 validate_unsigned_int_max(y2,&y1);
	 random_crop_letterbox_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr),
			srcSize,
			static_cast<Rpp8u*>(dstPtr),
			dstSize,
			x1,
			y1,
			x2,
			y2,
			RPPI_CHN_PLANAR, 3);
	return RPP_SUCCESS;
}

RppStatus
rppi_random_crop_letterbox_u8_pkd3_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,RppiSize dstSize,Rpp32u x1,Rpp32u y1,Rpp32u x2,Rpp32u y2)
{

 	 validate_image_size(srcSize);
 	 validate_image_size(dstSize);
 	 validate_unsigned_int_range( 0, srcSize.width - 1,& x1);
 	 validate_unsigned_int_range( 0, srcSize.height - 1, &y1);
 	 validate_unsigned_int_range( 0, srcSize.width - 1, &x2);
 	 validate_unsigned_int_range( 0, srcSize.height - 1, &y2);
	 validate_unsigned_int_max(x2,&x1);
	 validate_unsigned_int_max(y2,&y1);
	 random_crop_letterbox_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr),
			srcSize,
			static_cast<Rpp8u*>(dstPtr),
			dstSize,
			x1,
			y1,
			x2,
			y2,
			RPPI_CHN_PACKED, 3);
	return RPP_SUCCESS;
}

// ----------------------------------------
// Host occlusion functions calls
// ----------------------------------------


RppStatus
rppi_occlusion_u8_pln1_host(RppPtr_t srcPtr1,RppiSize srcSize1,RppPtr_t srcPtr2,RppiSize srcSize2,RppPtr_t dstPtr,Rpp32u src1x1,Rpp32u src1y1,Rpp32u src1x2,Rpp32u src1y2,Rpp32u src2x1,Rpp32u src2y1,Rpp32u src2x2,Rpp32u src2y2)
{

 	 validate_image_size(srcSize1);
 	 validate_image_size(srcSize2);
 	 validate_unsigned_int_range( 0, srcSize1.width - 1, &src1x1);
 	 validate_unsigned_int_range( 0, srcSize1.height - 1, &src1y1);
 	 validate_unsigned_int_range( 0, srcSize1.width - 1, &src1x2);
 	 validate_unsigned_int_range( 0, srcSize1.height - 1, &src1y2);
 	 validate_unsigned_int_range( 0, srcSize1.width - 1, &src2x1);
 	 validate_unsigned_int_range( 0, srcSize1.height - 1, &src2y1);
 	 validate_unsigned_int_range( 0, srcSize1.width - 1, &src2x2);
 	 validate_unsigned_int_range( 0, srcSize1.height - 1,&src2y2);
	 validate_unsigned_int_max(src1x2,&src1x1);
	 validate_unsigned_int_max(src1y2,&src1y1);
	 validate_unsigned_int_max(src2x2,&src2x1);
	 validate_unsigned_int_max(src2y2,&src2y1);
	 occlusion_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr1),
			srcSize1,
			static_cast<Rpp8u*>(srcPtr2),
			srcSize2,
			static_cast<Rpp8u*>(dstPtr),
			src1x1,
			src1y1,
			src1x2,
			src1y2,
			src2x1,
			src2y1,
			src2x2,
			src2y2,
			RPPI_CHN_PLANAR, 1);
	return RPP_SUCCESS;
}

RppStatus
rppi_occlusion_u8_pln3_host(RppPtr_t srcPtr1,RppiSize srcSize1,RppPtr_t srcPtr2,RppiSize srcSize2,RppPtr_t dstPtr,Rpp32u src1x1,Rpp32u src1y1,Rpp32u src1x2,Rpp32u src1y2,Rpp32u src2x1,Rpp32u src2y1,Rpp32u src2x2,Rpp32u src2y2)
{

 	 validate_image_size(srcSize1);
 	 validate_image_size(srcSize2);
 	 validate_unsigned_int_range( 0, srcSize1.width - 1, &src1x1);
 	 validate_unsigned_int_range( 0, srcSize1.height - 1, &src1y1);
 	 validate_unsigned_int_range( 0, srcSize1.width - 1, &src1x2);
 	 validate_unsigned_int_range( 0, srcSize1.height - 1, &src1y2);
 	 validate_unsigned_int_range( 0, srcSize1.width - 1, &src2x1);
 	 validate_unsigned_int_range( 0, srcSize1.height - 1, &src2y1);
 	 validate_unsigned_int_range( 0, srcSize1.width - 1, &src2x2);
 	 validate_unsigned_int_range( 0, srcSize1.height - 1,&src2y2);
	 validate_unsigned_int_max(src1x2,&src1x1);
	 validate_unsigned_int_max(src1y2,&src1y1);
	 validate_unsigned_int_max(src2x2,&src2x1);
	 validate_unsigned_int_max(src2y2,&src2y1);
	 occlusion_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr1),
			srcSize1,
			static_cast<Rpp8u*>(srcPtr2),
			srcSize2,
			static_cast<Rpp8u*>(dstPtr),
			src1x1,
			src1y1,
			src1x2,
			src1y2,
			src2x1,
			src2y1,
			src2x2,
			src2y2,
			RPPI_CHN_PLANAR, 3);
	return RPP_SUCCESS;
}

RppStatus
rppi_occlusion_u8_pkd3_host(RppPtr_t srcPtr1,RppiSize srcSize1,RppPtr_t srcPtr2,RppiSize srcSize2,RppPtr_t dstPtr,Rpp32u src1x1,Rpp32u src1y1,Rpp32u src1x2,Rpp32u src1y2,Rpp32u src2x1,Rpp32u src2y1,Rpp32u src2x2,Rpp32u src2y2)
{

 	 validate_image_size(srcSize1);
 	 validate_image_size(srcSize2);
 	 validate_unsigned_int_range( 0, srcSize1.width - 1, &src1x1);
 	 validate_unsigned_int_range( 0, srcSize1.height - 1, &src1y1);
 	 validate_unsigned_int_range( 0, srcSize1.width - 1, &src1x2);
 	 validate_unsigned_int_range( 0, srcSize1.height - 1, &src1y2);
 	 validate_unsigned_int_range( 0, srcSize1.width - 1, &src2x1);
 	 validate_unsigned_int_range( 0, srcSize1.height - 1, &src2y1);
 	 validate_unsigned_int_range( 0, srcSize1.width - 1, &src2x2);
 	 validate_unsigned_int_range( 0, srcSize1.height - 1,&src2y2);
	 validate_unsigned_int_max(src1x2,&src1x1);
	 validate_unsigned_int_max(src1y2,&src1y1);
	 validate_unsigned_int_max(src2x2,&src2x1);
	 validate_unsigned_int_max(src2y2,&src2y1);
	 occlusion_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr1),
			srcSize1,
			static_cast<Rpp8u*>(srcPtr2),
			srcSize2,
			static_cast<Rpp8u*>(dstPtr),
			src1x1,
			src1y1,
			src1x2,
			src1y2,
			src2x1,
			src2y1,
			src2x2,
			src2y2,
			RPPI_CHN_PACKED, 3);
	return RPP_SUCCESS;
}

// ----------------------------------------
// Host exposure functions calls
// ----------------------------------------

RppStatus
rppi_exposure_u8_pln1_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f exposureValue)
{

 	 validate_image_size(srcSize);
 	 validate_float_range( -4, 4, &exposureValue);
	 exposure_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr),
			srcSize,
			static_cast<Rpp8u*>(dstPtr),
			exposureValue,
			RPPI_CHN_PLANAR, 1);
	return RPP_SUCCESS;
}

RppStatus
rppi_exposure_u8_pln3_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f exposureValue)
{

 	 validate_image_size(srcSize);
 	 validate_float_range( -4, 4, &exposureValue);
	 exposure_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr),
			srcSize,
			static_cast<Rpp8u*>(dstPtr),
			exposureValue,
			RPPI_CHN_PLANAR, 3);
	return RPP_SUCCESS;
}

RppStatus
rppi_exposure_u8_pkd3_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f exposureValue)
{

 	 validate_image_size(srcSize);
 	 validate_float_range( -4, 4, &exposureValue);
	 exposure_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr),
			srcSize,
			static_cast<Rpp8u*>(dstPtr),
			exposureValue,
			RPPI_CHN_PACKED, 3);
	return RPP_SUCCESS;
}


 
// ----------------------------------------
// Host histogram_balance functions calls 
// ----------------------------------------


RppStatus
rppi_histogram_balance_u8_pln1_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr)
{
    histogram_balance_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize, static_cast<Rpp8u*>(dstPtr), 
                       RPPI_CHN_PLANAR,1);

    return RPP_SUCCESS;

}

RppStatus
rppi_histogram_balance_u8_pln3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr)
{
    histogram_balance_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize, static_cast<Rpp8u*>(dstPtr), 
                       RPPI_CHN_PLANAR,3);

    return RPP_SUCCESS;

}

RppStatus
rppi_histogram_balance_u8_pkd3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr)
{
    histogram_balance_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize, static_cast<Rpp8u*>(dstPtr), 
                       RPPI_CHN_PACKED,3);

    return RPP_SUCCESS;

}

// ----------------------------------------
// Host histogram_equalize functions calls 
// ----------------------------------------


RppStatus
rppi_histogram_equalize_u8_pln1_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr)
{

 	 validate_image_size(srcSize);
	 histogram_balance_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), 
			srcSize,
			static_cast<Rpp8u*>(dstPtr), 
			RPPI_CHN_PLANAR, 1);
	return RPP_SUCCESS;
}

RppStatus
rppi_histogram_equalize_u8_pln3_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr)
{

 	 validate_image_size(srcSize);
	 histogram_balance_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), 
			srcSize,
			static_cast<Rpp8u*>(dstPtr), 
			RPPI_CHN_PLANAR, 3);
	return RPP_SUCCESS;
}

RppStatus
rppi_histogram_equalize_u8_pkd3_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr)
{

 	 validate_image_size(srcSize);
	 histogram_balance_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), 
			srcSize,
			static_cast<Rpp8u*>(dstPtr), 
			RPPI_CHN_PACKED, 3);
	return RPP_SUCCESS;
}

// ----------------------------------------
// GPU blur functions  calls
// ----------------------------------------


RppStatus
rppi_blur_u8_pln1_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f stdDev, RppHandle_t rppHandle)
{

 	 validate_image_size(srcSize);
 	 validate_float_min(0, &stdDev);
	 unsigned int kernelSize = 3;
#ifdef OCL_COMPILE
 	 {
 	 blur_cl(static_cast<cl_mem>(srcPtr),
			srcSize,
			static_cast<cl_mem>(dstPtr),
			kernelSize,
			RPPI_CHN_PLANAR, 1,
			static_cast<cl_command_queue>(rppHandle));
 	 }
#elif defined (HIP_COMPILE)
 	 {
 	 }
#endif //BACKEND
		return RPP_SUCCESS;
}

RppStatus
rppi_blur_u8_pln3_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f stdDev, RppHandle_t rppHandle)
{

 	 validate_image_size(srcSize);
 	 validate_float_min(0, &stdDev);
	 unsigned int kernelSize = 3;
#ifdef OCL_COMPILE
 	 {
 	 blur_cl(static_cast<cl_mem>(srcPtr),
			srcSize,
			static_cast<cl_mem>(dstPtr),
			kernelSize,
			RPPI_CHN_PLANAR, 3,
			static_cast<cl_command_queue>(rppHandle));
 	 }
#elif defined (HIP_COMPILE)
 	 {
 	 }
#endif //BACKEND
		return RPP_SUCCESS;
}

RppStatus
rppi_blur_u8_pkd3_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f stdDev, RppHandle_t rppHandle)
{

 	 validate_image_size(srcSize);
 	 validate_float_min(0, &stdDev);
	 unsigned int kernelSize = 3;

#ifdef OCL_COMPILE
 	 {
 	 blur_cl(static_cast<cl_mem>(srcPtr),
			srcSize,
			static_cast<cl_mem>(dstPtr),
			kernelSize,
			RPPI_CHN_PACKED, 3,
			static_cast<cl_command_queue>(rppHandle));
 	 }
#elif defined (HIP_COMPILE)
 	 {
 	 }
#endif //BACKEND
		return RPP_SUCCESS;
}

// ----------------------------------------
// GPU contrast functions  calls
// ----------------------------------------


RppStatus
rppi_contrast_u8_pln1_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32u newMin,Rpp32u newMax, RppHandle_t rppHandle)
{

 	 validate_image_size(srcSize);
 	 validate_unsigned_int_max(newMax, &newMin);

#ifdef OCL_COMPILE
 	 {
 	 contrast_cl(static_cast<cl_mem>(srcPtr),
			srcSize,
			static_cast<cl_mem>(dstPtr),
			newMin,
			newMax,
			RPPI_CHN_PLANAR, 1,
			static_cast<cl_command_queue>(rppHandle));
 	 }
#elif defined (HIP_COMPILE)
 	 {
 	 }
#endif //BACKEND
		return RPP_SUCCESS;
}

RppStatus
rppi_contrast_u8_pln3_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32u newMin,Rpp32u newMax, RppHandle_t rppHandle)
{

 	 validate_image_size(srcSize);
 	 validate_unsigned_int_max(newMax, &newMin);

#ifdef OCL_COMPILE
 	 {
 	 contrast_cl(static_cast<cl_mem>(srcPtr),
			srcSize,
			static_cast<cl_mem>(dstPtr),
			newMin,
			newMax,
			RPPI_CHN_PLANAR, 3,
			static_cast<cl_command_queue>(rppHandle));
 	 }
#elif defined (HIP_COMPILE)
 	 {
 	 }
#endif //BACKEND
		return RPP_SUCCESS;
}

RppStatus
rppi_contrast_u8_pkd3_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32u newMin,Rpp32u newMax, RppHandle_t rppHandle)
{

 	 validate_image_size(srcSize);
 	 validate_unsigned_int_max(newMax, &newMin);

#ifdef OCL_COMPILE
 	 {
 	 contrast_cl(static_cast<cl_mem>(srcPtr),
			srcSize,
			static_cast<cl_mem>(dstPtr),
			newMin,
			newMax,
			RPPI_CHN_PACKED, 3,
			static_cast<cl_command_queue>(rppHandle));
 	 }
#elif defined (HIP_COMPILE)
 	 {
 	 }
#endif //BACKEND
		return RPP_SUCCESS;
}

// ----------------------------------------
// GPU brightness functions  calls
// ----------------------------------------


RppStatus
rppi_brightness_u8_pln1_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f alpha,Rpp32f beta, RppHandle_t rppHandle)
{

 	 validate_image_size(srcSize);
 	 validate_float_range_b( 0, 2, &alpha);
 	 validate_float_range( 0, 255, &beta);

#ifdef OCL_COMPILE
 	 {
 	 brightness_cl(static_cast<cl_mem>(srcPtr),
			srcSize,
			static_cast<cl_mem>(dstPtr),
			alpha,
			beta,
			RPPI_CHN_PLANAR, 1,
			static_cast<cl_command_queue>(rppHandle));
 	 }
#elif defined (HIP_COMPILE)
 	 {
 	 }
#endif //BACKEND
		return RPP_SUCCESS;
}

RppStatus
rppi_brightness_u8_pln3_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f alpha,Rpp32f beta, RppHandle_t rppHandle)
{

 	 validate_image_size(srcSize);
 	 validate_float_range( 0, 2, &alpha);
 	 validate_float_range( 0, 255, &beta);

#ifdef OCL_COMPILE
 	 {
 	 brightness_cl(static_cast<cl_mem>(srcPtr),
			srcSize,
			static_cast<cl_mem>(dstPtr),
			alpha,
			beta,
			RPPI_CHN_PLANAR, 3,
			static_cast<cl_command_queue>(rppHandle));
 	 }
#elif defined (HIP_COMPILE)
 	 {
 	 }
#endif //BACKEND
		return RPP_SUCCESS;
}

RppStatus
rppi_brightness_u8_pkd3_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f alpha,Rpp32f beta, RppHandle_t rppHandle)
{

 	 validate_image_size(srcSize);
 	 validate_float_range( 0, 2, &alpha);
 	 validate_float_range( 0, 255, &beta);

#ifdef OCL_COMPILE
 	 {
 	 brightness_cl(static_cast<cl_mem>(srcPtr),
			srcSize,
			static_cast<cl_mem>(dstPtr),
			alpha,
			beta,
			RPPI_CHN_PACKED, 3,
			static_cast<cl_command_queue>(rppHandle));
 	 }
#elif defined (HIP_COMPILE)
 	 {
 	 }
#endif //BACKEND
		return RPP_SUCCESS;
}

// ----------------------------------------
// GPU gamma_correction functions  calls
// ----------------------------------------


RppStatus
rppi_gamma_correction_u8_pln1_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f gamma, RppHandle_t rppHandle)
{

 	 validate_image_size(srcSize);
 	 validate_float_min(0, &gamma);

#ifdef OCL_COMPILE
 	 {
 	 gamma_correction_cl(static_cast<cl_mem>(srcPtr),
			srcSize,
			static_cast<cl_mem>(dstPtr),
			gamma,
			RPPI_CHN_PLANAR, 1,
			static_cast<cl_command_queue>(rppHandle));
 	 }
#elif defined (HIP_COMPILE)
 	 {
 	 }
#endif //BACKEND
		return RPP_SUCCESS;
}

RppStatus
rppi_gamma_correction_u8_pln3_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f gamma, RppHandle_t rppHandle)
{

 	 validate_image_size(srcSize);
 	 validate_float_min(0, &gamma);

#ifdef OCL_COMPILE
 	 {
 	 gamma_correction_cl(static_cast<cl_mem>(srcPtr),
			srcSize,
			static_cast<cl_mem>(dstPtr),
			gamma,
			RPPI_CHN_PLANAR, 3,
			static_cast<cl_command_queue>(rppHandle));
 	 }
#elif defined (HIP_COMPILE)
 	 {
 	 }
#endif //BACKEND
		return RPP_SUCCESS;
}

RppStatus
rppi_gamma_correction_u8_pkd3_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f gamma, RppHandle_t rppHandle)
{
   	 validate_image_size(srcSize);
 	 validate_float_min(0, &gamma);

#ifdef OCL_COMPILE
 	 {
 	 gamma_correction_cl(static_cast<cl_mem>(srcPtr),
			srcSize,
			static_cast<cl_mem>(dstPtr),
			gamma,
			RPPI_CHN_PACKED, 3,
			static_cast<cl_command_queue>(rppHandle));
 	 }
#elif defined (HIP_COMPILE)
 	 {
 	 }
#endif //BACKEND
		return RPP_SUCCESS;
}

// ----------------------------------------
// GPU rain functions  calls
// ----------------------------------------


RppStatus
rppi_rain_u8_pln1_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f rainPercentage,Rpp32u rainWidth,Rpp32u rainHeight, Rpp32f transparency, RppHandle_t rppHandle)
{

 	 validate_image_size(srcSize);
 	 validate_float_range( 0, 1, &rainPercentage);

#ifdef OCL_COMPILE
 	 {
 	 rain_cl(static_cast<cl_mem>(srcPtr),
			srcSize,
			static_cast<cl_mem>(dstPtr),
			rainPercentage,
			rainWidth,
			rainHeight,
            transparency,
			RPPI_CHN_PLANAR, 1,
			static_cast<cl_command_queue>(rppHandle));
 	 }
#elif defined (HIP_COMPILE)
 	 {
 	 }
#endif //BACKEND
		return RPP_SUCCESS;
}

RppStatus
rppi_rain_u8_pln3_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f rainPercentage,Rpp32u rainWidth,Rpp32u rainHeight, Rpp32f transparency, RppHandle_t rppHandle)
{

 	 validate_image_size(srcSize);
 	 validate_float_range( 0, 1, &rainPercentage);

#ifdef OCL_COMPILE
 	 {
 	 rain_cl(static_cast<cl_mem>(srcPtr),
			srcSize,
			static_cast<cl_mem>(dstPtr),
			rainPercentage,
			rainWidth,
			rainHeight,
            transparency,
			RPPI_CHN_PLANAR, 3,
			static_cast<cl_command_queue>(rppHandle));
 	 }
#elif defined (HIP_COMPILE)
 	 {
 	 }
#endif //BACKEND
		return RPP_SUCCESS;
}

RppStatus
rppi_rain_u8_pkd3_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f rainPercentage,Rpp32u rainWidth,Rpp32u rainHeight, Rpp32f transparency, RppHandle_t rppHandle)
{

 	 validate_image_size(srcSize);
 	 validate_float_range( 0, 1, &rainPercentage);

#ifdef OCL_COMPILE
 	 {
 	 rain_cl(static_cast<cl_mem>(srcPtr),
			srcSize,
			static_cast<cl_mem>(dstPtr),
			rainPercentage,
			rainWidth,
			rainHeight,
            transparency,
			RPPI_CHN_PACKED, 3,
			static_cast<cl_command_queue>(rppHandle));
 	 }
#elif defined (HIP_COMPILE)
 	 {
 	 }
#endif //BACKEND
		return RPP_SUCCESS;
}

// ----------------------------------------
// GPU snow functions  calls
// ----------------------------------------


RppStatus
rppi_snow_u8_pln1_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f snowValue, RppHandle_t rppHandle)
{

 	 validate_image_size(srcSize);
 	 validate_float_range( 0, 1, &snowValue);

#ifdef OCL_COMPILE
 	 {
 	 snow_cl(static_cast<cl_mem>(srcPtr),
			srcSize,
			static_cast<cl_mem>(dstPtr),
			snowValue,
			RPPI_CHN_PLANAR, 1,
			static_cast<cl_command_queue>(rppHandle));
 	 }
#elif defined (HIP_COMPILE)
 	 {
 	 }
#endif //BACKEND
		return RPP_SUCCESS;
}

RppStatus
rppi_snow_u8_pln3_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f snowValue, RppHandle_t rppHandle)
{

 	 validate_image_size(srcSize);
 	 validate_float_range( 0, 1, &snowValue);

#ifdef OCL_COMPILE
 	 {
 	 snow_cl(static_cast<cl_mem>(srcPtr),
			srcSize,
			static_cast<cl_mem>(dstPtr),
			snowValue,
			RPPI_CHN_PLANAR, 3,
			static_cast<cl_command_queue>(rppHandle));
 	 }
#elif defined (HIP_COMPILE)
 	 {
 	 }
#endif //BACKEND
		return RPP_SUCCESS;
}

RppStatus
rppi_snow_u8_pkd3_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f snowValue, RppHandle_t rppHandle)
{

 	 validate_image_size(srcSize);
 	 validate_float_range( 0, 1, &snowValue);

#ifdef OCL_COMPILE
 	 {
 	 snow_cl(static_cast<cl_mem>(srcPtr),
			srcSize,
			static_cast<cl_mem>(dstPtr),
			snowValue,
			RPPI_CHN_PACKED, 3,
			static_cast<cl_command_queue>(rppHandle));
 	 }
#elif defined (HIP_COMPILE)
 	 {
 	 }
#endif //BACKEND
		return RPP_SUCCESS;
}

// ----------------------------------------
// GPU random_shadow functions  calls
// ----------------------------------------


RppStatus
rppi_random_shadow_u8_pln1_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32u x1 ,Rpp32u y1,Rpp32u x2,Rpp32u y2,Rpp32u numberOfShadows,Rpp32u maxSizeX,Rpp32u maxSizeY, RppHandle_t rppHandle)
{

 	 validate_image_size(srcSize);
 	 validate_unsigned_int_range( 0, srcSize.width - 1,& x1);
 	 validate_unsigned_int_range( 0, srcSize.height - 1, &y1);
 	 validate_unsigned_int_range( 0, srcSize.width - 1, &x2);
 	 validate_unsigned_int_range( 0, srcSize.height - 1, &y2);
 	 validate_unsigned_int_min(1, &numberOfShadows);
	 validate_unsigned_int_max(x2,&x1);
	 validate_unsigned_int_max(y2,&y1);

#ifdef OCL_COMPILE
 	 {
 	 random_shadow_cl(static_cast<cl_mem>(srcPtr),
			srcSize,
			static_cast<cl_mem>(dstPtr),
			x1,
			y1,
			x2,
			y2,
			numberOfShadows,
			maxSizeX,
			maxSizeY,
			RPPI_CHN_PLANAR, 1,
			static_cast<cl_command_queue>(rppHandle));
 	 }
#elif defined (HIP_COMPILE)
 	 {
 	 }
#endif //BACKEND
		return RPP_SUCCESS;
}

RppStatus
rppi_random_shadow_u8_pln3_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32u x1 ,Rpp32u y1,Rpp32u x2,Rpp32u y2,Rpp32u numberOfShadows,Rpp32u maxSizeX,Rpp32u maxSizeY, RppHandle_t rppHandle)
{

 	 validate_image_size(srcSize);
 	 validate_unsigned_int_range( 0, srcSize.width - 1,& x1);
 	 validate_unsigned_int_range( 0, srcSize.height - 1, &y1);
 	 validate_unsigned_int_range( 0, srcSize.width - 1, &x2);
 	 validate_unsigned_int_range( 0, srcSize.height - 1, &y2);
 	 validate_unsigned_int_min(1, &numberOfShadows);
	 validate_unsigned_int_max(x2,&x1);
	 validate_unsigned_int_max(y2,&y1);
#ifdef OCL_COMPILE
 	 {
 	 random_shadow_cl(static_cast<cl_mem>(srcPtr),
			srcSize,
			static_cast<cl_mem>(dstPtr),
			x1,
			y1,
			x2,
			y2,
			numberOfShadows,
			maxSizeX,
			maxSizeY,
			RPPI_CHN_PLANAR, 3,
			static_cast<cl_command_queue>(rppHandle));
 	 }
#elif defined (HIP_COMPILE)
 	 {
 	 }
#endif //BACKEND
		return RPP_SUCCESS;
}

RppStatus
rppi_random_shadow_u8_pkd3_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32u x1 ,Rpp32u y1,Rpp32u x2,Rpp32u y2,Rpp32u numberOfShadows,Rpp32u maxSizeX,Rpp32u maxSizeY, RppHandle_t rppHandle)
{

 	 validate_image_size(srcSize);
 	 validate_unsigned_int_range( 0, srcSize.width - 1,& x1);
 	 validate_unsigned_int_range( 0, srcSize.height - 1, &y1);
 	 validate_unsigned_int_range( 0, srcSize.width - 1, &x2);
 	 validate_unsigned_int_range( 0, srcSize.height - 1, &y2);
 	 validate_unsigned_int_min(1, &numberOfShadows);
	 validate_unsigned_int_max(x2,&x1);
	 validate_unsigned_int_max(y2,&y1);
#ifdef OCL_COMPILE
 	 {
 	 random_shadow_cl(static_cast<cl_mem>(srcPtr),
			srcSize,
			static_cast<cl_mem>(dstPtr),
			x1,
			y1,
			x2,
			y2,
			numberOfShadows,
			maxSizeX,
			maxSizeY,
			RPPI_CHN_PACKED, 3,
			static_cast<cl_command_queue>(rppHandle));
 	 }
#elif defined (HIP_COMPILE)
 	 {
 	 }
#endif //BACKEND
		return RPP_SUCCESS;
}

// ----------------------------------------
// GPU blend functions  calls
// ----------------------------------------


RppStatus
rppi_blend_u8_pln1_gpu(RppPtr_t srcPtr1,RppPtr_t srcPtr2,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f alpha, RppHandle_t rppHandle)
{

 	 validate_image_size(srcSize);
 	 validate_float_range( 0, 1, &alpha);

#ifdef OCL_COMPILE
 	 {
 	 blend_cl(static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			srcSize,
			static_cast<cl_mem>(dstPtr),
			alpha,
			RPPI_CHN_PLANAR, 1,
			static_cast<cl_command_queue>(rppHandle));
 	 }
#elif defined (HIP_COMPILE)
 	 {
 	 }
#endif //BACKEND
		return RPP_SUCCESS;
}

RppStatus
rppi_blend_u8_pln3_gpu(RppPtr_t srcPtr1,RppPtr_t srcPtr2,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f alpha, RppHandle_t rppHandle)
{

 	 validate_image_size(srcSize);
 	 validate_float_range( 0, 1, &alpha);

#ifdef OCL_COMPILE
 	 {
 	 blend_cl(static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			srcSize,
			static_cast<cl_mem>(dstPtr),
			alpha,
			RPPI_CHN_PLANAR, 3,
			static_cast<cl_command_queue>(rppHandle));
 	 }
#elif defined (HIP_COMPILE)
 	 {
 	 }
#endif //BACKEND
		return RPP_SUCCESS;
}

RppStatus
rppi_blend_u8_pkd3_gpu(RppPtr_t srcPtr1,RppPtr_t srcPtr2,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f alpha, RppHandle_t rppHandle)
{

 	 validate_image_size(srcSize);
 	 validate_float_range( 0, 1, &alpha);

#ifdef OCL_COMPILE
 	 {
 	 blend_cl(static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			srcSize,
			static_cast<cl_mem>(dstPtr),
			alpha,
			RPPI_CHN_PACKED, 3,
			static_cast<cl_command_queue>(rppHandle));
 	 }
#elif defined (HIP_COMPILE)
 	 {
 	 }
#endif //BACKEND
		return RPP_SUCCESS;
}

// ----------------------------------------
// GPU pixelate functions  calls
// ----------------------------------------


RppStatus
rppi_pixelate_u8_pln1_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32u x1 ,Rpp32u y1,Rpp32u x2,Rpp32u y2 , RppHandle_t rppHandle)
{

 	 validate_image_size(srcSize);
 	 validate_unsigned_int_range( 0, srcSize.width - 1,& x1);
 	 validate_unsigned_int_range( 0, srcSize.height - 1, &y1);
 	 validate_unsigned_int_range( 0, srcSize.width - 1, &x2);
 	 validate_unsigned_int_range( 0, srcSize.height - 1, &y2);
	 Rpp32u kernelSize = 3;
	 validate_unsigned_int_max(x2,&x1);
	 validate_unsigned_int_max(y2,&y1);
#ifdef OCL_COMPILE
 	 {
 	 pixelate_cl(static_cast<cl_mem>(srcPtr),
			srcSize,
			static_cast<cl_mem>(dstPtr),
			kernelSize,
			x1,
			y1,
			x2,
			y2,
			RPPI_CHN_PLANAR, 1,
			static_cast<cl_command_queue>(rppHandle));
 	 }
#elif defined (HIP_COMPILE)
 	 {
 	 }
#endif //BACKEND
		return RPP_SUCCESS;
}

RppStatus
rppi_pixelate_u8_pln3_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32u x1 ,Rpp32u y1,Rpp32u x2,Rpp32u y2, RppHandle_t rppHandle)
{

 	 validate_image_size(srcSize);
 	 validate_unsigned_int_range( 0, srcSize.width - 1,& x1);
 	 validate_unsigned_int_range( 0, srcSize.height - 1, &y1);
 	 validate_unsigned_int_range( 0, srcSize.width - 1, &x2);
 	 validate_unsigned_int_range( 0, srcSize.height - 1, &y2);
	 Rpp32u kernelSize = 3;
	 validate_unsigned_int_max(x2,&x1);
	 validate_unsigned_int_max(y2,&y1);

#ifdef OCL_COMPILE
 	 {
 	 pixelate_cl(static_cast<cl_mem>(srcPtr),
			srcSize,
			static_cast<cl_mem>(dstPtr),
			kernelSize,
			x1,
			y1,
			x2,
			y2,
			RPPI_CHN_PLANAR, 3,
			static_cast<cl_command_queue>(rppHandle));
 	 }
#elif defined (HIP_COMPILE)
 	 {
 	 }
#endif //BACKEND
		return RPP_SUCCESS;
}

RppStatus
rppi_pixelate_u8_pkd3_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32u x1 ,Rpp32u y1,Rpp32u x2,Rpp32u y2, RppHandle_t rppHandle)
{

 	 validate_image_size(srcSize);
 	 validate_unsigned_int_range( 0, srcSize.width - 1,& x1);
 	 validate_unsigned_int_range( 0, srcSize.height - 1, &y1);
 	 validate_unsigned_int_range( 0, srcSize.width - 1, &x2);
 	 validate_unsigned_int_range( 0, srcSize.height - 1, &y2);
	 Rpp32u kernelSize = 3;
	 validate_unsigned_int_max(x2,&x1);
	 validate_unsigned_int_max(y2,&y1);

#ifdef OCL_COMPILE
 	 {
 	 pixelate_cl(static_cast<cl_mem>(srcPtr),
			srcSize,
			static_cast<cl_mem>(dstPtr),
			kernelSize,
			x1,
			y1,
			x2,
			y2,
			RPPI_CHN_PACKED, 3,
			static_cast<cl_command_queue>(rppHandle));
 	 }
#elif defined (HIP_COMPILE)
 	 {
 	 }
#endif //BACKEND
		return RPP_SUCCESS;
}

// ----------------------------------------
// GPU random_crop_letterbox functions  calls
// ----------------------------------------


RppStatus
rppi_random_crop_letterbox_u8_pln1_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,RppiSize dstSize,Rpp32u x1,Rpp32u y1,Rpp32u x2,Rpp32u y2, RppHandle_t rppHandle)
{

 	 validate_image_size(srcSize);
 	 validate_image_size(dstSize);
 	 validate_unsigned_int_range( 0, srcSize.width - 1,& x1);
 	 validate_unsigned_int_range( 0, srcSize.height - 1, &y1);
 	 validate_unsigned_int_range( 0, srcSize.width - 1, &x2);
 	 validate_unsigned_int_range( 0, srcSize.height - 1, &y2);
	 validate_unsigned_int_max(x2,&x1);
	 validate_unsigned_int_max(y2,&y1);
     unsigned int padding=(unsigned int)((dstSize.width/100)*5);
#ifdef OCL_COMPILE
 	 {
 	 resize_crop_cl(static_cast<cl_mem>(srcPtr),
			srcSize,
			static_cast<cl_mem>(dstPtr),
			dstSize,
			x1,
			y1,
			x2,
			y2,
            padding, 1,
			RPPI_CHN_PLANAR, 1,
			static_cast<cl_command_queue>(rppHandle));
 	 }
#elif defined (HIP_COMPILE)
 	 {
 	 }
#endif //BACKEND
		return RPP_SUCCESS;
}

RppStatus
rppi_random_crop_letterbox_u8_pln3_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,RppiSize dstSize,Rpp32u x1,Rpp32u y1,Rpp32u x2,Rpp32u y2, RppHandle_t rppHandle)
{

 	 validate_image_size(srcSize);
 	 validate_image_size(dstSize);
 	 validate_unsigned_int_range( 0, srcSize.width - 1,& x1);
 	 validate_unsigned_int_range( 0, srcSize.height - 1, &y1);
 	 validate_unsigned_int_range( 0, srcSize.width - 1, &x2);
 	 validate_unsigned_int_range( 0, srcSize.height - 1, &y2);
	 validate_unsigned_int_max(x2,&x1);
	 validate_unsigned_int_max(y2,&y1);
     unsigned int padding=(unsigned int)((dstSize.width/100)*5);
#ifdef OCL_COMPILE
 	 {
 	 resize_crop_cl(static_cast<cl_mem>(srcPtr),
			srcSize,
			static_cast<cl_mem>(dstPtr),
			dstSize,
			x1,
			y1,
			x2,
			y2,
            padding, 1,
			RPPI_CHN_PLANAR, 3,
			static_cast<cl_command_queue>(rppHandle));
 	 }
#elif defined (HIP_COMPILE)
 	 {
 	 }
#endif //BACKEND
		return RPP_SUCCESS;
}

RppStatus
rppi_random_crop_letterbox_u8_pkd3_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,RppiSize dstSize,Rpp32u x1,Rpp32u y1,Rpp32u x2,Rpp32u y2, RppHandle_t rppHandle)
{

 	 validate_image_size(srcSize);
 	 validate_image_size(dstSize);
 	 validate_unsigned_int_range( 0, srcSize.width - 1,& x1);
 	 validate_unsigned_int_range( 0, srcSize.height - 1, &y1);
 	 validate_unsigned_int_range( 0, srcSize.width - 1, &x2);
 	 validate_unsigned_int_range( 0, srcSize.height - 1, &y2);
	 validate_unsigned_int_max(x2,&x1);
	 validate_unsigned_int_max(y2,&y1);
     unsigned int padding=(unsigned int)((dstSize.width/100)*5);
#ifdef OCL_COMPILE
 	 {
 	 resize_crop_cl(static_cast<cl_mem>(srcPtr),
			srcSize,
			static_cast<cl_mem>(dstPtr),
			dstSize,
			x1,
			y1,
			x2,
			y2,
            padding, 1,
			RPPI_CHN_PACKED, 3,
			static_cast<cl_command_queue>(rppHandle));
 	 }
#elif defined (HIP_COMPILE)
 	 {
 	 }
#endif //BACKEND
		return RPP_SUCCESS;
}

// ----------------------------------------
// GPU occlusion functions  calls
// ----------------------------------------


RppStatus
rppi_occlusion_u8_pln1_gpu(RppPtr_t srcPtr1,RppiSize srcSize1,RppPtr_t srcPtr2,RppiSize srcSize2,RppPtr_t dstPtr,Rpp32u src1x1,Rpp32u src1y1,Rpp32u src1x2,Rpp32u src1y2,Rpp32u src2x1,Rpp32u src2y1,Rpp32u src2x2,Rpp32u src2y2, RppHandle_t rppHandle)
{

 	 validate_image_size(srcSize1);
 	 validate_image_size(srcSize2);
 	 validate_unsigned_int_range( 0, srcSize1.width - 1, &src1x1);
 	 validate_unsigned_int_range( 0, srcSize1.height - 1, &src1y1);
 	 validate_unsigned_int_range( 0, srcSize1.width - 1, &src1x2);
 	 validate_unsigned_int_range( 0, srcSize1.height - 1, &src1y2);
 	 validate_unsigned_int_range( 0, srcSize1.width - 1, &src2x1);
 	 validate_unsigned_int_range( 0, srcSize1.height - 1, &src2y1);
 	 validate_unsigned_int_range( 0, srcSize1.width - 1, &src2x2);
 	 validate_unsigned_int_range( 0, srcSize1.height - 1,&src2y2);
	 validate_unsigned_int_max(src1x2,&src1x1);
	 validate_unsigned_int_max(src1y2,&src1y1);
	 validate_unsigned_int_max(src2x2,&src2x1);
	 validate_unsigned_int_max(src2y2,&src2y1);

#ifdef OCL_COMPILE
 	 {
 	 occlusion_cl(static_cast<cl_mem>(srcPtr1),
			srcSize1,
			static_cast<cl_mem>(srcPtr2),
			srcSize2,
			static_cast<cl_mem>(dstPtr),
			src1x1,
			src1y1,
			src1x2,
			src1y2,
			src2x1,
			src2y1,
			src2x2,
			src2y2,
			RPPI_CHN_PLANAR, 1,
			static_cast<cl_command_queue>(rppHandle));
 	 }
#elif defined (HIP_COMPILE)
 	 {
 	 }
#endif //BACKEND
		return RPP_SUCCESS;
}

RppStatus
rppi_occlusion_u8_pln3_gpu(RppPtr_t srcPtr1,RppiSize srcSize1,RppPtr_t srcPtr2,RppiSize srcSize2,RppPtr_t dstPtr,
			Rpp32u src1x1,Rpp32u src1y1,Rpp32u src1x2,Rpp32u src1y2,Rpp32u src2x1,Rpp32u src2y1,
			Rpp32u src2x2,Rpp32u src2y2, RppHandle_t rppHandle)
{

 	 validate_image_size(srcSize1);
 	 validate_image_size(srcSize2);
 	 validate_unsigned_int_range( 0, srcSize1.width - 1, &src1x1);
 	 validate_unsigned_int_range( 0, srcSize1.height - 1, &src1y1);
 	 validate_unsigned_int_range( 0, srcSize1.width - 1, &src1x2);
 	 validate_unsigned_int_range( 0, srcSize1.height - 1, &src1y2);
 	 validate_unsigned_int_range( 0, srcSize1.width - 1, &src2x1);
 	 validate_unsigned_int_range( 0, srcSize1.height - 1, &src2y1);
 	 validate_unsigned_int_range( 0, srcSize1.width - 1, &src2x2);
 	 validate_unsigned_int_range( 0, srcSize1.height - 1,&src2y2);
	 validate_unsigned_int_max(src1x2,&src1x1);
	 validate_unsigned_int_max(src1y2,&src1y1);
	 validate_unsigned_int_max(src2x2,&src2x1);
	 validate_unsigned_int_max(src2y2,&src2y1);
#ifdef OCL_COMPILE
 	 {
 	 occlusion_cl(static_cast<cl_mem>(srcPtr1),
			srcSize1,
			static_cast<cl_mem>(srcPtr2),
			srcSize2,
			static_cast<cl_mem>(dstPtr),
			src1x1,
			src1y1,
			src1x2,
			src1y2,
			src2x1,
			src2y1,
			src2x2,
			src2y2,
			RPPI_CHN_PLANAR, 3,
			static_cast<cl_command_queue>(rppHandle));
 	 }
#elif defined (HIP_COMPILE)
 	 {
 	 }
#endif //BACKEND
		return RPP_SUCCESS;
}

RppStatus
rppi_occlusion_u8_pkd3_gpu(RppPtr_t srcPtr1,RppiSize srcSize1,RppPtr_t srcPtr2,RppiSize srcSize2,RppPtr_t dstPtr,Rpp32u src1x1,Rpp32u src1y1,Rpp32u src1x2,Rpp32u src1y2,Rpp32u src2x1,Rpp32u src2y1,Rpp32u src2x2,Rpp32u src2y2, RppHandle_t rppHandle)
{

 	 validate_image_size(srcSize1);
 	 validate_image_size(srcSize2);
 	 validate_unsigned_int_range( 0, srcSize1.width - 1, &src1x1);
 	 validate_unsigned_int_range( 0, srcSize1.height - 1, &src1y1);
 	 validate_unsigned_int_range( 0, srcSize1.width - 1, &src1x2);
 	 validate_unsigned_int_range( 0, srcSize1.height - 1, &src1y2);
 	 validate_unsigned_int_range( 0, srcSize1.width - 1, &src2x1);
 	 validate_unsigned_int_range( 0, srcSize1.height - 1, &src2y1);
 	 validate_unsigned_int_range( 0, srcSize1.width - 1, &src2x2);
 	 validate_unsigned_int_range( 0, srcSize1.height - 1,&src2y2);
	 validate_unsigned_int_max(src1x2,&src1x1);
	 validate_unsigned_int_max(src1y2,&src1y1);
	 validate_unsigned_int_max(src2x2,&src2x1);
	 validate_unsigned_int_max(src2y2,&src2y1);

#ifdef OCL_COMPILE
 	 {

 	 occlusion_cl(static_cast<cl_mem>(srcPtr1),
			srcSize1,
			static_cast<cl_mem>(srcPtr2),
			srcSize2,
			static_cast<cl_mem>(dstPtr),
			src1x1,
			src1y1,
			src1x2,
			src1y2,
			src2x1,
			src2y1,
			src2x2,
			src2y2,
			RPPI_CHN_PACKED, 3,
			static_cast<cl_command_queue>(rppHandle));
 	 }
#elif defined (HIP_COMPILE)
 	 {
 	 }
#endif //BACKEND
		return RPP_SUCCESS;
}

// ----------------------------------------
// GPU exposure functions  calls
// ----------------------------------------


RppStatus
rppi_exposure_u8_pln1_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f exposureValue, RppHandle_t rppHandle)
{

 	 validate_image_size(srcSize);
 	 validate_float_range( -4, 4, &exposureValue);

#ifdef OCL_COMPILE
 	 {
 	 exposure_cl(static_cast<cl_mem>(srcPtr),
			srcSize,
			static_cast<cl_mem>(dstPtr),
			exposureValue,
			RPPI_CHN_PLANAR, 1,
			static_cast<cl_command_queue>(rppHandle));
 	 }
#elif defined (HIP_COMPILE)
 	 {
 	 }
#endif //BACKEND
		return RPP_SUCCESS;
}

RppStatus
rppi_exposure_u8_pln3_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f exposureValue, RppHandle_t rppHandle)
{

 	 validate_image_size(srcSize);
 	 validate_float_range( -4, 4, &exposureValue);

#ifdef OCL_COMPILE
 	 {
 	 exposure_cl(static_cast<cl_mem>(srcPtr),
			srcSize,
			static_cast<cl_mem>(dstPtr),
			exposureValue,
			RPPI_CHN_PLANAR, 3,
			static_cast<cl_command_queue>(rppHandle));
 	 }
#elif defined (HIP_COMPILE)
 	 {
 	 }
#endif //BACKEND
		return RPP_SUCCESS;
}

RppStatus
rppi_exposure_u8_pkd3_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f exposureValue, RppHandle_t rppHandle)
{

 	 validate_image_size(srcSize);
 	 validate_float_range( -4, 4, &exposureValue);

#ifdef OCL_COMPILE
 	 {
 	 exposure_cl(static_cast<cl_mem>(srcPtr),
			srcSize,
			static_cast<cl_mem>(dstPtr),
			exposureValue,
			RPPI_CHN_PACKED, 3,
			static_cast<cl_command_queue>(rppHandle));
 	 }
#elif defined (HIP_COMPILE)
 	 {
 	 }
#endif //BACKEND
		return RPP_SUCCESS;
}

// ----------------------------------------
// GPU jitter functions  calls
// ----------------------------------------
RppStatus
rppi_jitter_u8_pln1_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32u kernelSize, RppHandle_t rppHandle)
{
   	 validate_image_size(srcSize);

#ifdef OCL_COMPILE
 	 {
 	 jitter_cl(static_cast<cl_mem>(srcPtr),
			srcSize,
			static_cast<cl_mem>(dstPtr),
			kernelSize,
			RPPI_CHN_PLANAR, 1,
			static_cast<cl_command_queue>(rppHandle));
 	 }
#elif defined (HIP_COMPILE)
 	 {
 	 }
#endif //BACKEND
		return RPP_SUCCESS;
}

RppStatus
rppi_jitter_u8_pln3_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32u kernelSize, RppHandle_t rppHandle)
{
   	 validate_image_size(srcSize);
#ifdef OCL_COMPILE
 	 {
 	 jitter_cl(static_cast<cl_mem>(srcPtr),
			srcSize,
			static_cast<cl_mem>(dstPtr),
			kernelSize,
			RPPI_CHN_PLANAR, 3,
			static_cast<cl_command_queue>(rppHandle));
 	 }
#elif defined (HIP_COMPILE)
 	 {
 	 }
#endif //BACKEND
		return RPP_SUCCESS;
}

RppStatus
rppi_jitter_u8_pkd3_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32u kernelSize, RppHandle_t rppHandle)
{
   	 validate_image_size(srcSize);
#ifdef OCL_COMPILE
 	 {
 	 jitter_cl(static_cast<cl_mem>(srcPtr),
			srcSize,
			static_cast<cl_mem>(dstPtr),
			kernelSize,
			RPPI_CHN_PACKED, 3,
			static_cast<cl_command_queue>(rppHandle));
 	 }
#elif defined (HIP_COMPILE)
 	 {
 	 }
#endif //BACKEND
		return RPP_SUCCESS;
}


RppStatus
rppi_snpNoise_u8_pln1_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32f noiseProbability, RppHandle_t rppHandle)
{
   	validate_image_size(srcSize);
 	validate_float_range( 0, 1, &noiseProbability);
#ifdef OCL_COMPILE
 	{
            snpNoise_cl(static_cast<cl_mem>(srcPtr),
                srcSize,
                static_cast<cl_mem>(dstPtr),
                noiseProbability,
                RPPI_CHN_PLANAR, 1,
                static_cast<cl_command_queue>(rppHandle));
 	}
#elif defined (HIP_COMPILE)
 	{
 	}
#endif //BACKEND
	return RPP_SUCCESS;
}
RppStatus
rppi_snpNoise_u8_pln3_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32f noiseProbability, RppHandle_t rppHandle)
{
   	validate_image_size(srcSize);
 	validate_float_range( 0, 1, &noiseProbability);

#ifdef OCL_COMPILE
 	{
            snpNoise_cl(static_cast<cl_mem>(srcPtr),
                srcSize,
                static_cast<cl_mem>(dstPtr),
                noiseProbability,
                RPPI_CHN_PLANAR, 3,
                static_cast<cl_command_queue>(rppHandle));
 	}
#elif defined (HIP_COMPILE)
 	{
 	}
#endif //BACKEND
	return RPP_SUCCESS;
}
RppStatus
rppi_snpNoise_u8_pkd3_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32f noiseProbability, RppHandle_t rppHandle)
{
   	validate_image_size(srcSize);
 	validate_float_range( 0, 1, &noiseProbability);

#ifdef OCL_COMPILE
 	{
            snpNoise_cl(static_cast<cl_mem>(srcPtr),
                srcSize,
                static_cast<cl_mem>(dstPtr),
                noiseProbability,
                RPPI_CHN_PACKED, 3,
                static_cast<cl_command_queue>(rppHandle));
 	}
#elif defined (HIP_COMPILE)
 	{
 	}
#endif //BACKEND
	return RPP_SUCCESS;
}
RppStatus
rppi_gaussianNoise_u8_pln1_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32f mean, Rpp32f sigma, RppHandle_t rppHandle)
{
   	validate_image_size(srcSize);

#ifdef OCL_COMPILE
 	{
            gaussianNoise_cl(static_cast<cl_mem>(srcPtr),
                srcSize,
                static_cast<cl_mem>(dstPtr),
                mean, sigma,
                RPPI_CHN_PLANAR, 1,
                static_cast<cl_command_queue>(rppHandle));
 	}
#elif defined (HIP_COMPILE)
 	{
 	}
#endif //BACKEND
	return RPP_SUCCESS;
}
RppStatus
rppi_gaussianNoise_u8_pln3_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32f mean, Rpp32f sigma, RppHandle_t rppHandle)
{
   	validate_image_size(srcSize);

#ifdef OCL_COMPILE
 	{
            gaussianNoise_cl(static_cast<cl_mem>(srcPtr),
                srcSize,
                static_cast<cl_mem>(dstPtr),
                mean, sigma,
                RPPI_CHN_PLANAR, 3,
                static_cast<cl_command_queue>(rppHandle));
 	}
#elif defined (HIP_COMPILE)
 	{
 	}
#endif //BACKEND
	return RPP_SUCCESS;
}
RppStatus
rppi_gaussianNoise_u8_pkd3_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32f mean, Rpp32f sigma, RppHandle_t rppHandle)
{
   	validate_image_size(srcSize);

#ifdef OCL_COMPILE
 	{
            gaussianNoise_cl(static_cast<cl_mem>(srcPtr),
                srcSize,
                static_cast<cl_mem>(dstPtr),
                mean, sigma,
                RPPI_CHN_PACKED, 3,
                static_cast<cl_command_queue>(rppHandle));
 	}
#elif defined (HIP_COMPILE)
 	{
 	}
#endif //BACKEND
	return RPP_SUCCESS;
}


RppStatus
rppi_fog_u8_pln1_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr, Rpp32f fogValue,RppHandle_t rppHandle)
{
 	validate_float_range( 0, 1,&fogValue);
    validate_image_size(srcSize);
	unsigned int kernelSize = 3;
#ifdef OCL_COMPILE
 	{

 	blur_cl(static_cast<cl_mem>(srcPtr),
			srcSize,
			static_cast<cl_mem>(dstPtr),
			kernelSize,
			RPPI_CHN_PLANAR, 1,
			static_cast<cl_command_queue>(rppHandle));
    fog_cl(static_cast<cl_mem>(dstPtr),
			srcSize,
			fogValue,
			RPPI_CHN_PLANAR, 1, 
            static_cast<cl_command_queue>(rppHandle), static_cast<cl_mem>(srcPtr) );
 	 } 
#elif defined (HIP_COMPILE) 
 	 { 
 	 } 
#endif //BACKEND 
		return RPP_SUCCESS;
}

RppStatus
rppi_fog_u8_pln3_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr, Rpp32f fogValue, RppHandle_t rppHandle)
{
 	validate_float_range( 0, 1,&fogValue);
    validate_image_size(srcSize);
	unsigned int kernelSize = 3;
#ifdef OCL_COMPILE
 	{

 	blur_cl(static_cast<cl_mem>(srcPtr),
			srcSize,
			static_cast<cl_mem>(dstPtr),
			kernelSize,
			RPPI_CHN_PLANAR, 3,
			static_cast<cl_command_queue>(rppHandle));
    fog_cl(static_cast<cl_mem>(dstPtr),
			srcSize,
			fogValue,
			RPPI_CHN_PLANAR, 3, 
            static_cast<cl_command_queue>(rppHandle) , static_cast<cl_mem>(srcPtr) );
 	 } 
#elif defined (HIP_COMPILE) 
 	 { 
 	 } 
#endif //BACKEND 
		return RPP_SUCCESS;
}

RppStatus
rppi_fog_u8_pkd3_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr, Rpp32f fogValue, RppHandle_t rppHandle)
{
 	validate_float_range( 0, 1,&fogValue);
    validate_image_size(srcSize);
	unsigned int kernelSize = 3;
#ifdef OCL_COMPILE
 	{

 	blur_cl(static_cast<cl_mem>(srcPtr),
			srcSize,
			static_cast<cl_mem>(dstPtr),
			kernelSize,
			RPPI_CHN_PACKED, 3,
			static_cast<cl_command_queue>(rppHandle));
    fog_cl(static_cast<cl_mem>(dstPtr),
			srcSize,
			fogValue,
			RPPI_CHN_PACKED, 3, 
            static_cast<cl_command_queue>(rppHandle) , static_cast<cl_mem>(srcPtr) );
 	 } 
#elif defined (HIP_COMPILE) 
 	 { 
 	 } 
#endif //BACKEND 
		return RPP_SUCCESS;
}

RppStatus
rppi_histogram_balance_u8_pln1_gpu(RppPtr_t srcPtr, RppiSize srcSize,
                                    RppPtr_t dstPtr, RppHandle_t rppHandle)
{

    validate_image_size(srcSize);
#ifdef OCL_COMPILE
 	{
 	histogram_balance_cl(static_cast<cl_mem>(srcPtr),
			srcSize,
			static_cast<cl_mem>(dstPtr),
			RPPI_CHN_PLANAR, 1,
			static_cast<cl_command_queue>(rppHandle));
 	 }
#elif defined (HIP_COMPILE)
 	 {
 	 }
#endif //BACKEND
		return RPP_SUCCESS;
}


RppStatus
rppi_histogram_balance_u8_pln3_gpu(RppPtr_t srcPtr, RppiSize srcSize,
                                    RppPtr_t dstPtr, RppHandle_t rppHandle)
{

    validate_image_size(srcSize);
#ifdef OCL_COMPILE
 	{
 	histogram_balance_cl(static_cast<cl_mem>(srcPtr),
			srcSize,
			static_cast<cl_mem>(dstPtr),
			RPPI_CHN_PLANAR, 3,
			static_cast<cl_command_queue>(rppHandle));
 	 }
#elif defined (HIP_COMPILE)
 	 {
 	 }
#endif //BACKEND
		return RPP_SUCCESS;
}

RppStatus
rppi_histogram_balance_u8_pkd3_gpu(RppPtr_t srcPtr, RppiSize srcSize,
                                    RppPtr_t dstPtr, RppHandle_t rppHandle)
{

    validate_image_size(srcSize);
#ifdef OCL_COMPILE
 	{
 	histogram_balance_cl(static_cast<cl_mem>(srcPtr),
			srcSize,
			static_cast<cl_mem>(dstPtr),
			RPPI_CHN_PACKED, 3,
			static_cast<cl_command_queue>(rppHandle));
 	 }
#elif defined (HIP_COMPILE)
 	 {
 	 }
#endif //BACKEND
		return RPP_SUCCESS;
}

// ----------------------------------------
// GPU histogram_equalize functions  calls 
// ----------------------------------------


RppStatus
rppi_histogram_equalize_u8_pln1_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr, RppHandle_t rppHandle) 
{

 	 validate_image_size(srcSize);

#ifdef OCL_COMPILE
 	 {
 	 histogram_balance_cl(static_cast<cl_mem>(srcPtr), 
			srcSize,
			static_cast<cl_mem>(dstPtr), 
			RPPI_CHN_PLANAR, 1,
			static_cast<cl_command_queue>(rppHandle));
 	 } 
#elif defined (HIP_COMPILE) 
 	 { 
 	 } 
#endif //BACKEND 
		return RPP_SUCCESS;
}

RppStatus
rppi_histogram_equalize_u8_pln3_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr, RppHandle_t rppHandle) 
{

 	 validate_image_size(srcSize);

#ifdef OCL_COMPILE
 	 {
 	 histogram_balance_cl(static_cast<cl_mem>(srcPtr), 
			srcSize,
			static_cast<cl_mem>(dstPtr), 
			RPPI_CHN_PLANAR, 3,
			static_cast<cl_command_queue>(rppHandle));
 	 } 
#elif defined (HIP_COMPILE) 
 	 { 
 	 } 
#endif //BACKEND 
		return RPP_SUCCESS;
}

RppStatus
rppi_histogram_equalize_u8_pkd3_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr, RppHandle_t rppHandle) 
{

 	 validate_image_size(srcSize);

#ifdef OCL_COMPILE
 	 {
 	 histogram_balance_cl(static_cast<cl_mem>(srcPtr), 
			srcSize,
			static_cast<cl_mem>(dstPtr), 
			RPPI_CHN_PACKED, 3,
			static_cast<cl_command_queue>(rppHandle));
 	 } 
#elif defined (HIP_COMPILE) 
 	 { 
 	 } 
#endif //BACKEND 
		return RPP_SUCCESS;
}