#ifndef RPPI_ADVANCED_AUGMENTATIONS_H
#define RPPI_ADVANCED_AUGMENTATIONS_H

#include "rppdefs.h"
#include "rpp.h"
#ifdef __cplusplus
extern "C"
{
#endif // cpusplus

/******************** water ********************/

// Performs a water augmentation for a batch of images

// *param[in] srcPtr Input image batch
// *param[in] srcSize Array containing an RppiSize for each image in the batch
// *param[in] maxSrcSize A single RppiSize which is the maxWidth and maxHeight for all images in the batch
// *param[out] dstPtr Output image batch
// *param[in] ampl_x Array containing an Rpp32f amplitude of water wave in x direction for each image in the batch (ampl_x[n] >= 0)
// *param[in] ampl_y Array containing an Rpp32f amplitude of water wave in y direction for each image in the batch (ampl_y[n] >= 0)
// *param[in] freq_x Array containing an Rpp32f frequency of water wave in x direction for each image in the batch (freq_x[n] >= 0)
// *param[in] freq_y Array containing an Rpp32f frequency of water wave in y direction for each image in the batch (freq_y[n] >= 0)
// *param[in] phase_x Array containing an Rpp32f phase of water wave in x direction for each image in the batch (phase_x[n] >= 0)
// *param[in] phase_y Array containing an Rpp32f phase of water wave in y direction for each image in the batch (phase_y[n] >= 0)
// *param[in] outputFormatToggle An Rpp32u flag to set layout toggling on/off for each image in the batch (outputFormatToggle = 0/1)
// *param[in] nbatchSize Batch size or the number of images in the batch
// *param[in] rppHandle OpenCL-handle/HIP-handle for "_gpu" variants and Host-handle for "_host" variants
// *returns a  RppStatus enumeration.
// *retval RPP_SUCCESS : No error, Succesful completion
// *retval RPP_ERROR : Error

RppStatus rppi_water_u8_pln1_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *ampl_x, Rpp32f *ampl_y, Rpp32f *freq_x, Rpp32f *freq_y, Rpp32f *phase_x, Rpp32f *phase_y, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_water_f16_pln1_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *ampl_x, Rpp32f *ampl_y, Rpp32f *freq_x, Rpp32f *freq_y, Rpp32f *phase_x, Rpp32f *phase_y, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_water_f32_pln1_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *ampl_x, Rpp32f *ampl_y, Rpp32f *freq_x, Rpp32f *freq_y, Rpp32f *phase_x, Rpp32f *phase_y, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_water_i8_pln1_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *ampl_x, Rpp32f *ampl_y, Rpp32f *freq_x, Rpp32f *freq_y, Rpp32f *phase_x, Rpp32f *phase_y, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_water_u8_pln3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *ampl_x, Rpp32f *ampl_y, Rpp32f *freq_x, Rpp32f *freq_y, Rpp32f *phase_x, Rpp32f *phase_y, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_water_f16_pln3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *ampl_x, Rpp32f *ampl_y, Rpp32f *freq_x, Rpp32f *freq_y, Rpp32f *phase_x, Rpp32f *phase_y, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_water_f32_pln3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *ampl_x, Rpp32f *ampl_y, Rpp32f *freq_x, Rpp32f *freq_y, Rpp32f *phase_x, Rpp32f *phase_y, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_water_i8_pln3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *ampl_x, Rpp32f *ampl_y, Rpp32f *freq_x, Rpp32f *freq_y, Rpp32f *phase_x, Rpp32f *phase_y, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_water_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *ampl_x, Rpp32f *ampl_y, Rpp32f *freq_x, Rpp32f *freq_y, Rpp32f *phase_x, Rpp32f *phase_y, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_water_f16_pkd3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *ampl_x, Rpp32f *ampl_y, Rpp32f *freq_x, Rpp32f *freq_y, Rpp32f *phase_x, Rpp32f *phase_y, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_water_f32_pkd3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *ampl_x, Rpp32f *ampl_y, Rpp32f *freq_x, Rpp32f *freq_y, Rpp32f *phase_x, Rpp32f *phase_y, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_water_i8_pkd3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *ampl_x, Rpp32f *ampl_y, Rpp32f *freq_x, Rpp32f *freq_y, Rpp32f *phase_x, Rpp32f *phase_y, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_water_u8_pln1_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *ampl_x, Rpp32f *ampl_y, Rpp32f *freq_x, Rpp32f *freq_y, Rpp32f *phase_x, Rpp32f *phase_y, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_water_f16_pln1_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *ampl_x, Rpp32f *ampl_y, Rpp32f *freq_x, Rpp32f *freq_y, Rpp32f *phase_x, Rpp32f *phase_y, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_water_f32_pln1_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *ampl_x, Rpp32f *ampl_y, Rpp32f *freq_x, Rpp32f *freq_y, Rpp32f *phase_x, Rpp32f *phase_y, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_water_i8_pln1_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *ampl_x, Rpp32f *ampl_y, Rpp32f *freq_x, Rpp32f *freq_y, Rpp32f *phase_x, Rpp32f *phase_y, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_water_u8_pln3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *ampl_x, Rpp32f *ampl_y, Rpp32f *freq_x, Rpp32f *freq_y, Rpp32f *phase_x, Rpp32f *phase_y, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_water_f16_pln3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *ampl_x, Rpp32f *ampl_y, Rpp32f *freq_x, Rpp32f *freq_y, Rpp32f *phase_x, Rpp32f *phase_y, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_water_f32_pln3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *ampl_x, Rpp32f *ampl_y, Rpp32f *freq_x, Rpp32f *freq_y, Rpp32f *phase_x, Rpp32f *phase_y, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_water_i8_pln3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *ampl_x, Rpp32f *ampl_y, Rpp32f *freq_x, Rpp32f *freq_y, Rpp32f *phase_x, Rpp32f *phase_y, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_water_u8_pkd3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *ampl_x, Rpp32f *ampl_y, Rpp32f *freq_x, Rpp32f *freq_y, Rpp32f *phase_x, Rpp32f *phase_y, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_water_f16_pkd3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *ampl_x, Rpp32f *ampl_y, Rpp32f *freq_x, Rpp32f *freq_y, Rpp32f *phase_x, Rpp32f *phase_y, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_water_f32_pkd3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *ampl_x, Rpp32f *ampl_y, Rpp32f *freq_x, Rpp32f *freq_y, Rpp32f *phase_x, Rpp32f *phase_y, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_water_i8_pkd3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *ampl_x, Rpp32f *ampl_y, Rpp32f *freq_x, Rpp32f *freq_y, Rpp32f *phase_x, Rpp32f *phase_y, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);

/******************** non_linear_blend ********************/

// Performs a non linear blend augmentation between corresponding pixels of two batches of images

// *param[in] srcPtr1 Input image1 batch
// *param[in] srcPtr2 Input image2 batch
// *param[in] srcSize Array containing an RppiSize for each image in the batch
// *param[in] maxSrcSize A single RppiSize which is the maxWidth and maxHeight for all images in the batch
// *param[out] dstPtr Output image batch
// *param[in] stdDev Array containing an Rpp32f standard deviation to decide non linearity for each image in the batch (stdDev[n] >= 0)
// *param[in] outputFormatToggle An Rpp32u flag to set layout toggling on/off for each image in the batch (outputFormatToggle = 0/1)
// *param[in] nbatchSize Batch size or the number of images in the batch
// *param[in] rppHandle OpenCL-handle/HIP-handle for "_gpu" variants and Host-handle for "_host" variants
// *returns a  RppStatus enumeration.
// *retval RPP_SUCCESS : No error, Succesful completion
// *retval RPP_ERROR : Error

RppStatus rppi_non_linear_blend_u8_pln1_batchPD_gpu(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *std_dev, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_non_linear_blend_f16_pln1_batchPD_gpu(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *std_dev, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_non_linear_blend_f32_pln1_batchPD_gpu(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *std_dev, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_non_linear_blend_i8_pln1_batchPD_gpu(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *std_dev, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_non_linear_blend_u8_pln3_batchPD_gpu(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *std_dev, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_non_linear_blend_f16_pln3_batchPD_gpu(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *std_dev, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_non_linear_blend_f32_pln3_batchPD_gpu(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *std_dev, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_non_linear_blend_i8_pln3_batchPD_gpu(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *std_dev, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_non_linear_blend_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *std_dev, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_non_linear_blend_f16_pkd3_batchPD_gpu(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *std_dev, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_non_linear_blend_f32_pkd3_batchPD_gpu(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *std_dev, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_non_linear_blend_i8_pkd3_batchPD_gpu(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *std_dev, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_non_linear_blend_u8_pln1_batchPD_host(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *std_dev, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_non_linear_blend_f16_pln1_batchPD_host(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *std_dev, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_non_linear_blend_f32_pln1_batchPD_host(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *std_dev, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_non_linear_blend_i8_pln1_batchPD_host(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *std_dev, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_non_linear_blend_u8_pln3_batchPD_host(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *std_dev, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_non_linear_blend_f16_pln3_batchPD_host(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *std_dev, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_non_linear_blend_f32_pln3_batchPD_host(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *std_dev, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_non_linear_blend_i8_pln3_batchPD_host(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *std_dev, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_non_linear_blend_u8_pkd3_batchPD_host(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *std_dev, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_non_linear_blend_f16_pkd3_batchPD_host(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *std_dev, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_non_linear_blend_f32_pkd3_batchPD_host(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *std_dev, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_non_linear_blend_i8_pkd3_batchPD_host(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *std_dev, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);

/******************** color_cast ********************/

// Performs a color cast augmentation for a batch of images

// *param[in] srcPtr Input image batch
// *param[in] srcSize Array containing an RppiSize for each image in the batch
// *param[in] maxSrcSize A single RppiSize which is the maxWidth and maxHeight for all images in the batch
// *param[out] dstPtr Output image batch
// *param[in] r Array containing an Rpp8u r value for each image in the batch. (0 <= r[n] <= 255)
// *param[in] g Array containing an Rpp8u g value for each image in the batch. (0 <= g[n] <= 255)
// *param[in] b Array containing an Rpp8u b value for each image in the batch. (0 <= b[n] <= 255)
// *param[in] alpha Array containing an Rpp32f alpha value for each image in the batch. The alpha value is used to blend the srcPtr's r/g/b pixel with the user's r/g/b pixel (alpha[n] >= 0)
// *param[in] outputFormatToggle An Rpp32u flag to set layout toggling on/off for each image in the batch (outputFormatToggle = 0/1)
// *param[in] nbatchSize Batch size or the number of images in the batch
// *param[in] rppHandle OpenCL-handle/HIP-handle for "_gpu" variants and Host-handle for "_host" variants
// *returns a  RppStatus enumeration.
// *retval RPP_SUCCESS : No error, Succesful completion
// *retval RPP_ERROR : Error

RppStatus rppi_color_cast_u8_pln3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp8u *r, Rpp8u *g, Rpp8u *b, Rpp32f *alpha, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_color_cast_f16_pln3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp8u *r, Rpp8u *g, Rpp8u *b, Rpp32f *alpha, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_color_cast_f32_pln3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp8u *r, Rpp8u *g, Rpp8u *b, Rpp32f *alpha, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_color_cast_i8_pln3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp8u *r, Rpp8u *g, Rpp8u *b, Rpp32f *alpha, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_color_cast_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp8u *r, Rpp8u *g, Rpp8u *b, Rpp32f *alpha, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_color_cast_f16_pkd3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp8u *r, Rpp8u *g, Rpp8u *b, Rpp32f *alpha, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_color_cast_f32_pkd3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp8u *r, Rpp8u *g, Rpp8u *b, Rpp32f *alpha, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_color_cast_i8_pkd3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp8u *r, Rpp8u *g, Rpp8u *b, Rpp32f *alpha, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_color_cast_u8_pln3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp8u *r, Rpp8u *g, Rpp8u *b, Rpp32f *alpha, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_color_cast_f16_pln3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp8u *r, Rpp8u *g, Rpp8u *b, Rpp32f *alpha, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_color_cast_f32_pln3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp8u *r, Rpp8u *g, Rpp8u *b, Rpp32f *alpha, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_color_cast_i8_pln3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp8u *r, Rpp8u *g, Rpp8u *b, Rpp32f *alpha, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_color_cast_u8_pkd3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp8u *r, Rpp8u *g, Rpp8u *b, Rpp32f *alpha, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_color_cast_f16_pkd3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp8u *r, Rpp8u *g, Rpp8u *b, Rpp32f *alpha, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_color_cast_f32_pkd3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp8u *r, Rpp8u *g, Rpp8u *b, Rpp32f *alpha, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_color_cast_i8_pkd3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp8u *r, Rpp8u *g, Rpp8u *b, Rpp32f *alpha, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);

/******************** erase ********************/

// Performs an erase augmentation that erases one or more user defined regions from an image, for a batch of images

// *param[in] srcPtr Input image batch
// *param[in] srcSize Array containing an RppiSize for each image in the batch
// *param[in] maxSrcSize A single RppiSize which is the maxWidth and maxHeight for all images in the batch
// *param[out] dstPtr Output image batch
// *param[in] anchor_box_info Array containing a set of 4 Rpp32u x1/y1/x2/y2 for each erase-region inside each image in the batch. (0 <= anchor_box_info[i] < respective image width/height)
// *param[in] colors Array containing 3 Rpp8u r,g,b values for each erase-region inside each image in the batch. (0 <= colors[i] <= 255)
// *param[in] box_offset Array containing an Rpp32u value that gives the number of boxes to offset for each image in the batch. Example - If num_of_boxes in each image = 3, box_offset[0] = 0, box_offset[1] = 1 * 3, box_offset[2] = 2 * 3 (box_offset[n] >= 0)
// *param[in] num_of_boxes Array containing an Rpp32u number of erase-regions per image, for each image in the batch. (num_of_boxes[n] >= 0)
// *param[in] outputFormatToggle An Rpp32u flag to set layout toggling on/off for each image in the batch (outputFormatToggle = 0/1)
// *param[in] nbatchSize Batch size or the number of images in the batch
// *param[in] rppHandle OpenCL-handle/HIP-handle for "_gpu" variants and Host-handle for "_host" variants
// *returns a  RppStatus enumeration.
// *retval RPP_SUCCESS : No error, Succesful completion
// *retval RPP_ERROR : Error

RppStatus rppi_erase_u8_pln1_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppPtr_t anchor_box_info, RppPtr_t colors, RppPtr_t box_offset, Rpp32u *num_of_boxes, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_erase_f16_pln1_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppPtr_t anchor_box_info, RppPtr_t colors, RppPtr_t box_offset, Rpp32u *num_of_boxes, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_erase_f32_pln1_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppPtr_t anchor_box_info, RppPtr_t colors, RppPtr_t box_offset, Rpp32u *num_of_boxes, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_erase_i8_pln1_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppPtr_t anchor_box_info, RppPtr_t colors, RppPtr_t box_offset, Rpp32u *num_of_boxes, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_erase_u8_pln3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppPtr_t anchor_box_info, RppPtr_t colors, RppPtr_t box_offset, Rpp32u *num_of_boxes, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_erase_f16_pln3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppPtr_t anchor_box_info, RppPtr_t colors, RppPtr_t box_offset, Rpp32u *num_of_boxes, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_erase_f32_pln3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppPtr_t anchor_box_info, RppPtr_t colors, RppPtr_t box_offset, Rpp32u *num_of_boxes, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_erase_i8_pln3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppPtr_t anchor_box_info, RppPtr_t colors, RppPtr_t box_offset, Rpp32u *num_of_boxes, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_erase_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppPtr_t anchor_box_info, RppPtr_t colors, RppPtr_t box_offset, Rpp32u *num_of_boxes, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_erase_f16_pkd3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppPtr_t anchor_box_info, RppPtr_t colors, RppPtr_t box_offset, Rpp32u *num_of_boxes, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_erase_f32_pkd3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppPtr_t anchor_box_info, RppPtr_t colors, RppPtr_t box_offset, Rpp32u *num_of_boxes, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_erase_i8_pkd3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppPtr_t anchor_box_info, RppPtr_t colors, RppPtr_t box_offset, Rpp32u *num_of_boxes, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_erase_u8_pln1_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppPtr_t anchor_box_info, RppPtr_t colors, RppPtr_t box_offset, Rpp32u *num_of_boxes, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_erase_u8_pln3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppPtr_t anchor_box_info, RppPtr_t colors, RppPtr_t box_offset, Rpp32u *num_of_boxes, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_erase_u8_pkd3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppPtr_t anchor_box_info, RppPtr_t colors, RppPtr_t box_offset, Rpp32u *num_of_boxes, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_erase_f16_pln1_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppPtr_t anchor_box_info, RppPtr_t colors, RppPtr_t box_offset, Rpp32u *num_of_boxes, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_erase_f16_pln3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppPtr_t anchor_box_info, RppPtr_t colors, RppPtr_t box_offset, Rpp32u *num_of_boxes, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_erase_f16_pkd3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppPtr_t anchor_box_info, RppPtr_t colors, RppPtr_t box_offset, Rpp32u *num_of_boxes, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_erase_f32_pln1_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppPtr_t anchor_box_info, RppPtr_t colors, RppPtr_t box_offset, Rpp32u *num_of_boxes, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_erase_f32_pln3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppPtr_t anchor_box_info, RppPtr_t colors, RppPtr_t box_offset, Rpp32u *num_of_boxes, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_erase_f32_pkd3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppPtr_t anchor_box_info, RppPtr_t colors, RppPtr_t box_offset, Rpp32u *num_of_boxes, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_erase_i8_pln1_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppPtr_t anchor_box_info, RppPtr_t colors, RppPtr_t box_offset, Rpp32u *num_of_boxes, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_erase_i8_pln3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppPtr_t anchor_box_info, RppPtr_t colors, RppPtr_t box_offset, Rpp32u *num_of_boxes, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_erase_i8_pkd3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppPtr_t anchor_box_info, RppPtr_t colors, RppPtr_t box_offset, Rpp32u *num_of_boxes, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);

/******************** crop_and_patch ********************/

// Performs a crop and patch augmentation taking crops from the images in batch2 and patching the crop into the corresponding images of batch 1

// *param[in] srcPtr1 Input image1 batch
// *param[in] srcPtr2 Input image2 batch
// *param[in] srcSize Array containing an RppiSize for each image in the batch
// *param[in] maxSrcSize A single RppiSize which is the maxWidth and maxHeight for all images in the batch
// *param[out] dstPtr Output image batch
// *param[in] x11 Array containing an Rpp32u x1 location for Top-Left ROI for each image in batch 1
// *param[in] y11 Array containing an Rpp32u y1 location for Top-Left ROI for each image in batch 1
// *param[in] x12 Array containing an Rpp32u x2 location for Bottom-Right ROI for each image in batch 1
// *param[in] y12 Array containing an Rpp32u y2 location for Bottom-Right ROI for each image in batch 1
// *param[in] x21 Array containing an Rpp32u x1 location for Top-Left ROI for each image in batch 2
// *param[in] y21 Array containing an Rpp32u y1 location for Top-Left ROI for each image in batch 2
// *param[in] x22 Array containing an Rpp32u x2 location for Bottom-Right ROI for each image in batch 2
// *param[in] y22 Array containing an Rpp32u y2 location for Bottom-Right ROI for each image in batch 2
// *param[in] outputFormatToggle An Rpp32u flag to set layout toggling on/off for each image in the batch (outputFormatToggle = 0/1)
// *param[in] nbatchSize Batch size or the number of images in the batch
// *param[in] rppHandle OpenCL-handle/HIP-handle for "_gpu" variants and Host-handle for "_host" variants
// *returns a  RppStatus enumeration.
// *retval RPP_SUCCESS : No error, Succesful completion
// *retval RPP_ERROR : Error

RppStatus rppi_crop_and_patch_u8_pln1_batchPD_gpu(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u *x11, Rpp32u *y11, Rpp32u *x12, Rpp32u *y12, Rpp32u *x21, Rpp32u *y21, Rpp32u *x22, Rpp32u *y22, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_crop_and_patch_f16_pln1_batchPD_gpu(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u *x11, Rpp32u *y11, Rpp32u *x12, Rpp32u *y12, Rpp32u *x21, Rpp32u *y21, Rpp32u *x22, Rpp32u *y22, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_crop_and_patch_f32_pln1_batchPD_gpu(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u *x11, Rpp32u *y11, Rpp32u *x12, Rpp32u *y12, Rpp32u *x21, Rpp32u *y21, Rpp32u *x22, Rpp32u *y22, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_crop_and_patch_i8_pln1_batchPD_gpu(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u *x11, Rpp32u *y11, Rpp32u *x12, Rpp32u *y12, Rpp32u *x21, Rpp32u *y21, Rpp32u *x22, Rpp32u *y22, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_crop_and_patch_u8_pln3_batchPD_gpu(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u *x11, Rpp32u *y11, Rpp32u *x12, Rpp32u *y12, Rpp32u *x21, Rpp32u *y21, Rpp32u *x22, Rpp32u *y22, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_crop_and_patch_f16_pln3_batchPD_gpu(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u *x11, Rpp32u *y11, Rpp32u *x12, Rpp32u *y12, Rpp32u *x21, Rpp32u *y21, Rpp32u *x22, Rpp32u *y22, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_crop_and_patch_f32_pln3_batchPD_gpu(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u *x11, Rpp32u *y11, Rpp32u *x12, Rpp32u *y12, Rpp32u *x21, Rpp32u *y21, Rpp32u *x22, Rpp32u *y22, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_crop_and_patch_i8_pln3_batchPD_gpu(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u *x11, Rpp32u *y11, Rpp32u *x12, Rpp32u *y12, Rpp32u *x21, Rpp32u *y21, Rpp32u *x22, Rpp32u *y22, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_crop_and_patch_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u *x11, Rpp32u *y11, Rpp32u *x12, Rpp32u *y12, Rpp32u *x21, Rpp32u *y21, Rpp32u *x22, Rpp32u *y22, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_crop_and_patch_f16_pkd3_batchPD_gpu(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u *x11, Rpp32u *y11, Rpp32u *x12, Rpp32u *y12, Rpp32u *x21, Rpp32u *y21, Rpp32u *x22, Rpp32u *y22, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_crop_and_patch_f32_pkd3_batchPD_gpu(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u *x11, Rpp32u *y11, Rpp32u *x12, Rpp32u *y12, Rpp32u *x21, Rpp32u *y21, Rpp32u *x22, Rpp32u *y22, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_crop_and_patch_i8_pkd3_batchPD_gpu(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u *x11, Rpp32u *y11, Rpp32u *x12, Rpp32u *y12, Rpp32u *x21, Rpp32u *y21, Rpp32u *x22, Rpp32u *y22, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_crop_and_patch_u8_pln1_batchPD_host(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u *x11, Rpp32u *y11, Rpp32u *x12, Rpp32u *y12, Rpp32u *x21, Rpp32u *y21, Rpp32u *x22, Rpp32u *y22, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_crop_and_patch_f16_pln1_batchPD_host(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u *x11, Rpp32u *y11, Rpp32u *x12, Rpp32u *y12, Rpp32u *x21, Rpp32u *y21, Rpp32u *x22, Rpp32u *y22, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_crop_and_patch_f32_pln1_batchPD_host(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u *x11, Rpp32u *y11, Rpp32u *x12, Rpp32u *y12, Rpp32u *x21, Rpp32u *y21, Rpp32u *x22, Rpp32u *y22, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_crop_and_patch_i8_pln1_batchPD_host(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u *x11, Rpp32u *y11, Rpp32u *x12, Rpp32u *y12, Rpp32u *x21, Rpp32u *y21, Rpp32u *x22, Rpp32u *y22, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_crop_and_patch_u8_pln3_batchPD_host(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u *x11, Rpp32u *y11, Rpp32u *x12, Rpp32u *y12, Rpp32u *x21, Rpp32u *y21, Rpp32u *x22, Rpp32u *y22, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_crop_and_patch_f16_pln3_batchPD_host(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u *x11, Rpp32u *y11, Rpp32u *x12, Rpp32u *y12, Rpp32u *x21, Rpp32u *y21, Rpp32u *x22, Rpp32u *y22, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_crop_and_patch_f32_pln3_batchPD_host(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u *x11, Rpp32u *y11, Rpp32u *x12, Rpp32u *y12, Rpp32u *x21, Rpp32u *y21, Rpp32u *x22, Rpp32u *y22, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_crop_and_patch_i8_pln3_batchPD_host(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u *x11, Rpp32u *y11, Rpp32u *x12, Rpp32u *y12, Rpp32u *x21, Rpp32u *y21, Rpp32u *x22, Rpp32u *y22, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_crop_and_patch_u8_pkd3_batchPD_host(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u *x11, Rpp32u *y11, Rpp32u *x12, Rpp32u *y12, Rpp32u *x21, Rpp32u *y21, Rpp32u *x22, Rpp32u *y22, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_crop_and_patch_f16_pkd3_batchPD_host(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u *x11, Rpp32u *y11, Rpp32u *x12, Rpp32u *y12, Rpp32u *x21, Rpp32u *y21, Rpp32u *x22, Rpp32u *y22, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_crop_and_patch_f32_pkd3_batchPD_host(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u *x11, Rpp32u *y11, Rpp32u *x12, Rpp32u *y12, Rpp32u *x21, Rpp32u *y21, Rpp32u *x22, Rpp32u *y22, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_crop_and_patch_i8_pkd3_batchPD_host(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u *x11, Rpp32u *y11, Rpp32u *x12, Rpp32u *y12, Rpp32u *x21, Rpp32u *y21, Rpp32u *x22, Rpp32u *y22, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);

/******************** lut ********************/

// Performs a table look-up for each pixel in a batch of images

// *param[in] srcPtr Input image batch
// *param[in] srcSize Array containing an RppiSize for each image in the batch
// *param[in] maxSrcSize A single RppiSize which is the maxWidth and maxHeight for all images in the batch
// *param[out] dstPtr Output image batch
// *param[in] lut Array containing an Rpp8u* look up table of length 256, for each image in the batch
// *param[in] outputFormatToggle An Rpp32u flag to set layout toggling on/off for each image in the batch (outputFormatToggle = 0/1)
// *param[in] nbatchSize Batch size or the number of images in the batch
// *param[in] rppHandle OpenCL-handle/HIP-handle for "_gpu" variants and Host-handle for "_host" variants
// *returns a  RppStatus enumeration.
// *retval RPP_SUCCESS : No error, Succesful completion
// *retval RPP_ERROR : Error

RppStatus rppi_lut_u8_pln1_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppPtr_t lut, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_lut_i8_pln1_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppPtr_t lut, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_lut_u8_pln3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppPtr_t lut, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_lut_i8_pln3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppPtr_t lut, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_lut_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppPtr_t lut, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_lut_i8_pkd3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppPtr_t lut, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_lut_u8_pln1_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppPtr_t lut, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_lut_i8_pln1_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppPtr_t lut, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_lut_u8_pln3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppPtr_t lut, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_lut_i8_pln3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppPtr_t lut, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_lut_u8_pkd3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppPtr_t lut, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_lut_i8_pkd3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppPtr_t lut, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);

/******************** glitch ********************/

// Performs a glitch augmentation for a batch of images

// *param[in] srcPtr Input image batch
// *param[in] srcSize Array containing an RppiSize for each image in the batch
// *param[in] maxSrcSize A single RppiSize which is the maxWidth and maxHeight for all images in the batch
// *param[out] dstPtr Output image batch
// *param[in] x_offset_r Array containing an Rpp32u x offset value for the r-channel pixels in each image in the batch (x_offset_r[n] >= 0)
// *param[in] y_offset_r Array containing an Rpp32u y offset value for the r-channel pixels in each image in the batch (y_offset_r[n] >= 0)
// *param[in] x_offset_g Array containing an Rpp32u x offset value for the g-channel pixels in each image in the batch (x_offset_g[n] >= 0)
// *param[in] y_offset_g Array containing an Rpp32u y offset value for the g-channel pixels in each image in the batch (y_offset_g[n] >= 0)
// *param[in] x_offset_b Array containing an Rpp32u x offset value for the b-channel pixels in each image in the batch (x_offset_b[n] >= 0)
// *param[in] y_offset_b Array containing an Rpp32u y offset value for the b-channel pixels in each image in the batch (y_offset_b[n] >= 0)
// *param[in] outputFormatToggle An Rpp32u flag to set layout toggling on/off for each image in the batch (outputFormatToggle = 0/1)
// *param[in] nbatchSize Batch size or the number of images in the batch
// *param[in] rppHandle OpenCL-handle/HIP-handle for "_gpu" variants and Host-handle for "_host" variants
// *returns a  RppStatus enumeration.
// *retval RPP_SUCCESS : No error, Succesful completion
// *retval RPP_ERROR : Error

RppStatus rppi_glitch_u8_pln3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u *x_offset_r, Rpp32u *y_offset_r, Rpp32u *x_offset_g, Rpp32u *y_offset_g, Rpp32u *x_offset_b, Rpp32u *y_offset_b, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_glitch_f16_pln3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u *x_offset_r, Rpp32u *y_offset_r, Rpp32u *x_offset_g, Rpp32u *y_offset_g, Rpp32u *x_offset_b, Rpp32u *y_offset_b, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_glitch_f32_pln3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u *x_offset_r, Rpp32u *y_offset_r, Rpp32u *x_offset_g, Rpp32u *y_offset_g, Rpp32u *x_offset_b, Rpp32u *y_offset_b, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_glitch_i8_pln3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u *x_offset_r, Rpp32u *y_offset_r, Rpp32u *x_offset_g, Rpp32u *y_offset_g, Rpp32u *x_offset_b, Rpp32u *y_offset_b, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_glitch_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u *x_offset_r, Rpp32u *y_offset_r, Rpp32u *x_offset_g, Rpp32u *y_offset_g, Rpp32u *x_offset_b, Rpp32u *y_offset_b, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_glitch_f16_pkd3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u *x_offset_r, Rpp32u *y_offset_r, Rpp32u *x_offset_g, Rpp32u *y_offset_g, Rpp32u *x_offset_b, Rpp32u *y_offset_b, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_glitch_f32_pkd3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u *x_offset_r, Rpp32u *y_offset_r, Rpp32u *x_offset_g, Rpp32u *y_offset_g, Rpp32u *x_offset_b, Rpp32u *y_offset_b, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_glitch_i8_pkd3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u *x_offset_r, Rpp32u *y_offset_r, Rpp32u *x_offset_g, Rpp32u *y_offset_g, Rpp32u *x_offset_b, Rpp32u *y_offset_b, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_glitch_u8_pln3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u *x_offset_r, Rpp32u *y_offset_r, Rpp32u *x_offset_g, Rpp32u *y_offset_g, Rpp32u *x_offset_b, Rpp32u *y_offset_b, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_glitch_f16_pln3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u *x_offset_r, Rpp32u *y_offset_r, Rpp32u *x_offset_g, Rpp32u *y_offset_g, Rpp32u *x_offset_b, Rpp32u *y_offset_b, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_glitch_f32_pln3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u *x_offset_r, Rpp32u *y_offset_r, Rpp32u *x_offset_g, Rpp32u *y_offset_g, Rpp32u *x_offset_b, Rpp32u *y_offset_b, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_glitch_i8_pln3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u *x_offset_r, Rpp32u *y_offset_r, Rpp32u *x_offset_g, Rpp32u *y_offset_g, Rpp32u *x_offset_b, Rpp32u *y_offset_b, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_glitch_u8_pkd3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u *x_offset_r, Rpp32u *y_offset_r, Rpp32u *x_offset_g, Rpp32u *y_offset_g, Rpp32u *x_offset_b, Rpp32u *y_offset_b, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_glitch_f16_pkd3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u *x_offset_r, Rpp32u *y_offset_r, Rpp32u *x_offset_g, Rpp32u *y_offset_g, Rpp32u *x_offset_b, Rpp32u *y_offset_b, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_glitch_f32_pkd3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u *x_offset_r, Rpp32u *y_offset_r, Rpp32u *x_offset_g, Rpp32u *y_offset_g, Rpp32u *x_offset_b, Rpp32u *y_offset_b, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_glitch_i8_pkd3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u *x_offset_r, Rpp32u *y_offset_r, Rpp32u *x_offset_g, Rpp32u *y_offset_g, Rpp32u *x_offset_b, Rpp32u *y_offset_b, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);

/******************** ricap ********************/

// Performs a crop and patch augmentation taking crops from the images in batch2 and patching the crop into the corresponding images of batch 1

// *param[in] srcPtr1 Input image1 batch
// *param[in] srcSize Array containing an RppiSize for each image in the batch
// *param[in] maxSrcSize A single RppiSize which is the maxWidth and maxHeight for all images in the batch
// *param[out] dstPtr Output image batch
// *param[in] Permuted Indices Array 1 for a batch of images
// *param[in] Permuted Indices Array 2 for a batch of images
// *param[in] Permuted Indices Array 3 for a batch of images
// *param[in] Permuted Indices Array 4 for a batch of images
// *param[in] Permuted Crop Region 1 for a batch of images for 1st Permuted Array of a batch of images
// *param[in] Permuted Crop Region 2 for a batch of images for 2nd Permuted Array of a batch of images
// *param[in] Permuted Crop Region 3 for a batch of images for 3rd Permuted Array of a batch of images
// *param[in] Permuted Crop Region 4 for a batch of images for 4th Permuted Array of a batch of images
// *param[in] outputFormatToggle An Rpp32u flag to set layout toggling on/off for each image in the batch (outputFormatToggle = 0/1)
// *param[in] nbatchSize Batch size or the number of images in the batch
// *param[in] rppHandle OpenCL-handle/HIP-handle for "_gpu" variants and Host-handle for "_host" variants
// *returns a  RppStatus enumeration.
// *retval RPP_SUCCESS : No error, Succesful completion
// *retval RPP_ERROR : Error

RppStatus rppi_ricap_u8_pln1_batchPD_gpu(RppPtr_t srcPtr1, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u *permutedIndices1, Rpp32u *permutedIndices2, Rpp32u *permutedIndices3, Rpp32u *permutedIndices4, Rpp32u *cropRegion1, Rpp32u *cropRegion2, Rpp32u *cropRegion3, Rpp32u *cropRegion4, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
// RppStatus rppi_ricap_f16_pln1_batchPD_gpu(RppPtr_t srcPtr1, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u *permutedIndices1, Rpp32u *permutedIndices2, Rpp32u *permutedIndices3, Rpp32u *permutedIndices4, Rpp32u *cropRegion1, Rpp32u *cropRegion2, Rpp32u *cropRegion3, Rpp32u *cropRegion4, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
// RppStatus rppi_ricap_f32_pln1_batchPD_gpu(RppPtr_t srcPtr1, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u *permutedIndices1, Rpp32u *permutedIndices2, Rpp32u *permutedIndices3, Rpp32u *permutedIndices4, Rpp32u *cropRegion1, Rpp32u *cropRegion2, Rpp32u *cropRegion3, Rpp32u *cropRegion4, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
// RppStatus rppi_ricap_i8_pln1_batchPD_gpu(RppPtr_t srcPtr1, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u *permutedIndices1, Rpp32u *permutedIndices2, Rpp32u *permutedIndices3, Rpp32u *permutedIndices4, Rpp32u *cropRegion1, Rpp32u *cropRegion2, Rpp32u *cropRegion3, Rpp32u *cropRegion4, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_ricap_u8_pln3_batchPD_gpu(RppPtr_t srcPtr1, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u *permutedIndices1, Rpp32u *permutedIndices2, Rpp32u *permutedIndices3, Rpp32u *permutedIndices4, Rpp32u *cropRegion1, Rpp32u *cropRegion2, Rpp32u *cropRegion3, Rpp32u *cropRegion4, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
// RppStatus rppi_ricap_f16_pln3_batchPD_gpu(RppPtr_t srcPtr1, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u *permutedIndices1, Rpp32u *permutedIndices2, Rpp32u *permutedIndices3, Rpp32u *permutedIndices4, Rpp32u *cropRegion1, Rpp32u *cropRegion2, Rpp32u *cropRegion3, Rpp32u *cropRegion4, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
// RppStatus rppi_ricap_f32_pln3_batchPD_gpu(RppPtr_t srcPtr1, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u *permutedIndices1, Rpp32u *permutedIndices2, Rpp32u *permutedIndices3, Rpp32u *permutedIndices4, Rpp32u *cropRegion1, Rpp32u *cropRegion2, Rpp32u *cropRegion3, Rpp32u *cropRegion4, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
// RppStatus rppi_ricap_i8_pln3_batchPD_gpu(RppPtr_t srcPtr1, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u *permutedIndices1, Rpp32u *permutedIndices2, Rpp32u *permutedIndices3, Rpp32u *permutedIndices4, Rpp32u *cropRegion1, Rpp32u *cropRegion2, Rpp32u *cropRegion3, Rpp32u *cropRegion4, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_ricap_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr1, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u *permutedIndices1, Rpp32u *permutedIndices2, Rpp32u *permutedIndices3, Rpp32u *permutedIndices4, Rpp32u *cropRegion1, Rpp32u *cropRegion2, Rpp32u *cropRegion3, Rpp32u *cropRegion4, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
// RppStatus rppi_ricap_f16_pkd3_batchPD_gpu(RppPtr_t srcPtr1, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u *permutedIndices1, Rpp32u *permutedIndices2, Rpp32u *permutedIndices3, Rpp32u *permutedIndices4, Rpp32u *cropRegion1, Rpp32u *cropRegion2, Rpp32u *cropRegion3, Rpp32u *cropRegion4, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
// RppStatus rppi_ricap_f32_pkd3_batchPD_gpu(RppPtr_t srcPtr1, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u *permutedIndices1, Rpp32u *permutedIndices2, Rpp32u *permutedIndices3, Rpp32u *permutedIndices4, Rpp32u *cropRegion1, Rpp32u *cropRegion2, Rpp32u *cropRegion3, Rpp32u *cropRegion4, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
// RppStatus rppi_ricap_i8_pkd3_batchPD_gpu(RppPtr_t srcPtr1, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u *permutedIndices1, Rpp32u *permutedIndices2, Rpp32u *permutedIndices3, Rpp32u *permutedIndices4, Rpp32u *cropRegion1, Rpp32u *cropRegion2, Rpp32u *cropRegion3, Rpp32u *cropRegion4, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_ricap_u8_pln1_batchPD_host(RppPtr_t srcPtr1, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u *permutedIndices1, Rpp32u *permutedIndices2, Rpp32u *permutedIndices3, Rpp32u *permutedIndices4, Rpp32u *cropRegion1, Rpp32u *cropRegion2, Rpp32u *cropRegion3, Rpp32u *cropRegion4, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
// RppStatus rppi_ricap_f16_pln1_batchPD_host(RppPtr_t srcPtr1, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u *permutedIndices1, Rpp32u *permutedIndices2, Rpp32u *permutedIndices3, Rpp32u *permutedIndices4, Rpp32u *cropRegion1, Rpp32u *cropRegion2, Rpp32u *cropRegion3, Rpp32u *cropRegion4, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
// RppStatus rppi_ricap_f32_pln1_batchPD_host(RppPtr_t srcPtr1, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u *permutedIndices1, Rpp32u *permutedIndices2, Rpp32u *permutedIndices3, Rpp32u *permutedIndices4, Rpp32u *cropRegion1, Rpp32u *cropRegion2, Rpp32u *cropRegion3, Rpp32u *cropRegion4, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
// RppStatus rppi_ricap_i8_pln1_batchPD_host(RppPtr_t srcPtr1, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u *permutedIndices1, Rpp32u *permutedIndices2, Rpp32u *permutedIndices3, Rpp32u *permutedIndices4, Rpp32u *cropRegion1, Rpp32u *cropRegion2, Rpp32u *cropRegion3, Rpp32u *cropRegion4, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_ricap_u8_pln3_batchPD_host(RppPtr_t srcPtr1, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u *permutedIndices1, Rpp32u *permutedIndices2, Rpp32u *permutedIndices3, Rpp32u *permutedIndices4, Rpp32u *cropRegion1, Rpp32u *cropRegion2, Rpp32u *cropRegion3, Rpp32u *cropRegion4, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
// RppStatus rppi_ricap_f16_pln3_batchPD_host(RppPtr_t srcPtr1, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u *permutedIndices1, Rpp32u *permutedIndices2, Rpp32u *permutedIndices3, Rpp32u *permutedIndices4, Rpp32u *cropRegion1, Rpp32u *cropRegion2, Rpp32u *cropRegion3, Rpp32u *cropRegion4, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
// RppStatus rppi_ricap_f32_pln3_batchPD_host(RppPtr_t srcPtr1, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u *permutedIndices1, Rpp32u *permutedIndices2, Rpp32u *permutedIndices3, Rpp32u *permutedIndices4, Rpp32u *cropRegion1, Rpp32u *cropRegion2, Rpp32u *cropRegion3, Rpp32u *cropRegion4, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
// RppStatus rppi_ricap_i8_pln3_batchPD_host(RppPtr_t srcPtr1, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u *permutedIndices1, Rpp32u *permutedIndices2, Rpp32u *permutedIndices3, Rpp32u *permutedIndices4, Rpp32u *cropRegion1, Rpp32u *cropRegion2, Rpp32u *cropRegion3, Rpp32u *cropRegion4, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_ricap_u8_pkd3_batchPD_host(RppPtr_t srcPtr1, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u *permutedIndices1, Rpp32u *permutedIndices2, Rpp32u *permutedIndices3, Rpp32u *permutedIndices4, Rpp32u *cropRegion1, Rpp32u *cropRegion2, Rpp32u *cropRegion3, Rpp32u *cropRegion4, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
// RppStatus rppi_ricap_f16_pkd3_batchPD_host(RppPtr_t srcPtr1, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u *permutedIndices1, Rpp32u *permutedIndices2, Rpp32u *permutedIndices3, Rpp32u *permutedIndices4, Rpp32u *cropRegion1, Rpp32u *cropRegion2, Rpp32u *cropRegion3, Rpp32u *cropRegion4, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
// RppStatus rppi_ricap_f32_pkd3_batchPD_host(RppPtr_t srcPtr1, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u *permutedIndices1, Rpp32u *permutedIndices2, Rpp32u *permutedIndices3, Rpp32u *permutedIndices4, Rpp32u *cropRegion1, Rpp32u *cropRegion2, Rpp32u *cropRegion3, Rpp32u *cropRegion4, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
// RppStatus rppi_ricap_i8_pkd3_batchPD_host(RppPtr_t srcPtr1, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u *permutedIndices1, Rpp32u *permutedIndices2, Rpp32u *permutedIndices3, Rpp32u *permutedIndices4, Rpp32u *cropRegion1, Rpp32u *cropRegion2, Rpp32u *cropRegion3, Rpp32u *cropRegion4, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);

#ifdef __cplusplus
}
#endif

#endif
