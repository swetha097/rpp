#include <hip/hip_runtime.h>
#include "hip/rpp_hip_common.hpp"

template <typename T>
__global__ void ricap_pkd_tensor(T *srcPtr,
                                      int nStrideSrc,
                                      int hStrideSrc,
                                      T *dstPtr,
                                      int nStrideDst,
                                      int hStrideDst,
                                      uint *permutedIndices1,
                                      uint *permutedIndices2,
                                      uint *permutedIndices3,
                                      uint *permutedIndices4,
                                      RpptROIPtr crop_region,
                                      RpptROIPtr roiTensorPtrSrc)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    uint srcIdx, dstIdx, permuteIdx;
    d_float8 pix_f8;

    if ((id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight) || (id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth * 3))
    {
        return;
    }

    permuteIdx = id_z;

    if ((id_x >= 0) && (id_y >= 0) && (id_y <= crop_region[0].xywhROI.roiHeight) && (id_x <=  crop_region[0].xywhROI.roiWidth * 3))
        srcIdx = (permutedIndices1[permuteIdx] * nStrideSrc) +  ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * hStrideSrc) + (id_x + roiTensorPtrSrc[id_z].xywhROI.xy.x * 3);
   else if ((id_y >= 0) && (id_x >= crop_region[0].xywhROI.roiWidth * 3) && (id_y <= (crop_region[1].xywhROI.roiHeight)) && (id_x <= ( roiTensorPtrSrc[id_z].xywhROI.roiWidth * 3)))
        srcIdx = (permutedIndices2[permuteIdx] * nStrideSrc) + ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * hStrideSrc) + (id_x + roiTensorPtrSrc[id_z].xywhROI.xy.x * 3);
    else if ((id_y >= crop_region[1].xywhROI.roiHeight) && (id_x >= 0) &&  (id_y <= (roiTensorPtrSrc[id_z].xywhROI.roiHeight)) && (id_x <= crop_region[2].xywhROI.roiWidth * 3))
        srcIdx = (permutedIndices3[permuteIdx] * nStrideSrc) + ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * hStrideSrc) + (id_x + roiTensorPtrSrc[id_z].xywhROI.xy.x * 3);
    else if ((id_y >=crop_region[1].xywhROI.roiHeight) && (id_x >=  crop_region[2].xywhROI.roiWidth * 3) &&  (id_y <= (roiTensorPtrSrc[id_z].xywhROI.roiHeight)) && (id_x <=  (roiTensorPtrSrc[id_z].xywhROI.roiWidth * 3)))
        srcIdx = (permutedIndices4[permuteIdx] * nStrideSrc) + ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * hStrideSrc) + (id_x + roiTensorPtrSrc[id_z].xywhROI.xy.x * 3);

    dstIdx = (id_z * nStrideDst) + (id_y * hStrideDst) + id_x;

    rpp_hip_load8_and_unpack_to_float8(srcPtr, srcIdx, &pix_f8);
    rpp_hip_pack_float8_and_store8(dstPtr, dstIdx, &pix_f8);
}

template <typename T>
__global__ void ricap_pln_tensor(T *srcPtr,
                                      int nStrideSrc,
                                      int cStrideSrc,
                                      int hStrideSrc,
                                      T *dstPtr,
                                      int nStrideDst,
                                      int cStrideDst,
                                      int hStrideDst,
                                      int channelsDst,
                                      uint *permutedIndices1,
                                      uint *permutedIndices2,
                                      uint *permutedIndices3,
                                      uint *permutedIndices4,
                                      RpptROIPtr crop_region,
                                      RpptROIPtr roiTensorPtrSrc)
{

    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    uint srcIdx, dstIdx;
    d_float8 pix_f8;

    if ((id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight) || (id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
    {
        return;
    }

    if ((id_x >= 0) && (id_y >= 0) && (id_y <= crop_region[0].xywhROI.roiHeight) && (id_x <=  crop_region[0].xywhROI.roiWidth))
        srcIdx = (permutedIndices1[id_z] * nStrideSrc) +  ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * hStrideSrc) + (id_x + roiTensorPtrSrc[id_z].xywhROI.xy.x);
   else  if ((id_y >= 0) && (id_x >= crop_region[0].xywhROI.roiWidth) && (id_y <= (crop_region[1].xywhROI.roiHeight)) && (id_x <= (roiTensorPtrSrc[id_z].xywhROI.roiWidth)))
        srcIdx = (permutedIndices2[id_z] * nStrideSrc) + ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * hStrideSrc) + (id_x + roiTensorPtrSrc[id_z].xywhROI.xy.x);
    else if ((id_y >= crop_region[1].xywhROI.roiHeight) && (id_x >= 0) &&  (id_y <= (roiTensorPtrSrc[id_z].xywhROI.roiHeight)) && (id_x <= crop_region[2].xywhROI.roiWidth))
        srcIdx = (permutedIndices3[id_z] * nStrideSrc) + ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * hStrideSrc) + (id_x + roiTensorPtrSrc[id_z].xywhROI.xy.x );
    else if ((id_y >=crop_region[1].xywhROI.roiHeight) && (id_x >=  crop_region[2].xywhROI.roiWidth ) &&  (id_y <= (roiTensorPtrSrc[id_z].xywhROI.roiHeight)) && (id_x <=  (roiTensorPtrSrc[id_z].xywhROI.roiWidth)))
        srcIdx = (permutedIndices4[id_z] * nStrideSrc) + ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * hStrideSrc) + (id_x + roiTensorPtrSrc[id_z].xywhROI.xy.x);

    dstIdx = (id_z * nStrideDst) + (id_y * hStrideDst) + id_x;

    rpp_hip_load8_and_unpack_to_float8(srcPtr, srcIdx, &pix_f8);
    rpp_hip_pack_float8_and_store8(dstPtr, dstIdx, &pix_f8);

    if (channelsDst == 3)
    {
        srcIdx += cStrideSrc;
        dstIdx += cStrideDst;

        rpp_hip_load8_and_unpack_to_float8(srcPtr, srcIdx, &pix_f8);
        rpp_hip_pack_float8_and_store8(dstPtr, dstIdx, &pix_f8);

        srcIdx += cStrideSrc;
        dstIdx += cStrideDst;

        rpp_hip_load8_and_unpack_to_float8(srcPtr, srcIdx, &pix_f8);
        rpp_hip_pack_float8_and_store8(dstPtr, dstIdx, &pix_f8);
    }
}

template <typename T>
__global__ void ricap_pkd3_pln3_tensor(T *srcPtr,
                                            uint2 srcStridesNH,
                                            T *dstPtr,
                                            uint3 dstStridesNCH,
                                            uint *permutedIndices1,
                                            uint *permutedIndices2,
                                            uint *permutedIndices3,
                                            uint *permutedIndices4,
                                            RpptROIPtr crop_region,
                                            RpptROIPtr roiTensorPtrSrc)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if ((id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight) || (id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
    {
        return;
    }

    uint srcIdx, dstIdx;

    d_float24 pix_f24;

    if ((id_x >= 0) && (id_y >= 0) && (id_y <= crop_region[0].xywhROI.roiHeight) && (id_x <=  crop_region[0].xywhROI.roiWidth))
        srcIdx = (permutedIndices1[id_z] * srcStridesNH.x) + ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNH.y) + ((id_x + roiTensorPtrSrc[id_z].xywhROI.xy.x) * 3);
   else if ((id_y >= 0) && (id_x >= crop_region[0].xywhROI.roiWidth) && (id_y <= (crop_region[1].xywhROI.roiHeight)) && (id_x <= ( roiTensorPtrSrc[id_z].xywhROI.roiWidth)))
            srcIdx = (permutedIndices2[id_z] * srcStridesNH.x) + ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNH.y) + ((id_x + roiTensorPtrSrc[id_z].xywhROI.xy.x) * 3);
    else if ((id_y >= crop_region[1].xywhROI.roiHeight) && (id_x >= 0) &&  (id_y <= (roiTensorPtrSrc[id_z].xywhROI.roiHeight)) && (id_x <= crop_region[2].xywhROI.roiWidth))
        srcIdx = (permutedIndices3[id_z] * srcStridesNH.x) + ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNH.y) + ((id_x + roiTensorPtrSrc[id_z].xywhROI.xy.x) * 3);
    else if ((id_y >=crop_region[1].xywhROI.roiHeight) && (id_x >=  crop_region[2].xywhROI.roiWidth) &&  (id_y <= (roiTensorPtrSrc[id_z].xywhROI.roiHeight)) && (id_x <=  (roiTensorPtrSrc[id_z].xywhROI.roiWidth)))
        srcIdx = (permutedIndices4[id_z] * srcStridesNH.x) + ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNH.y) + ((id_x + roiTensorPtrSrc[id_z].xywhROI.xy.x) * 3);

    dstIdx = (id_z * dstStridesNCH.x) + (id_y * dstStridesNCH.z) + id_x;

    rpp_hip_load24_pkd3_and_unpack_to_float24_pln3(srcPtr, srcIdx, &pix_f24);
    rpp_hip_pack_float24_pln3_and_store24_pln3(dstPtr, dstIdx, dstStridesNCH.y, &pix_f24);
}

template <typename T>
__global__ void ricap_pln3_pkd3_tensor(T *srcPtr,
                                            int nStrideSrc,
                                            int cStrideSrc,
                                            int hStrideSrc,
                                            T *dstPtr,
                                            int nStrideDst,
                                            int hStrideDst,
                                            uint *permutedIndices1,
                                            uint *permutedIndices2,
                                            uint *permutedIndices3,
                                            uint *permutedIndices4,
                                            RpptROIPtr crop_region,
                                            RpptROIPtr roiTensorPtrSrc)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if ((id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight) || (id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
    {
        return;
    }

    uint srcIdx , dstIdx;
    if ((id_x >= 0) && (id_y >= 0) && (id_y <= crop_region[0].xywhROI.roiHeight) && (id_x <=  crop_region[0].xywhROI.roiWidth))
        srcIdx = (permutedIndices1[id_z] * nStrideSrc) +  ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * hStrideSrc) + (id_x + roiTensorPtrSrc[id_z].xywhROI.xy.x);
   else  if ((id_y >= 0) && (id_x >= crop_region[0].xywhROI.roiWidth) && (id_y <= (crop_region[1].xywhROI.roiHeight)) && (id_x <= (roiTensorPtrSrc[id_z].xywhROI.roiWidth)))
        srcIdx = (permutedIndices2[id_z] * nStrideSrc) + ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * hStrideSrc) + (id_x + roiTensorPtrSrc[id_z].xywhROI.xy.x);
    else if ((id_y >= crop_region[1].xywhROI.roiHeight) && (id_x >= 0) &&  (id_y <= (roiTensorPtrSrc[id_z].xywhROI.roiHeight)) && (id_x <= crop_region[2].xywhROI.roiWidth))
        srcIdx = (permutedIndices3[id_z] * nStrideSrc) + ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * hStrideSrc) + (id_x + roiTensorPtrSrc[id_z].xywhROI.xy.x );
    else if ((id_y >=crop_region[1].xywhROI.roiHeight) && (id_x >=  crop_region[2].xywhROI.roiWidth ) &&  (id_y <= (roiTensorPtrSrc[id_z].xywhROI.roiHeight)) && (id_x <=  (roiTensorPtrSrc[id_z].xywhROI.roiWidth)))
        srcIdx = (permutedIndices4[id_z] * nStrideSrc) + ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * hStrideSrc) + (id_x + roiTensorPtrSrc[id_z].xywhROI.xy.x);

    dstIdx = (id_z * nStrideDst) + (id_y * hStrideDst) + id_x * 3;

    d_float24 pix_f24;

    rpp_hip_load24_pln3_and_unpack_to_float24_pkd3(srcPtr, srcIdx, cStrideSrc, &pix_f24);
    rpp_hip_pack_float24_pkd3_and_store24_pkd3(dstPtr, dstIdx, &pix_f24);
}

template <typename T>
RppStatus hip_exec_ricap_tensor(T *srcPtr,
                                     RpptDescPtr srcDescPtr,
                                     T *dstPtr,
                                     RpptDescPtr dstDescPtr,
                                     RpptROIPtr roiTensorPtrSrc,
                                     RpptROIPtr cropRegion,
                                     rpp::Handle& handle)
{
    int localThreads_x = 16;
    int localThreads_y = 16;
    int localThreads_z = 1;
    int globalThreads_x = (dstDescPtr->strides.hStride + 7) >> 3;
    int globalThreads_y = dstDescPtr->h;
    int globalThreads_z = handle.GetBatchSize();

    if ((srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
    {
        hipLaunchKernelGGL(ricap_pkd_tensor,
                           dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                           dim3(localThreads_x, localThreads_y, localThreads_z),
                           0,
                           handle.GetStream(),
                           srcPtr,
                           srcDescPtr->strides.nStride,
                           srcDescPtr->strides.hStride,
                           dstPtr,
                           dstDescPtr->strides.nStride,
                           dstDescPtr->strides.hStride,
                            handle.GetInitHandle()->mem.mgpu.uintArr[0].uintmem,
                            handle.GetInitHandle()->mem.mgpu.uintArr[1].uintmem,
                            handle.GetInitHandle()->mem.mgpu.uintArr[2].uintmem,
                            handle.GetInitHandle()->mem.mgpu.uintArr[3].uintmem,
                           cropRegion,
                           roiTensorPtrSrc);
    }
    else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
    {
        hipLaunchKernelGGL(ricap_pln_tensor,
                           dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                           dim3(localThreads_x, localThreads_y, localThreads_z),
                           0,
                           handle.GetStream(),
                           srcPtr,
                           srcDescPtr->strides.nStride,
                           srcDescPtr->strides.cStride,
                           srcDescPtr->strides.hStride,
                           dstPtr,
                           dstDescPtr->strides.nStride,
                           dstDescPtr->strides.cStride,
                           dstDescPtr->strides.hStride,
                           dstDescPtr->c,
                            handle.GetInitHandle()->mem.mgpu.uintArr[0].uintmem,
                            handle.GetInitHandle()->mem.mgpu.uintArr[1].uintmem,
                            handle.GetInitHandle()->mem.mgpu.uintArr[2].uintmem,
                            handle.GetInitHandle()->mem.mgpu.uintArr[3].uintmem,
                           cropRegion,
                           roiTensorPtrSrc);
    }
    else if ((srcDescPtr->c == 3) && (dstDescPtr->c == 3))
    {
        if ((srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            hipLaunchKernelGGL(ricap_pkd3_pln3_tensor,
                               dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                               dim3(localThreads_x, localThreads_y, localThreads_z),
                               0,
                               handle.GetStream(),
                               srcPtr,
                               make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                               dstPtr,
                               make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride, dstDescPtr->strides.hStride),
                            handle.GetInitHandle()->mem.mgpu.uintArr[0].uintmem,
                            handle.GetInitHandle()->mem.mgpu.uintArr[1].uintmem,
                            handle.GetInitHandle()->mem.mgpu.uintArr[2].uintmem,
                            handle.GetInitHandle()->mem.mgpu.uintArr[3].uintmem,
                               cropRegion,
                               roiTensorPtrSrc);
        }
        else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            globalThreads_x = (srcDescPtr->strides.hStride + 7) >> 3;
            hipLaunchKernelGGL(ricap_pln3_pkd3_tensor,
                               dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                               dim3(localThreads_x, localThreads_y, localThreads_z),
                               0,
                               handle.GetStream(),
                               srcPtr,
                               srcDescPtr->strides.nStride,
                               srcDescPtr->strides.cStride,
                               srcDescPtr->strides.hStride,
                               dstPtr,
                               dstDescPtr->strides.nStride,
                               dstDescPtr->strides.hStride,
                            handle.GetInitHandle()->mem.mgpu.uintArr[0].uintmem,
                            handle.GetInitHandle()->mem.mgpu.uintArr[1].uintmem,
                            handle.GetInitHandle()->mem.mgpu.uintArr[2].uintmem,
                            handle.GetInitHandle()->mem.mgpu.uintArr[3].uintmem,
                               cropRegion,
                               roiTensorPtrSrc);
        }
    }

    return RPP_SUCCESS;
}
