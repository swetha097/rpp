#include <hip/hip_runtime.h>
#include "hip/rpp_hip_common.hpp"

__device__ void ricap_hip_compute(uchar *srcPtr, d_float8 *src_f8, d_float8 *dst_f8)
{
    dst_f8->x = src_f8->x ;
    dst_f8->y = src_f8->y ;
}

__device__ void ricap_hip_compute(float *srcPtr, d_float8 *src_f8, d_float8 *dst_f8)
{
    dst_f8->x = src_f8->x ;
    dst_f8->y = src_f8->y ;
}

__device__ void ricap_hip_compute(signed char *srcPtr, d_float8 *src_f8, d_float8 *dst_f8)
{
    dst_f8->x = src_f8->x;
    dst_f8->y = src_f8->y ;
}

__device__ void ricap_hip_compute(half *srcPtr, d_float8 *src_f8, d_float8 *dst_f8)
{
    dst_f8->x = src_f8->x;
    dst_f8->y = src_f8->y;
}

template <typename T>
__global__ void ricap_pkd_tensor(T *srcPtr,
                                      int nStrideSrc,
                                      int hStrideSrc,
                                      T *dstPtr,
                                      int nStrideDst,
                                      int hStrideDst,
                                      uint *permutations1,
                                      uint *permutations2,
                                      uint *permutations3,
                                      uint *permutations4,
                                      RpptROIPtr crop_region,
                                      RpptROIPtr roiTensorPtrSrc)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    uint srcIdx, dstIdx;
    d_float8 src_f8, dst_f8;

    if ((id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight) || (id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth * 3))
    {
        return;
    }

    if ((id_x >= 0) && (id_y >= 0) && (id_y <= crop_region[0].xywhROI.roiHeight) && (id_x <=  crop_region[0].xywhROI.roiWidth * 3))
        srcIdx = (permutations1[id_z] * nStrideSrc) +  ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * hStrideSrc) + (id_x + roiTensorPtrSrc[id_z].xywhROI.xy.x * 3);
   else if ((id_y >= 0) && (id_x >= crop_region[0].xywhROI.roiWidth * 3) && (id_y <= (crop_region[1].xywhROI.roiHeight)) && (id_x <= ((crop_region[0].xywhROI.roiWidth  + crop_region[1].xywhROI.roiWidth) * 3)))
        srcIdx = (permutations2[id_z] * nStrideSrc) + ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * hStrideSrc) + (id_x + roiTensorPtrSrc[id_z].xywhROI.xy.x * 3);
    else if ((id_y >= crop_region[1].xywhROI.roiHeight) && (id_x >= 0) &&  (id_y <= (crop_region[1].xywhROI.roiHeight + crop_region[3].xywhROI.roiHeight)) && (id_x <= crop_region[2].xywhROI.roiWidth * 3))
        srcIdx = (permutations3[id_z] * nStrideSrc) + ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * hStrideSrc) + (id_x + roiTensorPtrSrc[id_z].xywhROI.xy.x * 3);
    else if ((id_y >=crop_region[1].xywhROI.roiHeight) && (id_x >=  crop_region[2].xywhROI.roiWidth * 3) &&  (id_y <= (crop_region[1].xywhROI.roiHeight + crop_region[3].xywhROI.roiHeight)) && (id_x <=  (crop_region[2].xywhROI.roiWidth  * 3 + crop_region[3].xywhROI.roiWidth * 3)))
        srcIdx = (permutations4[id_z] * nStrideSrc) + ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * hStrideSrc) + (id_x + roiTensorPtrSrc[id_z].xywhROI.xy.x * 3);

    dstIdx = (id_z * nStrideDst) + (id_y * hStrideDst) + id_x;

    rpp_hip_load8_and_unpack_to_float8(srcPtr, srcIdx, &src_f8);
    ricap_hip_compute(srcPtr, &src_f8, &dst_f8);
    rpp_hip_pack_float8_and_store8(dstPtr, dstIdx, &dst_f8);
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
                                      uint *permutations1,
                                      uint *permutations2,
                                      uint *permutations3,
                                      uint *permutations4,
                                      RpptROIPtr crop_region,
                                      RpptROIPtr roiTensorPtrSrc)
{

    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    uint srcIdx, dstIdx;
    d_float8 src_f8, dst_f8;

    if ((id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight) || (id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
    {
        return;
    }

    if ((id_x >= 0) && (id_y >= 0) && (id_y <= crop_region[0].xywhROI.roiHeight) && (id_x <=  crop_region[0].xywhROI.roiWidth))
        srcIdx = (permutations1[id_z] * nStrideSrc) +  ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * hStrideSrc) + (id_x + roiTensorPtrSrc[id_z].xywhROI.xy.x);
   else  if ((id_y >= 0) && (id_x >= crop_region[0].xywhROI.roiWidth) && (id_y <= (crop_region[1].xywhROI.roiHeight)) && (id_x <= (crop_region[0].xywhROI.roiWidth + crop_region[1].xywhROI.roiWidth)))
        srcIdx = (permutations2[id_z] * nStrideSrc) + ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * hStrideSrc) + (id_x + roiTensorPtrSrc[id_z].xywhROI.xy.x);
    else if ((id_y >= crop_region[1].xywhROI.roiHeight) && (id_x >= 0) &&  (id_y <= (crop_region[1].xywhROI.roiHeight + crop_region[3].xywhROI.roiHeight)) && (id_x <= crop_region[2].xywhROI.roiWidth))
        srcIdx = (permutations3[id_z] * nStrideSrc) + ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * hStrideSrc) + (id_x + roiTensorPtrSrc[id_z].xywhROI.xy.x );
    else if ((id_y >=crop_region[1].xywhROI.roiHeight) && (id_x >=  crop_region[2].xywhROI.roiWidth ) &&  (id_y <= (crop_region[1].xywhROI.roiHeight + crop_region[3].xywhROI.roiHeight)) && (id_x <=  (crop_region[2].xywhROI.roiWidth  + crop_region[3].xywhROI.roiWidth)))
        srcIdx = (permutations4[id_z] * nStrideSrc) + ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * hStrideSrc) + (id_x + roiTensorPtrSrc[id_z].xywhROI.xy.x);

    dstIdx = (id_z * nStrideDst) + (id_y * hStrideDst) + id_x;

    rpp_hip_load8_and_unpack_to_float8(srcPtr, srcIdx, &src_f8);
    ricap_hip_compute(srcPtr, &src_f8, &dst_f8);
    rpp_hip_pack_float8_and_store8(dstPtr, dstIdx, &dst_f8);

    if (channelsDst == 3)
    {
        srcIdx += cStrideSrc;
        dstIdx += cStrideDst;

        rpp_hip_load8_and_unpack_to_float8(srcPtr, srcIdx, &src_f8);
        ricap_hip_compute(srcPtr, &src_f8, &dst_f8);
        rpp_hip_pack_float8_and_store8(dstPtr, dstIdx, &dst_f8);

        srcIdx += cStrideSrc;
        dstIdx += cStrideDst;

        rpp_hip_load8_and_unpack_to_float8(srcPtr, srcIdx, &src_f8);
        ricap_hip_compute(srcPtr, &src_f8, &dst_f8);
        rpp_hip_pack_float8_and_store8(dstPtr, dstIdx, &dst_f8);
    }
}

template <typename T>
__global__ void ricap_pkd3_pln3_tensor(T *srcPtr,
                                            int nStrideSrc,
                                            int hStrideSrc,
                                            T *dstPtr,
                                            int nStrideDst,
                                            int cStrideDst,
                                            int hStrideDst,
                                            uint *permutations1,
                                            uint *permutations2,
                                            uint *permutations3,
                                            uint *permutations4,
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

    uint srcIdx = (id_z * nStrideSrc) + ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * hStrideSrc) + ((id_x + roiTensorPtrSrc[id_z].xywhROI.xy.x) * 3);
    uint dstIdx = (id_z * nStrideDst) + (id_y * hStrideDst) + id_x;

    d_float24 src_f24, dst_f24;

    rpp_hip_load24_pkd3_and_unpack_to_float24_pln3(srcPtr, srcIdx, &src_f24);
    ricap_hip_compute(srcPtr, &src_f24.x, &dst_f24.x);
    rpp_hip_pack_float8_and_store8(dstPtr, dstIdx, &dst_f24.x);

    dstIdx += cStrideDst;

    ricap_hip_compute(srcPtr, &src_f24.y, &dst_f24.y);
    rpp_hip_pack_float8_and_store8(dstPtr, dstIdx, &dst_f24.y);

    dstIdx += cStrideDst;

    ricap_hip_compute(srcPtr, &src_f24.z, &dst_f24.z);
    rpp_hip_pack_float8_and_store8(dstPtr, dstIdx, &dst_f24.z);
}

template <typename T>
__global__ void ricap_pln3_pkd3_tensor(T *srcPtr,
                                            int nStrideSrc,
                                            int cStrideSrc,
                                            int hStrideSrc,
                                            T *dstPtr,
                                            int nStrideDst,
                                            int hStrideDst,
                                            uint *permutations1,
                                            uint *permutations2,
                                            uint *permutations3,
                                            uint *permutations4,
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

    uint srcIdx = (id_z * nStrideSrc) + ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * hStrideSrc) + (id_x + roiTensorPtrSrc[id_z].xywhROI.xy.x);
    uint dstIdx = (id_z * nStrideDst) + (id_y * hStrideDst) + id_x * 3;

    d_float24 src_f24, dst_f24;

    rpp_hip_load24_pln3_and_unpack_to_float24_pkd3(srcPtr, srcIdx, cStrideSrc, &src_f24);
    ricap_hip_compute(srcPtr, &src_f24.x, &dst_f24.x);
    ricap_hip_compute(srcPtr, &src_f24.y, &dst_f24.y);
    ricap_hip_compute(srcPtr, &src_f24.z, &dst_f24.z);
    rpp_hip_pack_float24_pkd3_and_store24_pkd3(dstPtr, dstIdx, &dst_f24);
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
    // else if ((srcDescPtr->c == 3) && (dstDescPtr->c == 3))
    // {
    //     if ((srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
    //     {
    //         hipLaunchKernelGGL(ricap_pkd3_pln3_tensor,
    //                            dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
    //                            dim3(localThreads_x, localThreads_y, localThreads_z),
    //                            0,
    //                            handle.GetStream(),
    //                            srcPtr,
    //                            srcDescPtr->strides.nStride,
    //                            srcDescPtr->strides.hStride,
    //                            dstPtr,
    //                            dstDescPtr->strides.nStride,
    //                            dstDescPtr->strides.cStride,
    //                            dstDescPtr->strides.hStride,
    //                            handle.GetInitHandle()->mem.mgpu.floatArr[0].floatmem,
    //                            handle.GetInitHandle()->mem.mgpu.floatArr[1].floatmem,
    //                            roiTensorPtrSrc);
    //     }
    //     else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
    //     {
    //         globalThreads_x = (srcDescPtr->strides.hStride + 7) >> 3;
    //         hipLaunchKernelGGL(ricap_pln3_pkd3_tensor,
    //                            dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
    //                            dim3(localThreads_x, localThreads_y, localThreads_z),
    //                            0,
    //                            handle.GetStream(),
    //                            srcPtr,
    //                            srcDescPtr->strides.nStride,
    //                            srcDescPtr->strides.cStride,
    //                            srcDescPtr->strides.hStride,
    //                            dstPtr,
    //                            dstDescPtr->strides.nStride,
    //                            dstDescPtr->strides.hStride,
    //                            handle.GetInitHandle()->mem.mgpu.floatArr[0].floatmem,
    //                            handle.GetInitHandle()->mem.mgpu.floatArr[1].floatmem,
    //                            roiTensorPtrSrc);
    //     }
    // }

    return RPP_SUCCESS;
}
