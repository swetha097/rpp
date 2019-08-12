#include <cl/rpp_cl_common.hpp>
#include "cl_declarations.hpp"
// /********************** Blur ************************/
// float box_3x3[] = {
// 0.111, 0.111, 0.111,
// 0.111, 0.111, 0.111,
// 0.111, 0.111, 0.111,
// };


// cl_int
// box_filter_cl(cl_mem srcPtr, RppiSize srcSize,
//                 cl_mem dstPtr, unsigned int filterSize,
//                 RppiChnFormat chnFormat, unsigned int channel,
//                 cl_command_queue theQueue)
// {
//     unsigned short counter=0;
//     cl_int err;

//     float* filterBuffer;
//     if (filterSize == 3) filterBuffer= box_3x3;
//     else  std::cerr << "Unimplemeted kernel Size";

//     cl_context theContext;
//     clGetCommandQueueInfo(  theQueue,
//                             CL_QUEUE_CONTEXT,
//                             sizeof(cl_context), &theContext, NULL);
//     cl_mem filtPtr = clCreateBuffer(theContext, CL_MEM_READ_ONLY,
//                                     sizeof(float)*filterSize*filterSize, NULL, NULL);
//     err = clEnqueueWriteBuffer(theQueue, filtPtr, CL_TRUE, 0,
//                                    sizeof(float)*filterSize*filterSize,
//                                    filterBuffer, 0, NULL, NULL);


//     cl_kernel theKernel;
//     cl_program theProgram;


//     if (chnFormat == RPPI_CHN_PLANAR)
//     {
//         CreateProgramFromBinary(theQueue,"convolution.cl","convolution.cl.bin","naive_convolution_planar",theProgram,theKernel);
//         clRetainKernel(theKernel); 

//         // cl_kernel_initializer(  theQueue, "convolution.cl",
//         //                         "naive_convolution_planar", theProgram, theKernel);

//     }
//     else if (chnFormat == RPPI_CHN_PACKED)
//     {
//         CreateProgramFromBinary(theQueue,"convolution.cl","convolution.cl.bin","naive_convolution_packed",theProgram,theKernel);
//         clRetainKernel(theKernel); 

//         // cl_kernel_initializer(  theQueue, "convolution.cl",
//         //                         "naive_convolution_packed", theProgram, theKernel);
//     }
//     else
//     {std::cerr << "Internal error: Unknown Channel format";}




//     err  = clSetKernelArg(theKernel, counter++, sizeof(cl_mem), &srcPtr);
//     err |= clSetKernelArg(theKernel, counter++, sizeof(cl_mem), &dstPtr);
//     err |= clSetKernelArg(theKernel, counter++, sizeof(cl_mem), &filtPtr);
//     err |= clSetKernelArg(theKernel, counter++, sizeof(unsigned int), &srcSize.height);
//     err |= clSetKernelArg(theKernel, counter++, sizeof(unsigned int), &srcSize.width);
//     err |= clSetKernelArg(theKernel, counter++, sizeof(unsigned int), &channel);
//     err |= clSetKernelArg(theKernel, counter++, sizeof(unsigned int), &filterSize);

// //----
//     size_t gDim3[3];
//     gDim3[0] = srcSize.width;
//     gDim3[1] = srcSize.height;
//     gDim3[2] = channel;
//     cl_kernel_implementer (theQueue, gDim3, NULL/*Local*/, theProgram, theKernel);

//     clReleaseMemObject(filtPtr);

// }
