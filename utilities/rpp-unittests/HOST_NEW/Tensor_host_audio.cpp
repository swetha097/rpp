#include <stdio.h>
#include <dirent.h>
#include <string.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include "/opt/rocm/rpp/include/rpp.h"
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <time.h>
#include <omp.h>
#include <half.hpp>
#include <fstream>

/* Include this header file to use functions from libsndfile. */
#include <sndfile.h>
#include <dirent.h>

/* libsndfile can handle more than 6 channels but we'll restrict it to 6. */
#define		MAX_CHANNELS	6

using namespace cv;
using namespace std;
using half_float::half;

typedef half Rpp16f;

#define RPPPIXELCHECK(pixel) (pixel < (Rpp32f)0) ? ((Rpp32f)0) : ((pixel < (Rpp32f)255) ? pixel : ((Rpp32f)255))
#define RPPMAX2(a,b) ((a > b) ? a : b)
#define RPPMIN2(a,b) ((a < b) ? a : b)

int main(int argc, char **argv)
{
    // Handle inputs

    const int MIN_ARG_COUNT = 8;

    if (argc < MIN_ARG_COUNT)
    {
        printf("\nImproper Usage! Needs all arguments!\n");
        printf("\nUsage: ./Tensor_host_pkd3 <src1 folder> <src2 folder (place same as src1 folder for single image functionalities)> <dst folder> <u8 = 0 / f16 = 1 / f32 = 2 / u8->f16 = 3 / u8->f32 = 4 / i8 = 5 / u8->i8 = 6> <outputFormatToggle (pkd->pkd = 0 / pkd->pln = 1)> <case number = 0:84> <verbosity = 0/1>\n");
        return -1;
    }

    char *src = argv[1];
    // char *src_second = argv[2];
    // char *dst = argv[3];
    int ip_bitDepth = atoi(argv[2]);
    unsigned int outputFormatToggle = atoi(argv[3]);
    int test_case = atoi(argv[4]);

    bool additionalParamCase = (test_case == 21);
    bool kernelSizeCase = false;
    bool interpolationTypeCase = (test_case == 21);

    unsigned int verbosity = 0;//additionalParamCase ? atoi(argv[8]) : atoi(argv[7]);
    unsigned int additionalParam = 0;//additionalParamCase ? atoi(argv[7]) : 1;

    if (verbosity == 1)
    {
        printf("\nInputs for this test case are:");
        printf("\nsrc1 = %s", argv[1]);
        printf("\nu8 / f16 / f32 / u8->f16 / u8->f32 / i8 / u8->i8 (0/1/2/3/4/5/6) = %s", argv[2]);
        printf("\noutputFormatToggle (pkd->pkd = 0 / pkd->pln = 1) = %s", argv[3]);
        printf("\ncase number (0:84) = %s", argv[4]);
    }

    int ip_channel = 1;

    // Set case names

    char funcType[1000] = {"Tensor_HOST_PKD3"};
    char funcName[1000];
    switch (test_case)
    {
        case 0:
            strcpy(funcName, "non_silent_region_detection");
            break;
        default:
            strcpy(funcName, "test_case");
            break;
    }

    // Initialize tensor descriptors

    RpptDesc srcDesc;
    RpptDescPtr srcDescPtr;
    srcDescPtr = &srcDesc;

    // Set src/dst data types in tensor descriptors
    if (ip_bitDepth == 0)
    {
        strcat(funcName, "_u8_");
        srcDescPtr->dataType = RpptDataType::U8;
    }
    else if (ip_bitDepth == 1)
    {
        strcat(funcName, "_f16_");
        srcDescPtr->dataType = RpptDataType::F16;
    }
    else if (ip_bitDepth == 2)
    {
        strcat(funcName, "_f32_");
        srcDescPtr->dataType = RpptDataType::F32;
    }
    else if (ip_bitDepth == 5)
    {
        strcat(funcName, "_i8_");
        srcDescPtr->dataType = RpptDataType::I8;
    }

    // Other initializations

    int missingFuncFlag = 0;
    int i = 0, j = 0;
    int maxHeight = 0, maxLength = 0;
    int maxDstHeight = 0, maxDstWidth = 0;
    unsigned long long count = 0;
    unsigned long long ioBufferSize = 0;
    static int noOfAudioFiles = 0;

    // String ops on function name
    char src1[1000];
    strcpy(src1, src);
    strcat(src1, "/");

    char func[1000];
    strcpy(func, funcName);
    printf("\nRunning %s...", func);

    // Get number of audio files
    struct dirent *de;
    DIR *dr = opendir(src);
    while ((de = readdir(dr)) != NULL)
    {
        if (strcmp(de->d_name, ".") == 0 || strcmp(de->d_name, "..") == 0)
            continue;
        noOfAudioFiles += 1;
    }
    closedir(dr);

    // Initialize the AudioPatch for source
    Rpp32u *srcAudioSizes = (Rpp32u *) calloc(noOfAudioFiles, sizeof(Rpp32u));

    // Set maxLength
    const int audio_files = noOfAudioFiles;
    char audioNames[audio_files][1000];

    DIR *dr1 = opendir(src);
    while ((de = readdir(dr1)) != NULL)
    {
        if (strcmp(de->d_name, ".") == 0 || strcmp(de->d_name, "..") == 0)
            continue;
        strcpy(audioNames[count], de->d_name);
        char temp[1000];
        strcpy(temp, src1);
        strcat(temp, audioNames[count]);

        SNDFILE	*infile;
        SF_INFO sfinfo;
        int	readcount;
        
        //The SF_INFO struct must be initialized before using it
        memset (&sfinfo, 0, sizeof (sfinfo));
        if (! (infile = sf_open (temp, SFM_READ, &sfinfo)))
        {	
            cerr<<"Not able to open input file"<<endl;
            
            /* Print the error message from libsndfile. */
            puts (sf_strerror (NULL));
            return 1;
        }
        
        if (sfinfo.channels > MAX_CHANNELS)
        {	printf ("Not able to process more than %d channels\n", MAX_CHANNELS) ;
            sf_close (infile);
        }

        srcAudioSizes[i] = sfinfo.frames * sfinfo.channels;
        maxLength = RPPMAX2(maxLength, srcAudioSizes[i]);
 
        /* Close input*/
        sf_close (infile);
        count++;
    }
    closedir(dr1);

    // Set numDims, offset, n/c/h/w values for src/dst
    srcDescPtr->numDims = 4;
    srcDescPtr->offsetInBytes = 0;
    srcDescPtr->n = noOfAudioFiles;    
    srcDescPtr->h = 1;
    srcDescPtr->w = maxLength;
    srcDescPtr->c = ip_channel;

    // Set n/c/h/w strides for src/dst
    srcDescPtr->strides.nStride = ip_channel * srcDescPtr->w * srcDescPtr->h;
    srcDescPtr->strides.hStride = ip_channel * srcDescPtr->w;
    srcDescPtr->strides.wStride = ip_channel;
    srcDescPtr->strides.cStride = 1;

    // Set buffer sizes for src/dst
    ioBufferSize = (unsigned long long)srcDescPtr->h * (unsigned long long)srcDescPtr->w * (unsigned long long)ip_channel * (unsigned long long)noOfAudioFiles;

    // Initialize host buffers for src/dst
    Rpp8u *input = (Rpp8u *)calloc(ioBufferSize, sizeof(Rpp8u));
    Rpp16f *inputf16 = (Rpp16f *)calloc(ioBufferSize, sizeof(Rpp16f));
    Rpp32f *inputf32 = (Rpp32f *)calloc(ioBufferSize, sizeof(Rpp32f));
    Rpp8s *inputi8 = (Rpp8s *)calloc(ioBufferSize, sizeof(Rpp8s));
    
    Rpp32u *detectionIndex = (Rpp32u *)calloc(ioBufferSize, sizeof(Rpp32u));
    Rpp32u *detectionLength = (Rpp32u *)calloc(ioBufferSize, sizeof(Rpp32u));

    // Convert inputs to test various other bit depths
    if (ip_bitDepth == 1)
    {
        Rpp8u *inputTemp;
        Rpp16f *inputf16Temp;

        inputTemp = input;
        inputf16Temp = inputf16;

        for (int i = 0; i < ioBufferSize; i++)
        {
            *inputf16Temp = ((Rpp16f)*inputTemp) / 255.0;
            inputTemp++;
            inputf16Temp++;
        }
    }
    else if (ip_bitDepth == 2)
    {
        Rpp8u *inputTemp;
        Rpp32f *inputf32Temp;

        inputTemp = input;
        inputf32Temp = inputf32;

        for (int i = 0; i < ioBufferSize; i++)
        {
            *inputf32Temp = ((Rpp32f)*inputTemp) / 255.0;
            inputTemp++;
            inputf32Temp++;
        }
    }
    else if (ip_bitDepth == 5)
    {
        Rpp8u *inputTemp;
        Rpp8s *inputi8Temp;

        inputTemp = input;
        inputi8Temp = inputi8;

        for (int i = 0; i < ioBufferSize; i++)
        {
            *inputi8Temp = (Rpp8s) (((Rpp32s) *inputTemp) - 128);
            inputTemp++;
            inputi8Temp++;
        }
    }

    // Run case-wise RPP API and measure time
    rppHandle_t handle;
    rppCreateWithBatchSize(&handle, noOfAudioFiles);
    clock_t start, end;
    double start_omp, end_omp;
    double cpu_time_used, omp_time_used;

    string test_case_name;

    switch (test_case)
    { 
    case 0:
    {
        Rpp32u batchSize = 1;
        test_case_name = "audio_test";
        Rpp32s detectedIndex[batchSize];
        Rpp32s detectionLength[batchSize];
        Rpp32f cutOffDB[batchSize];
        Rpp32s windowLength[batchSize];
        Rpp32f referencePower[batchSize];
        Rpp32s resetInterval[batchSize];
        bool referenceMax[batchSize];
        Rpp32s audioLength[batchSize];
        srcDescPtr->n = batchSize;
     
        for (i = 0; i < batchSize; i++)
        {
            detectedIndex[i] = 0;
            detectionLength[i] = 0;
            cutOffDB[i] = -60.0;
            windowLength[i] = 3;
            referencePower[i] = 1.0;
            resetInterval[i] = -1;
            referenceMax[i] = true;
        }
 
        start_omp = omp_get_wtime();
        start = clock();
        if (ip_bitDepth == 2)
        {
            rppt_non_silent_region_detection_host(inputf32, srcDescPtr, audioLength, detectedIndex, detectionLength, cutOffDB, windowLength, referencePower, resetInterval, referenceMax, handle);
        }
        else
            missingFuncFlag = 1;
        
        //Print the detection index and length
        cout<<endl<<"Index, Length: "<<detectedIndex[0]<<" "<<detectionLength[0];
        
        break;
    }
    default:
        missingFuncFlag = 1;
        break;
    }

    end = clock();
    end_omp = omp_get_wtime();

    if (missingFuncFlag == 1)
    {
        printf("\nThe functionality %s doesn't yet exist in RPP\n", func);
        return -1;
    }

    // Display measured times

    cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
    omp_time_used = end_omp - start_omp;
    cout << "\nCPU Time - BatchPD : " << cpu_time_used;
    cout << "\nOMP Time - BatchPD : " << omp_time_used;
    printf("\n");

    rppDestroyHost(handle);

    // Free memory

    free(input);
    free(inputf16);
    free(inputf32);
    free(inputi8);
    free(detectionIndex);
    free(detectionLength);

    return 0;
}
