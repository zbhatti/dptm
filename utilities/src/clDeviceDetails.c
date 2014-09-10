#include <stdio.h>                                                                                                                                               
#include <stdlib.h>
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

int main() {

    int i, j;
    char* value;
    size_t valueSize;
    cl_uint platformCount;
    cl_platform_id* platforms;
    cl_uint deviceCount;
    cl_device_id* devices;
    cl_uint uintTmp;
		cl_ulong ulongTmp;
		size_t TmpSizeT;
		size_t workItemSizes[3];
		
    // get all platforms
    clGetPlatformIDs(0, NULL, &platformCount);
    platforms = (cl_platform_id*) malloc(sizeof(cl_platform_id) * platformCount);
    clGetPlatformIDs(platformCount, platforms, NULL);

    for (i = 0; i < platformCount; i++) {

        // get all devices
        clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 0, NULL, &deviceCount);
        devices = (cl_device_id*) malloc(sizeof(cl_device_id) * deviceCount);
        clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, deviceCount, devices, NULL);
				
        // for each device print critical attributes
        for (j = 0; j < deviceCount; j++) {

            // print device name
            clGetDeviceInfo(devices[j], CL_DEVICE_NAME, 0, NULL, &valueSize);
            value = (char*) malloc(valueSize);
            clGetDeviceInfo(devices[j], CL_DEVICE_NAME, valueSize, value, NULL);
            printf("%d. Device: %s\n", j+1, value);
            free(value);
						
						//print device type: CL_DEVICE_TYPE
						{}

            // print hardware device version
            clGetDeviceInfo(devices[j], CL_DEVICE_VERSION, 0, NULL, &valueSize);
            value = (char*) malloc(valueSize);
            clGetDeviceInfo(devices[j], CL_DEVICE_VERSION, valueSize, value, NULL);
            printf(" %d.%d Hardware version: %s\n", j+1, 1, value);
            free(value);

            // print software driver version
            clGetDeviceInfo(devices[j], CL_DRIVER_VERSION, 0, NULL, &valueSize);
            value = (char*) malloc(valueSize);
            clGetDeviceInfo(devices[j], CL_DRIVER_VERSION, valueSize, value, NULL);
            printf(" %d.%d Software version: %s\n", j+1, 2, value);
            free(value);
						
						//print size of global memory in bytes
						clGetDeviceInfo(devices[j], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(ulongTmp), &ulongTmp, NULL);
						printf(" %d.%d Global memory size: %ld MiB\n", j+1, 3, ulongTmp/1048576);
						
						//print size of global memory cache in bytes
						clGetDeviceInfo(devices[j], CL_DEVICE_GLOBAL_MEM_CACHE_SIZE, sizeof(ulongTmp), &ulongTmp, NULL);
						printf(" %d.%d Global memory cache size: %ld\n", j+1, 4, ulongTmp);
						
						//print size of local memory in bytes
						clGetDeviceInfo(devices[j], CL_DEVICE_LOCAL_MEM_SIZE , sizeof(ulongTmp), &ulongTmp, NULL);
						printf(" %d.%d Local memory size: %ld\n", j+1, 5, ulongTmp);
						
						// print parallel compute units
						clGetDeviceInfo(devices[j], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(uintTmp), &uintTmp, NULL);
						printf(" %d.%d Max compute units: %d\n", j+1, 6, uintTmp);

						//print device maximum clock frequency in MHz
						clGetDeviceInfo(devices[j], CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(uintTmp), &uintTmp, NULL);
						printf(" %d.%d Max Clock Frequency: %d\n", j+1, 7, uintTmp);
						
						//print device maximum memory allocation size
						clGetDeviceInfo(devices[j], CL_DEVICE_MAX_MEM_ALLOC_SIZE , sizeof(ulongTmp), &ulongTmp, NULL);
						printf(" %d.%d Maximum memory allocation: %ld MiB\n", j+1, 8, ulongTmp/1048576);
						
						//print device maximum constant buffer size
						clGetDeviceInfo(devices[j], CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE , sizeof(ulongTmp), &ulongTmp, NULL);
						printf(" %d.%d Maximum constant buffer size: %ld MiB\n", j+1, 9, ulongTmp/1048576);
						
						//print maximum work group size 
						clGetDeviceInfo(devices[j], CL_DEVICE_MAX_WORK_GROUP_SIZE , sizeof(TmpSizeT), &TmpSizeT, NULL);
						printf(" %d.%d Max work-items per work goup: %d\n", j+1, 10, TmpSizeT);
						
						//print maximum work item size CL_DEVICE_MAX_WORK_ITEM_SIZES
						clGetDeviceInfo(devices[j], CL_DEVICE_MAX_WORK_ITEM_SIZES , sizeof(workItemSizes), &workItemSizes, NULL);
            printf(" %d.%d Max items per dimension per group: \(%d,%d,%d)\n", j+1, 11, workItemSizes[0],workItemSizes[1],workItemSizes[2]);
						
						//print maximum work item size CL_DEVICE_MAX_WORK_ITEM_SIZES
						clGetDeviceInfo(devices[j], CL_DEVICE_ADDRESS_BITS , sizeof(uintTmp), &uintTmp, NULL);
            printf(" %d.%d Default address compute space size: \%d bytes\n", j+1, 12, uintTmp);
						printf("\n");
        }

        free(devices);

    }

    free(platforms);
    return 0;

}
/*
1. Device: Tesla K20Xm
 1.1 Hardware version: OpenCL 1.1 CUDA
 1.2 Software version: 331.67
 1.3 Global memory size: 6039339008
 1.4 Global memory cache size: 229376
 1.5 Local memory size: 49152
 1.6 Max compute units: 14
 1.7 Max Clock Frequency: 732
 1.8 Maximum memory allocation: 1509834752
 1.9 Maximum constant buffer size: 65536
 1.10 Max work-items per work goup: 1024
 1.11 Max items per dimension per group: (1024,1024,64)
 
1. Device: Hawaii
 1.1 Hardware version: OpenCL 1.2 AMD-APP (1411.4)
 1.2 Software version: 1411.4 (VM)
 1.3 Global memory size: 3221225472
 1.4 Global memory cache size: 16384
 1.5 Local memory size: 32768
 1.6 Max compute units: 44
 1.7 Max Clock Frequency: 930
 1.8 Maximum memory allocation: 1073741824
 1.9 Maximum constant buffer size: 65536
 1.10 Max work-items per work goup: 256
 1.11 Max items per dimension per group: (256,256,256)
 
2. Device: Intel(R) Xeon(R) CPU E5-2695 v2 @ 2.40GHz
 2.1 Hardware version: OpenCL 1.2 AMD-APP (1411.4)
 2.2 Software version: 1411.4 (sse2,avx)
 2.3 Global memory size: 135289802752
 2.4 Global memory cache size: 32768
 2.5 Local memory size: 32768
 2.6 Max compute units: 48
 2.7 Max Clock Frequency: 1200
 2.8 Maximum memory allocation: 33822450688
 2.9 Maximum constant buffer size: 65536
 2.10 Max work-items per work goup: 1024
 2.11 Max items per dimension per group: (1024,1024,1024)
 
1. Device:       Intel(R) Xeon(R) CPU E5-2695 v2 @ 2.40GHz
 1.1 Hardware version: OpenCL 1.2 (Build 44)
 1.2 Software version: 1.2.0.44
 1.3 Global memory size: 135289802752
 1.4 Global memory cache size: 262144
 1.5 Local memory size: 32768
 1.6 Max compute units: 48
 1.7 Max Clock Frequency: 2400
 1.8 Maximum memory allocation: 33822450688
 1.9 Maximum constant buffer size: 131072
 1.10 Max work-items per work goup: 8192
 1.11 Max items per dimension per group: (8192,8192,8192)
 
2. Device: Intel(R) Many Integrated Core Acceleration Card
 2.1 Hardware version: OpenCL 1.2 (Build 44)
 2.2 Software version: 1.2
 2.3 Global memory size: 12200046592
 2.4 Global memory cache size: 262144
 2.5 Local memory size: 32768
 2.6 Max compute units: 240
 2.7 Max Clock Frequency: 1333
 2.8 Maximum memory allocation: 4066680832
 2.9 Maximum constant buffer size: 131072
 2.10 Max work-items per work goup: 8192
 2.11 Max items per dimension per group: (8192,8192,8192)
*/

