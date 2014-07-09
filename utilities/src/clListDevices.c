#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <CL/opencl.h>
#define MAXPLATFORMS 5
#define MAXDEVICES 16

int main(int argc, char *argv[]){
  int i, j;
  char name[128];
  char platname[128];  

  //starting up opencl
  cl_device_id devices[MAXDEVICES];
  cl_platform_id platforms[MAXPLATFORMS];
  unsigned int num_devices;
  unsigned int num_platforms;
  
  clGetPlatformIDs(MAXPLATFORMS, platforms, &num_platforms);
  printf("number of platforms found: %d\n", num_platforms);
 
  for (i = 0; i < num_platforms; i++){
    clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, sizeof(platname), platname, NULL);
    clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, MAXDEVICES, devices, &num_devices);
   //list devices available on platform
    printf("platform %s with %d devices\n", platname, num_devices);
    for (j = 0; j < num_devices; j++){
      clGetDeviceInfo(devices[j], CL_DEVICE_NAME, sizeof(name), name, NULL);
      printf("   device %d %s\n", j, name);
    }


  }
  
  return 0;
}

