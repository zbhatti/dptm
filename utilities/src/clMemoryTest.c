#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <CL/opencl.h>
#include <time.h> //for random seed and timing
#define MAXPLATFORMS 5
#define MAXDEVICES 16
#define BEGIN gettimeofday(&start, NULL);	
#define END gettimeofday(&finish, NULL); time=timeDiff(start, finish);

float timeDiff(struct timeval start, struct timeval finish){
	return (float) ((finish.tv_sec*1000000 + finish.tv_usec) - (start.tv_sec*1000000 + start.tv_usec))/(1000);
}

void generateArray(float *A, int N){
	int i;
	for (i=0; i < N; i++){
		A[i] = rand();
	}
}

void memTest(cl_platform_id platform, cl_device_id device, float *Ain, float *Aout, int N){
		
		float time;
		struct timeval start, finish;
		cl_command_queue queue;
		cl_context ctx = 0;
		cl_mem buffer;
		cl_int err;
		cl_context_properties props[3] = { CL_CONTEXT_PLATFORM, 0, 0 };
		props[1] = (cl_context_properties) platform;
		ctx = clCreateContext(props, 1, &device, NULL, NULL, &err);
		queue = clCreateCommandQueue( ctx, device, 0, &err);
		
		//memcpyOut
		BEGIN
		buffer = clCreateBuffer(ctx, CL_MEM_READ_WRITE, N*sizeof(float), NULL, &err);
		err = clEnqueueWriteBuffer(queue, buffer, CL_TRUE, 0, N*sizeof(float), Ain, 0, NULL, NULL);
		err = clFinish(queue);
		END
		printf("\t out: %0.3f ms, ",time);

		//memcpyIn
		BEGIN
		err = clEnqueueReadBuffer(queue, buffer, CL_TRUE, 0, N*sizeof(float), Aout, 0, NULL, NULL);
		err = clFinish(queue);
		END
		printf("\t in:  %0.3f ms,\n",time);
		
		clReleaseMemObject(buffer);
		clReleaseCommandQueue(queue);
		clReleaseContext(ctx);
		
}

int main(int argc, char *argv[]){
  int i, j;
	int N;
  char name[128];
	N = 50000000;
	float *Aout, *Ain;
	
	Aout = (float*) malloc(N*sizeof(float));
	if (!Aout){
		fprintf(stderr,"error in Aout malloc");
	}
	
	Ain = (float*) malloc(N*sizeof(float));
	if (!Ain){
		fprintf(stderr,"error in Ain malloc");
	}
	
	printf("total Data: %0.3f Megabytes\n", (float) N*sizeof(float)/1048576);
	generateArray(Ain, N);
	
  //starting up opencl
  cl_device_id devices[MAXDEVICES];
  cl_platform_id platforms[MAXPLATFORMS];
  unsigned int num_devices, num_platforms;
  
  clGetPlatformIDs(MAXPLATFORMS, platforms, &num_platforms);
  printf("number of platforms found: %d\n", num_platforms);
  for (i = 0; i < num_platforms; i++){
    clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, sizeof(name), name, NULL);
    clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, MAXDEVICES, devices, &num_devices);
   //list devices available on platform
    printf("platform %s with %d devices\n", name, num_devices);
    for (j = 0; j < num_devices; j++){
      clGetDeviceInfo(devices[j], CL_DEVICE_NAME, sizeof(name), name, NULL);
      printf("\tdevice %d %s\n", j, name);
			memTest(platforms[i], devices[j], Ain, Aout, N);
    }
  }
  
  return 0;
}
