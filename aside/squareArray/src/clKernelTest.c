#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <CL/opencl.h>
#include <math.h>
#include <time.h> //for random seed and timing
#define MAXPLATFORMS 5
#define MAXDEVICES 16
#define MAXFILELENGTH 1000000
#define BEGIN gettimeofday(&start, NULL);	
#define END gettimeofday(&finish, NULL); time=timeDiff(start, finish);

float timeDiff(struct timeval start, struct timeval finish){
	return (float) ((finish.tv_sec*1000000 + finish.tv_usec) - (start.tv_sec*1000000 + start.tv_usec))/(1000);
}

void generateArray(float *A, int N){
	int i;
	srand(time(NULL));
	for (i=0; i < N; i++){
		A[i] = rand()/(RAND_MAX/99.9 + 1);
	}
}

void kernelTest(cl_platform_id platform, cl_device_id device, char* kernelSource, float *Ain, float *Aout, int N){
		
		float time;
		struct timeval start, finish;
		
		cl_context context;
		cl_command_queue queue;
		cl_program program;
		cl_kernel kernel;
		size_t globalSize, localSize, workGroupSizeMultiple;
		cl_mem Ain_d, Aout_d;
		cl_int err;
		
		cl_context_properties props[3] = { CL_CONTEXT_PLATFORM, 0, 0 };
		props[1] = (cl_context_properties) platform;
		context = clCreateContext(props, 1, &device, NULL, NULL, &err);
		queue = clCreateCommandQueue(context, device, 0, &err);
		program = clCreateProgramWithSource(context, 1, (const char **) &kernelSource, NULL, &err);
		clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
		kernel = clCreateKernel(program, "squareElements", &err);
		
		//allocate memory on the device:
		Ain_d = clCreateBuffer(context, CL_MEM_READ_WRITE, N*sizeof(float), NULL, &err);
		Aout_d = clCreateBuffer(context, CL_MEM_READ_WRITE, N*sizeof(float), NULL, &err);
		
		//memcpyOut
		BEGIN
		err = clEnqueueWriteBuffer(queue, Ain_d, CL_TRUE, 0, N*sizeof(float), Ain, 0, NULL, NULL);
		err = clFinish(queue);
		END
		printf("\t out: %0.3f ms\n",time);
		
		//set kernel arguments
		err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &Ain_d);
		err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &Aout_d);
		err = clSetKernelArg(kernel, 2, sizeof(int), &N);
		err = clGetKernelWorkGroupInfo(kernel, device, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &localSize, NULL);
		
		globalSize=N - N%localSize + localSize; //must get every element in the array
		printf("\t globalSize: %d, localSize: %d\n", globalSize, localSize);
		
		//launch kernel
		BEGIN
		err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalSize, &localSize, 0, NULL, NULL);
		err = clFinish(queue);
		END
		printf("\t kernel: %0.3f ms\n",time);
		
		//memcpyIn
		BEGIN
		err = clEnqueueReadBuffer(queue, Aout_d, CL_TRUE, 0, N*sizeof(float), Aout, 0, NULL, NULL);
		err = clFinish(queue);
		END
		printf("\t in:  %0.3f ms,\n",time);
		printf("Ain: %0.2f, Aout: %0.2f\n",Ain[0], Aout[0]);
		
		err=clReleaseMemObject(Ain_d);
		err=clReleaseMemObject(Aout_d);
		err=clReleaseCommandQueue(queue);
		err=clReleaseContext(context);
		err=clReleaseKernel(kernel);
		err=clReleaseProgram(program);
		
}

int main(int argc, char *argv[]){
  int i, j;
	int N;
  char name[128];
	N = 412351;
	float *Aout, *Ain;
	
	//open kernel file and load it into kernelString:
	char kernelString[MAXFILELENGTH];
	long fileLength;
	char fileName[] = "/home/zxb0111/Samples/src/squareElements.cl";
	FILE *fp = fopen(fileName, "r");
  fseek (fp, 0, SEEK_END);
  fileLength = ftell(fp);
  rewind (fp);
	fread (kernelString, 1, fileLength, fp);
	fclose(fp);
	
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
    printf("platform %s with %d devices\n", name, num_devices);
    for (j = 0; j < num_devices; j++){
      clGetDeviceInfo(devices[j], CL_DEVICE_NAME, sizeof(name), name, NULL);
      printf("\tdevice %d %s\n", j, name);
			kernelTest(platforms[i], devices[j], kernelString, Ain, Aout, N);
    }
  }
  
  return 0;
}
