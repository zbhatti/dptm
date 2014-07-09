#include <stddef.h>
#include <sys/types.h>
#include <unistd.h>
#include <stdlib.h>
#include <sys/un.h>
#include <math.h>
#include <stdio.h>
#include <sys/time.h> //for random seed and timing
#include <errno.h>
#include <string.h>
#ifdef __APPLE__
	#include <OpenCL/opencl.h>
#else
	#include <CL/opencl.h>
#endif
#define BEGIN gettimeofday(&start, NULL);	
#define END gettimeofday(&finish, NULL); time=timeDiff(start, finish);
#define MAXPLATFORMS 5
#define MAXDEVICES 16
#define MAXFILELENGTH 1000000

void generateArray(float *A, int N){
	int i;
	srand(time(NULL));
	for (i=0; i < N; i++){
		A[i] = rand()/(RAND_MAX/99999.9 + 1);
	}
}


float timeDiff(struct timeval start, struct timeval finish){
	return (float) ((finish.tv_sec*1000000 + finish.tv_usec) - (start.tv_sec*1000000 + start.tv_usec))/(1000);
}

int main(int argc, char *argv[]){
	int Asize, i,r ;
	float *Ain, *Aout;
	float time;
	struct timeval start, finish;
	char name[128];
	
	int PlatformChoice=0;
	int DeviceChoice=0;
	int MaxVectors=2000;
	int MinVectors=0;
	int StepSize=100;
	int Repeat=1;
		
	if (argc>1)
		PlatformChoice=atoi(argv[1]);
	
	if (argc>2)
		DeviceChoice=atoi(argv[2]);
	
	if (argc>3)
	  MaxVectors=atoi(argv[3]);
		
	if (argc>4)
	  MinVectors=atoi(argv[4]);	
		
	if (argc>5)
	  StepSize=atoi(argv[5]);

	if (argc>6)
	  Repeat=atoi(argv[6]);
		
	//declaring general opencl variables
	cl_context context;
	cl_command_queue queue;
	cl_program program;
	cl_kernel kernel;
	size_t globalSize, localSize;
	cl_device_id device_ids[MAXDEVICES];
	cl_platform_id platform_ids[MAXPLATFORMS];
	cl_mem Ain_d, Aout_d;
	unsigned int numDevices;
	cl_int err;
	
	cl_context_properties props[3] = { CL_CONTEXT_PLATFORM, 0, 0 };

	//open kernel file and load it into kernelString:
	char * kernelString;
	kernelString = (char*)malloc(MAXFILELENGTH);
	long fileLength;
	char fileName[] = "/home/zxb0111/Samples/src/squareElements.cl";
	FILE *fp = fopen(fileName, "r");
  fseek (fp, 0, SEEK_END);
  fileLength = ftell(fp);
  rewind (fp);
	fread (kernelString, 1, fileLength, fp);
	fclose(fp);
	
	//getting information names of Platforms and Devices:
	err=clGetPlatformIDs(MAXPLATFORMS, platform_ids, NULL);	
	err=clGetDeviceIDs(platform_ids[PlatformChoice], CL_DEVICE_TYPE_ALL, MAXDEVICES, device_ids, &numDevices);
	err=clGetPlatformInfo(platform_ids[PlatformChoice], CL_PLATFORM_NAME, sizeof(name), name, NULL);
	fprintf(stderr,"On Platform %s\n", name);
	err=clGetDeviceInfo(device_ids[DeviceChoice], CL_DEVICE_NAME, sizeof(name), name, NULL);
	fprintf(stderr,"using device %s\n", name);
	
	//starting up opencl on the selected device and platform:
	props[1] = (cl_context_properties) platform_ids[PlatformChoice];
	context = clCreateContext(props, 1, &device_ids[DeviceChoice], NULL, NULL, &err);
	queue = clCreateCommandQueue( context, device_ids[DeviceChoice], 0, &err);
	program = clCreateProgramWithSource(context, 1, (const char **) &kernelString, NULL, &err);
	err=clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
	kernel = clCreateKernel(program, "squareElements", &err);
	
	//allocate local memory for original array
	Ain = (float*) malloc(MaxVectors*sizeof(float));
	if (!Ain){
		fprintf(stderr,"error in Ain malloc");
	}
	
	//allocate local memory for return array
	Aout = (float*) malloc(MaxVectors*sizeof(float));
	if (!Aout){
		fprintf(stderr,"error in Aout malloc");
	}
	
	generateArray(Ain, MaxVectors);
	printf("nVectors,memcpyOut,plan,execu,memcpyIn\n");
	
	for (r=0;r<Repeat;r++) {
		for(i=MinVectors; i<= MaxVectors; i+=StepSize){
		
			Asize = i;
			if (i == 0)
				Asize = 1;
			printf("%d,", Asize);
			fflush(stdout);
			
			//memcpyOut
			BEGIN
			Ain_d = clCreateBuffer(context, CL_MEM_READ_WRITE, Asize*sizeof(float), NULL, &err);
			Aout_d = clCreateBuffer(context, CL_MEM_READ_WRITE, Asize*sizeof(float), NULL, &err);
			err = clEnqueueWriteBuffer(queue, Ain_d, CL_TRUE, 0, Asize*sizeof(float), Ain, 0, NULL, NULL);
			err = clFinish(queue);
			END
			printf("%0.3f,",time);
			fflush(stdout);
			
			//set kernel arguments (plan)
			BEGIN
			err = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*) &Ain_d);
			err = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*) &Aout_d);
			err = clSetKernelArg(kernel, 2, sizeof(int), &Asize);
			//OPTIMAL: err = clGetKernelWorkGroupInfo(kernel, device_ids[DeviceChoice], CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &localSize, NULL);
			localSize = 64;
			globalSize = Asize - Asize%localSize + localSize;
			END
			printf("%0.3f,",time);
			fflush(stdout);
			
			//execute
			BEGIN
			err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalSize, &localSize, 0, NULL, NULL);
			err = clFinish(queue);
			END
			printf("%0.3f,",time);
			fflush(stdout);			
			
			//memcpyIn
			BEGIN
			err = clEnqueueReadBuffer(queue, Aout_d, CL_TRUE, 0, Asize*sizeof(float), Aout, 0, NULL, NULL);
			err = clFinish(queue);
			END
			printf("%0.3f,\n",time);
			fflush(stdout);	
			
			err=clReleaseMemObject(Ain_d);
			err=clReleaseMemObject(Aout_d);
			
			//printf("Ain: %0.2f, Aout: %0.2f\n",Ain[Asize-1], Aout[Asize-1]);
		}
	}
	
	err=clReleaseCommandQueue(queue);
	err=clReleaseContext(context);
	err=clReleaseKernel(kernel);
	err=clReleaseProgram(program);
	
	fflush(stdout);
	free(Ain);
	free(Aout);
	return 0;
}

