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
		A[i] = rand()/(RAND_MAX/999.9 + 1);
		fflush(stdout);
	}
}

float timeDiff(struct timeval start, struct timeval finish){
	return (float) ((finish.tv_sec*1000000 + finish.tv_usec) - (start.tv_sec*1000000 + start.tv_usec))/(1000);
}

int main(int argc, char *argv[]){
	int i,r,nDim;
	float *Ain, *Aout;
	float time;
	struct timeval start, finish;
	char name[128];
	
	int PlatformChoice=0;
	int DeviceChoice=0;
	int Asize=2000;
	int Repeat=1;
		
	if (argc>1)
		PlatformChoice=atoi(argv[1]);
	
	if (argc>2)
		DeviceChoice=atoi(argv[2]);
	
	if (argc>3)
	  Asize=atoi(argv[3]);

	if (argc>4)
	  Repeat=atoi(argv[4]);
		
	//declaring general opencl variables
	cl_context context;
	cl_command_queue queue;
	cl_program program;
	cl_kernel kernel;
	size_t globalSize[3], localSize[3];
	cl_device_id device_ids[MAXDEVICES];
	cl_platform_id platform_ids[MAXPLATFORMS];
	cl_mem Ain_d, Aout_d;
	unsigned int numDevices;
	cl_int err;
	size_t MaxWorkGroupSize;
	
	cl_context_properties props[3] = { CL_CONTEXT_PLATFORM, 0, 0 };

	//open kernel file and load it into kernelString:
	char * kernelString;
	kernelString = (char*)malloc(MAXFILELENGTH);
	long fileLength;
	char fileName[] = "/home/zxb0111/dptm/aside/squareArray/src/squareElements.cl";
	FILE *fp = fopen(fileName, "r");
  fseek (fp, 0, SEEK_END);
  fileLength = ftell(fp);
  rewind (fp);
	fread (kernelString, 1, fileLength, fp);
	fclose(fp);
	
	//getting information of Platforms and Devices:
	err=clGetPlatformIDs(MAXPLATFORMS, platform_ids, NULL);	
	err=clGetDeviceIDs(platform_ids[PlatformChoice], CL_DEVICE_TYPE_ALL, MAXDEVICES, device_ids, &numDevices);
	err=clGetPlatformInfo(platform_ids[PlatformChoice], CL_PLATFORM_NAME, sizeof(name), name, NULL);
	fprintf(stderr,"On Platform %s\n", name);
	err=clGetDeviceInfo(device_ids[DeviceChoice], CL_DEVICE_NAME, sizeof(name), name, NULL);
	err=clGetDeviceInfo(device_ids[DeviceChoice], CL_DEVICE_MAX_WORK_GROUP_SIZE , sizeof(MaxWorkGroupSize), &MaxWorkGroupSize, NULL);
	fprintf(stderr,"using device %s\n", name);
	
	//starting up opencl on the selected device and platform:
	props[1] = (cl_context_properties) platform_ids[PlatformChoice];
	context = clCreateContext(props, 1, &device_ids[DeviceChoice], NULL, NULL, &err);
	queue = clCreateCommandQueue( context, device_ids[DeviceChoice], 0, &err);
	program = clCreateProgramWithSource(context, 1, (const char **) &kernelString, NULL, &err);
	err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
	kernel = clCreateKernel(program, "squareElements", &err);
	
	//allocate local memory for original array
	Ain = (float*) malloc(Asize*sizeof(float));
	if (!Ain){
		fprintf(stderr,"error in Ain malloc");
	}
	
	//allocate local memory for return array
	Aout = (float*) malloc(Asize*sizeof(float));
	if (!Aout){
		fprintf(stderr,"error in Aout malloc");
	}
	
	generateArray(Ain, Asize);
	printf("nDim,nVectors,memcpyOut,plan,execu,memcpyIn\n");
	
	
	for (nDim=1; nDim<4; nDim++){
		for (r=0;r<Repeat;r++) {
			printf("%d,\t%d,", nDim,Asize);
			fflush(stdout);
			
			//memcpyOut
			BEGIN
			Ain_d = clCreateBuffer(context, CL_MEM_READ_WRITE, Asize*sizeof(float), NULL, &err);
			Aout_d = clCreateBuffer(context, CL_MEM_READ_WRITE, Asize*sizeof(float), NULL, &err);
			err = clEnqueueWriteBuffer(queue, Ain_d, CL_TRUE, 0, Asize*sizeof(float), Ain, 0, NULL, NULL);
			err = clFinish(queue);
			END
			printf("\t%0.3f,",time);
			fflush(stdout);
			
			//set kernel arguments (plan)
			BEGIN
			err = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*) &Ain_d);
			err = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*) &Aout_d);
			err = clSetKernelArg(kernel, 2, sizeof(int), (void*) &Asize);
			
			//the localSize[] assignments will only fail on a device where the maximum work dimension localSize[2]<<<localSize[0,1]
			//must have localSize[0]*localSize[1]*localSize[2] < CL_DEVICE_MAX_WORK_GROUP_SIZE
			localSize[0]=localSize[1]=localSize[2]=(int)pow(MaxWorkGroupSize, 1/(float)nDim);
				
			//globalSize must hold atleast all the elements of the array; it tells us the total number of work-items
			//want globalSize[0]*globalSize[1]*globalSize[2] >= N
			//globalSize in 1 dimension: globalSize={N,1,1}
			//globalSize in 2 dimension: globalSize={sqrt(N),sqrt(N),1}
			//globalSize in 3 dimension: globalSize={cbrt(N),cbrt(N),cbrt(N)}
			globalSize[0]=globalSize[1]=globalSize[2]=((int)pow(Asize, 1/(float)nDim)) - ((int)pow(Asize, 1/(float)nDim))%localSize[0] + localSize[0];

			//printf("\n0: %d divby %d =%f", globalSize[0], localSize[0],(float)globalSize[0]/(float)localSize[0]);
			//printf("\n1: %d divby %d =%f", globalSize[1], localSize[1],(float)globalSize[1]/(float)localSize[1]);
			//printf("\n2: %d divby %d =%f\n", globalSize[2], localSize[2],(float)globalSize[2]/(float)localSize[2]);

			END
			printf("\t%0.3f,",time);
			fflush(stdout);
		
			//execute
			BEGIN
			err = clEnqueueNDRangeKernel(queue, kernel, nDim, NULL, globalSize, localSize, 0, NULL, NULL);
			err = clFinish(queue); 
			END
			printf("\t%0.3f,",time);
			fflush(stdout);			
		
			//memcpyIn
			BEGIN
			err = clEnqueueReadBuffer(queue, Aout_d, CL_TRUE, 0, Asize*sizeof(float), Aout, 0, NULL, NULL);
			err = clFinish(queue);
			END
			printf("\t%0.3f,\n",time);
			fflush(stdout);	
		
			err=clReleaseMemObject(Ain_d);
			err=clReleaseMemObject(Aout_d);
		}
	}
	
	printf("Ain: %0.2f, Aout: %0.2f\n",Ain[Asize-1], Aout[Asize-1]);
	//for(i=0;i<Asize;i++){
		//printf("i:%d,",i);
		//printf("Ain:%f, Aout:%f\n", Ain[i],Aout[i]);
	//}
	
	err=clReleaseCommandQueue(queue);
	err=clReleaseContext(context);
	err=clReleaseKernel(kernel);
	err=clReleaseProgram(program);
	
	fflush(stdout);
	free(Ain);
	free(Aout);
	return 0;
}

