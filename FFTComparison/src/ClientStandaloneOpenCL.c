#include <stddef.h>
#include <sys/types.h>
#include <unistd.h>
#include <stdlib.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <math.h>
#include <stdio.h>
#include <sys/types.h>//shared memory
#include <sys/ipc.h>//shared memory
#include <sys/shm.h>//shared memory
#include <sys/time.h> //for random seed and timing
#include <time.h> //for random seed and timing
#include <errno.h>
#include <string.h>
#ifdef __APPLE__
	#include <OpenCL/opencl.h>
#else
	#include <CL/opencl.h>
#endif
#include <clFFT.h>
#define BEGIN gettimeofday(&start, NULL);	
#define END gettimeofday(&finish, NULL); time=timeDiff(start, finish);
#define MAXPLATFORMS 5
#define MAXDEVICES 16

void printVectorsInterleaved(float *A, int Asize){
	int i;
	for (i = 0; i < 2*Asize; i=i+2){
		printf("%1.2f+j%1.2f, ", A[i], A[i+1]);
	}
	printf("\n");
}

void generateInterleaved(float *A, int Asize){
	int i;
	srand(time(NULL));
	for (i=0; i < 2*Asize; i=i+2){
		A[i] = rand() / (RAND_MAX/99.9 + 1);
		A[i+1] = rand() / (RAND_MAX/99.9 + 1);
	}
}

float timeDiff(struct timeval start, struct timeval finish){
	return (float) ((finish.tv_sec*1000000 + finish.tv_usec) - (start.tv_sec*1000000 + start.tv_usec))/(1000);
}

int compareFunction(const void *a,const void *b) {
	int *x = (int *) a;
	int *y = (int *) b;
return *x - *y;
}

void populateSizes(int A[], int MaxVectors){
	int i, j, k, a;
	i = 0; k = 0; a = 0;
	
	int Power2 = 0;
	int Power3 = 0;
	int Power5 = 0;
	j=Power3;
	//for(j=Power3; pow(5,i)*pow(3,j)*pow(2,k) < MaxVectors; j++){
		for(k=Power2; pow(5,i)*pow(3,j)*pow(2,k) < MaxVectors; k++){
			A[a++] = pow(5,i)*pow(3,j)*pow(2,k);
		}
		k=Power2;
	//}
	A[a] = 999999999;
	qsort(A, a, sizeof(int), compareFunction);
}

//receives a sorted list of integers to return the minimum index
int findMinIndex(int A[], int MaxVectors, int MinVectors){
	for (int i = 0; i < MaxVectors; i++){
			if (A[i] > MinVectors)
				return i-1;
	}
	return 0;
}

int main(int argc, char *argv[]){
	int Asize, i,j,k, shmid, ret, staticSize, minIndex;
	int VectorSizes[1000];
	float *Ain, *Aout;
	float time, filler;
	struct timeval start, finish;
	char name[128];
	
	int DeviceChoice=0;
	int PlatformChoice=0;
	int MaxVectors=2000;
	int StepSize=100;
	int Repeat=1;
	int MinVectors=0;
	
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
	cl_command_queue queue;
	cl_context context;
	cl_device_id device_ids[MAXDEVICES];
	cl_platform_id platform_ids[MAXPLATFORMS];
	cl_mem buffer;
	unsigned int num_devices;
	cl_context_properties props[3] = { CL_CONTEXT_PLATFORM, 0, 0 };
	cl_int err;
	
	//getting information names of Platforms and Devices:
	err= clGetPlatformIDs(MAXPLATFORMS, platform_ids, NULL);
	err= clGetDeviceIDs(platform_ids[PlatformChoice], CL_DEVICE_TYPE_ALL, MAXDEVICES, device_ids, &num_devices);
	clGetPlatformInfo(platform_ids[PlatformChoice], CL_PLATFORM_NAME, sizeof(name), name, NULL);
	fprintf(stderr,"On Platform %s, ", name);
	clGetDeviceInfo(device_ids[DeviceChoice], CL_DEVICE_NAME, sizeof(name), name, NULL);
	fprintf(stderr,"using device %s\n", name);
	
	//starting up opencl on the selected device and platform:
	props[1] = (cl_context_properties) platform_ids[PlatformChoice];
	context = clCreateContext(props, 1, &device_ids[DeviceChoice], NULL, NULL, &err);
	queue = clCreateCommandQueue( context, device_ids[DeviceChoice], 0, &err);
	
	//clFFT initialization
	clfftPlanHandle planHandle;
	clfftSetupData fftSetup;
	err = clfftSetup(&fftSetup);
	size_t clLengths[1];
	
	shmid = shmget(IPC_PRIVATE, MaxVectors*2*sizeof(float), IPC_CREAT | 0666);
	if (shmid == -1)
		printf("shmid: %s\n", strerror(errno));
	
	Ain = (float*) shmat(shmid, NULL, 0);
	if (!Ain){
		printf("error in shmget\n");
		return -1;
	}

	generateInterleaved(Ain, MaxVectors);
	populateSizes(VectorSizes, MaxVectors);
	minIndex = findMinIndex(VectorSizes, MaxVectors, MinVectors);
	staticSize = 0;
	printf("nVectors,memcpyOut,plan,execu,memcpyIn\n");
	
	//main loop: loops through powers of 2 and 3 stored in the sorted VectorSizes Array
	for(int i=minIndex; (VectorSizes[i]<= MaxVectors) && (staticSize==0); i++){
		for (int r=0;r<Repeat;r++) {
			Asize = VectorSizes[i];
			if (MaxVectors == MinVectors){
				Asize = MinVectors;
				staticSize=1;
			}
			printf("%d,", Asize);
			Aout = (float*) malloc(Asize*2*sizeof(float));
			if (!Aout){
				fprintf(stderr,"error in malloc 2");
				return -1;
			}
			
			//memcpyOut
			BEGIN
			buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, Asize*2*sizeof(float), NULL, &err);
			err = clEnqueueWriteBuffer(queue, buffer, CL_TRUE, 0, Asize*2*sizeof(float), Ain, 0, NULL, NULL);
			err = clFinish(queue);
			END
			printf("%0.3f,",time);
			
			//plan
			BEGIN
			clLengths[0] = (size_t) Asize;
			err = clfftCreateDefaultPlan(&planHandle, context, CLFFT_1D, clLengths);
			err = clfftSetPlanPrecision(planHandle, CLFFT_SINGLE);
			err = clfftSetLayout(planHandle, CLFFT_COMPLEX_INTERLEAVED, CLFFT_COMPLEX_INTERLEAVED);
			err = clfftSetResultLocation(planHandle, CLFFT_INPLACE);
			err = clfftBakePlan(planHandle, 1, &queue, NULL, NULL);
			err = clFinish(queue);
			END
			printf("%0.3f,",time);
			
			//execute
			BEGIN
			err = clfftEnqueueTransform(planHandle, CLFFT_FORWARD, 1, &queue, 0, NULL, NULL, &buffer, &buffer, NULL);
			err = clFinish(queue);
			END
			printf("%0.3f,",time);
			
			//memcpyIn
			BEGIN
			err = clEnqueueReadBuffer(queue, buffer, CL_TRUE, 0, Asize*2*sizeof(float), Aout, 0, NULL, NULL);
			err = clFinish(queue);
			END
			printf("%0.3f,",time);

			clReleaseMemObject(buffer);
			free(Aout);
			err = clfftDestroyPlan(&planHandle);
			printf("\n");
		}
	}
	
	clfftTeardown();
	fflush(stdout);
	shmdt( (void*) Ain);
	shmctl(shmid, IPC_RMID, NULL);
	return 0;
}

