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
#include <time.h>
#include <cuda.h>//cuda functions
#include <cuda_runtime.h>//cuda functions
#include <cufft.h>
#include <errno.h>
#define __float128 long double
#define BEGIN cudaEventRecord(begin, 0);
#define END cudaEventRecord(end, 0); cudaEventSynchronize(end); cudaEventElapsedTime(&time, begin, end);
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

//code from stackexchange to print cuda return messages
void gpuAssert(cudaError_t code, char *file, int line){
	int abort = 0;
   if (code != cudaSuccess){
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) 
				exit(code);
   }
}

void printCufft(cufftComplex *A, int Asize){
	int i;
	for (i = 0; i < Asize; i++){
		printf("%1.2f+j%1.2f, ", A[i].x, A[i].y);
	}
	printf("\n");
}

void generate(cufftComplex* A, int Asize){
	int i;
	srand(time(NULL));
	for (i = 0; i < Asize; i++){
		A[i].x = rand() / (RAND_MAX/99.9 + 1);
		A[i].y = rand() / (RAND_MAX/99.9 + 1);
	}
}

int main(int argc, char *argv[]){
	int Asize, i, shmid, ret;
	
	cudaEvent_t begin, end;
	float time;
	cudaEventCreate(&begin);
	cudaEventCreate(&end);
	
	cufftComplex *A_h, *A_d, *Ain;
	cufftHandle plancufft;

	int num_devices;
	cudaGetDeviceCount(&num_devices);
	
	fprintf(stderr,"Number of devices found: %d\n",num_devices);
	if (num_devices > 1) {

	  int max_multiprocessors = 0, max_device = 0, device;
	  for (device = 0; device < num_devices; device++) {
	    cudaDeviceProp properties;
	    cudaGetDeviceProperties(&properties, device);
	    
	    fprintf(stderr,"Device %d has %d multiProcessors. \n",device, properties.multiProcessorCount);

	    if (max_multiprocessors < properties.multiProcessorCount) {
	      max_multiprocessors = properties.multiProcessorCount;
	      max_device = device;
	    }
	  }

	  //optionally select GPU
	  if (argc>1)
	    max_device=atoi(argv[1]);

	  fprintf(stderr,"Selected Device %d.\n",max_device);
	  cudaSetDevice(max_device);
		cudaDeviceProp prop1;
		cudaGetDeviceProperties(&prop1, max_device);
		printf("%s\n", prop1.name);
	}
	
	
	int MaxVectors=2000;
	int MinVectors=0;
	int StepSize=100;
	int Repeat=1;
		
	if (argc>2)
	  MaxVectors=atoi(argv[2]);

	if (argc>3)
	  MinVectors=atoi(argv[3]);

	if (argc>4)
	  StepSize=atoi(argv[4]);

	if (argc>5)
	  Repeat=atoi(argv[5]);
	
	shmid = shmget(IPC_PRIVATE, MaxVectors * sizeof(cufftComplex), IPC_CREAT | 0666);
	if (shmid == -1)
		printf("shmid: %s\n", strerror(errno));
	
	Ain = (cufftComplex*) shmat(shmid, NULL, 0);
	if (!Ain){
		printf("error in shmget\n");
		return -1;
	}
	 
	generate(Ain, MaxVectors);
	printf("nVectors,memcpyOut,plan,execu,memcpyIn\n");
	
	//main loop
	for (int r=0;r<Repeat;r++) {
		for (i=MinVectors; i <= MaxVectors; i+=StepSize){
		//for (i=MinVectors; pow(2,i) <= MaxVectors; i++){
			Asize = i;
			//Asize = pow(2,i);
			if (i == 0)
				Asize = 2;
			printf("%d,", Asize);

			A_h = (cufftComplex*) malloc(Asize * sizeof(cufftComplex));
			if (!A_h){
			  fprintf(stderr,"error in malloc 2");
			  return -1;
			}
			
			BEGIN
			gpuErrchk(cudaMalloc((void**)&A_d, Asize * sizeof(cufftComplex)));
			gpuErrchk(cudaMemcpy(A_d, Ain, Asize * sizeof(cufftComplex), cudaMemcpyHostToDevice));
			END
			printf("%0.3f,", time);
			
			BEGIN
			ret = cufftPlan1d(&plancufft, Asize, CUFFT_C2C, 1);
			if (ret != 0){
			  printf("");
			  fprintf(stderr, "%s %d", "cufftplan1d fail err: ", ret);
				return -1;
			}
			END
			printf("%0.3f,", time);
			
			BEGIN
			ret = cufftExecC2C(plancufft, A_d, A_d, CUFFT_FORWARD);
			if (ret != 0){
			  fprintf(stderr, "%s %d", "cufftexecc2c fail err: ", ret);
			  return -1;
			}
			END
			printf("%0.3f,", time);
			
			BEGIN
			gpuErrchk(cudaMemcpy(A_h, A_d, Asize * sizeof(cufftComplex), cudaMemcpyDeviceToHost));	
			END
			printf("%0.3f,\n", time);
			
			fflush(stdout);
			gpuErrchk(cudaFree(A_d));
			cufftDestroy(plancufft); 
			free(A_h);
		}
	}
	shmdt( (void*) Ain);
	shmctl(shmid, IPC_RMID, NULL);
	return 0;
}

