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
#include <cuda.h>
#include <cuda_runtime.h>
#define BEGIN cudaEventRecord(begin, 0);
#define END cudaEventRecord(end, 0); cudaEventSynchronize(end); cudaEventElapsedTime(&time, begin, end);
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

//code from stackexchange to print cuda return messages
inline void gpuAssert(cudaError_t code, char *file, int line, bool abort=true){
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

// Kernel that executes on the CUDA device
__global__ void squareArray(float *Ain_d, float *Aout_d, int N){
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) 
		Aout_d[idx] = Ain_d[idx] * Ain_d[idx];
}

void generateArray(float *A, int N){
	int i;
	srand(time(NULL));
	for (i=0; i < N; i++){
		A[i] = rand()/(RAND_MAX/99999.9 + 1);
		fflush(stdout);
	}
}

float timeDiff(struct timeval start, struct timeval finish){
	return (float) ((finish.tv_sec*1000000 + finish.tv_usec) - (start.tv_sec*1000000 + start.tv_usec))/(1000);
}

int main(int argc, char *argv[]){
	int Asize, i,r, blockSize, nBlocks;
	float *Ain, *Aout, *Ain_d, *Aout_d;
	cudaEvent_t begin, end;
	float time;
	cudaEventCreate(&begin);
	cudaEventCreate(&end);
	cudaDeviceProp properties;
	char name[128];
	
	int MaxVectors=2000;
	int MinVectors=0;
	int StepSize=100;
	int Repeat=1;
		
	if (argc>1)
	  MaxVectors=atoi(argv[1]);
		
	if (argc>2)
	  MinVectors=atoi(argv[2]);	
		
	if (argc>3)
	  StepSize=atoi(argv[3]);

	if (argc>4)
	  Repeat=atoi(argv[4]);
	
	cudaGetDeviceProperties(&properties,0); //using device 0
	blockSize = properties.maxThreadsPerBlock;
	strcpy(name, properties.name);
	fprintf(stderr, "using %s\n", name);
	
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
			
			//memcpyOut
			BEGIN
			gpuErrchk(cudaMalloc((void **) &Ain_d, Asize*sizeof(float)));
			gpuErrchk(cudaMalloc((void **) &Aout_d, Asize*sizeof(float)));
			gpuErrchk(cudaMemcpy(Ain_d, Ain, Asize*sizeof(float), cudaMemcpyHostToDevice));
			END
			printf("%0.3f,",time);
			
			//set kernel arguments (plan)
			BEGIN
			nBlocks = Asize/blockSize - Asize%blockSize + blockSize;
			//printf("\n\t\t\t nBlocks: %d, blockSize: %d\n", nBlocks, blockSize);
			END
			printf("%0.3f,",time);
			
			//execute
			BEGIN
			squareArray <<< nBlocks, blockSize >>> (Ain_d, Aout_d, Asize);
			gpuErrchk( cudaPeekAtLastError() );
			gpuErrchk( cudaDeviceSynchronize() );
			END
			printf("%0.3f,",time);	
			
			//memcpyIn
			BEGIN
			gpuErrchk(cudaMemcpy(Aout, Aout_d, Asize*sizeof(float), cudaMemcpyDeviceToHost));
			END
			printf("%0.3f,\n",time);
			
			cudaFree(Ain_d);
			cudaFree(Aout_d);
			
			//printf("\t\tAin: %0.2f, Aout: %0.2f\n",Ain[Asize-1], Aout[Asize-1]);
		}
	}
	
	fflush(stdout);
	free(Ain);
	free(Aout);
	return 0;
}

