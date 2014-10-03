#include "dpCudaSquareArray.hpp"
#include "errorCheck.hpp"
#define cudaErrChk(ans) { cudaAssert((ans), __FILE__, __LINE__); }
#define BEGIN cudaEventRecord(begin, 0);
#define END cudaEventRecord(end, 0); cudaEventSynchronize(end); cudaEventElapsedTime(&delTime, begin, end);

// Kernel that executes on the CUDA device
__global__ void squareArray(float *Ain_d, float *Aout_d, int N){
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) 
		Aout_d[idx] = Ain_d[idx] * Ain_d[idx];
}

//notice unused parameters for CUDA kernel:
dpCudaSquareArray::dpCudaSquareArray(cl_context ctx, cl_command_queue q){

	workDimension = ONE_D;
	//name is same as cl alternative allowing the analysis script to later figure 
	//out this measurement was from a cuda kernel by inspecting the platform id from dpClient
	name = "SquareArray";

	cudaEventCreate(&begin);
	cudaEventCreate(&end);
	
	cudaGetDevice(&device);
	cudaGetDeviceProperties(&props, device);
}

void dpCudaSquareArray::setup(int dataMB, int xLocal, int yLocal, int zLocal){

	localSize[0] = localSize[1] = localSize[2] = 1;
	
	Asize = 1048576*dataMB/sizeof(float);
	MB = Asize * sizeof(float) / 1048576;
	
}

void dpCudaSquareArray::init(){
	//allocate local memory for original array
	Ain = (float*) malloc(Asize*sizeof(float));
	Aout = (float*) malloc(Asize*sizeof(float));
	if (!Aout || !Ain)
		fprintf(stderr,"error in malloc\n");
	
	generateArray(Ain, Asize);
	dataParameters.push_back(Asize);
	dataNames.push_back("nElements");

}

void dpCudaSquareArray::memoryCopyOut(){
	BEGIN
	cudaErrChk( cudaMalloc((void **) &Ain_d, Asize*sizeof(float) ));
	cudaErrChk( cudaMalloc((void **) &Aout_d, Asize*sizeof(float) ));
	cudaErrChk( cudaMemcpy(Ain_d, Ain, Asize*sizeof(float), cudaMemcpyHostToDevice) );
	END
	//printf("%0.3f,",delTime);
}

void dpCudaSquareArray::plan(){
	BEGIN
	blockSize = props.maxThreadsPerBlock;
	nBlocks = Asize/blockSize; //nblocks = ceil(Asize/blockSize)
		if (Asize%blockSize != 0)
			nBlocks++;
		nKernels=1;
    lastBlock=0;
		
		if(nBlocks>65535) {
			while(nBlocks>65535) {
				nBlocks-=65535;
				nKernels++;
			}
			if (nBlocks!=0){
				lastBlock=nBlocks;
				nKernels++;
			}
			nBlocks = 65535;
		}
	END
	
}

int dpCudaSquareArray::execute(){
	//printf("Asize: %d, nBlocks: %d, blockSize: %d, nKernels: %d\n",Asize, nBlocks, blockSize, nKernels);	
	cudaError_t err;
	BEGIN
	int NB=nBlocks;
	for (int run=0;run<nKernels;run++) {
		if (lastBlock!=0 && run == nKernels-1)
			NB = lastBlock; 
		squareArray <<< NB, blockSize >>> (Ain_d, Aout_d, Asize);
	}
	err = cudaPeekAtLastError() ;
	cudaErrChk(err);
	cudaErrChk(cudaDeviceSynchronize());
	END
	//printf("%0.3f,",delTime);
	if(err!=cudaSuccess)
		return -1;
	return 0;
}

void dpCudaSquareArray::memoryCopyIn(){
	BEGIN
	cudaErrChk(cudaMemcpy(Aout, Aout_d, Asize*sizeof(float), cudaMemcpyDeviceToHost));
	END
	//printf("%0.3f,\n",delTime);
}

void dpCudaSquareArray::cleanUp(){
	cudaFree(Ain_d);
	cudaFree(Aout_d);
	free(Ain);
	free(Aout);
}

void dpCudaSquareArray::generateArray(float *A, int N){
	int i;
	srand(time(NULL));
	for (i=0; i < N; i++){
		A[i]=rand() / (RAND_MAX/99999.9 + 1);
	}
}


/*
#include <stddef.h>
#include <sys/types.h>
#include <unistd.h>
#include <stdlib.h>
#include <sys/un.h>
#include <math.h>
#include <errno.h>
#include <string.h>


*/


