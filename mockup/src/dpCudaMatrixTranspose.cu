#include "dpCudaMatrixTranspose.hpp"
#include "errorCheck.hpp"
#define cudaErrChk(ans) { cudaAssert((ans), __FILE__, __LINE__); }
#define BEGIN cudaEventRecord(begin, 0);
#define END cudaEventRecord(end, 0); cudaEventSynchronize(end); cudaEventElapsedTime(&delTime, begin, end);

//source:
//http://code.msdn.microsoft.com/windowsdesktop/Matrix-Transpose-on-GPU-94f0a054
__global__ void matrixTranspose(const float* m, float* t, int matrixSize){ 
    int x = blockIdx.x * blockDim.x + threadIdx.x;        // col 
    int y = blockIdx.y * blockDim.y + threadIdx.y;        // row 
 
    if (x >= matrixSize || y >= matrixSize) 
        return; 
 
    int from = x + y * matrixSize; 
    int to   = y + x * matrixSize; 
 
    t[to] = m[from];            // t(j,i) = m(i,j) 
} 

dpCudaMatrixTranspose::dpCudaMatrixTranspose(cl_context ctx, cl_command_queue q){
	workDimension = TWO_D;
	//name is same as cl alternative allowing the analysis script to later figure 
	//out this measurement was from a cuda kernel by inspecting the platform id from dpClient
	name = "MatrixTranspose";

	cudaEventCreate(&begin);
	cudaEventCreate(&end);
	
	cudaGetDevice(&device);
	cudaGetDeviceProperties(&props, device);
}

void dpCudaMatrixTranspose::setup(int dataMB, int xLocal, int yLocal, int zLocal){

	localSize[0] = localSize[1] = localSize[2] = 1;
	M=(int)sqrt(dataMB*1048576/sizeof(float));
	N=M; //not tested with rectangle matrix
	
	MB = M*N*sizeof(float)/1048576;
}

void dpCudaMatrixTranspose::init(){

	dataParameters.push_back(M);
	dataParameters.push_back(N);
	dataNames.push_back("width");
	dataNames.push_back("height");
	
	mem_size = sizeof(float) * M * N;
	// allocate and initalize host memory
	Ain = (float*)malloc(mem_size);
	Aout = (float*)malloc(mem_size);
	if(!Ain || !Aout)
		fprintf(stderr, "error in malloc\n");
	generateMatrix(Ain, M, N);

}

void dpCudaMatrixTranspose::memoryCopyOut(){
	BEGIN
	cudaErrChk(cudaMalloc((void **) &Ain_d, mem_size));
	cudaErrChk(cudaMalloc((void **) &Aout_d, mem_size));
	cudaErrChk(cudaMemcpy(Ain_d, Ain, mem_size, cudaMemcpyHostToDevice));
	END
	//printf("%0.3f,",delTime);
}

void dpCudaMatrixTranspose::plan(){
	BEGIN
	//use the largest block possible:
	blockSize.x = props.maxThreadsDim[0];
	blockSize.y = props.maxThreadsDim[1];
	if (blockSize.x*blockSize.y > props.maxThreadsPerBlock){
		blockSize.x = (int) sqrt(props.maxThreadsPerBlock);
		blockSize.y = blockSize.x;
	}
	
	//specify number of blocks in width and height of grid:
	nBlocks.x = N/blockSize.x - N%blockSize.x + blockSize.x;
	nBlocks.y = M/blockSize.y - N%blockSize.y + blockSize.y;
	END
}

int dpCudaMatrixTranspose::execute(){
	dim3 grid(nBlocks.x, nBlocks.y);
	dim3 block(blockSize.x, blockSize.y);
	cudaError_t err;
	BEGIN
	matrixTranspose <<< grid, block >>> (Ain_d, Aout_d, M);
	err = cudaPeekAtLastError() ;
	cudaErrChk(err);
	cudaErrChk(cudaDeviceSynchronize());
	END
	//printf("%0.3f,",delTime);
	if(err!=cudaSuccess)
		return -1;
	return 0;
}

void dpCudaMatrixTranspose::memoryCopyIn(){
	BEGIN
	cudaErrChk(cudaMemcpy(Aout, Aout_d, mem_size, cudaMemcpyDeviceToHost));
	END
	//printf("%0.3f,\n",delTime);
}

void dpCudaMatrixTranspose::cleanUp(){
	free(Ain);
	free(Aout);
	cudaFree(Ain_d);
	cudaFree(Aout_d);
}

void dpCudaMatrixTranspose::generateMatrix(float *A, int height, int width){
	int i, j;
	srand(time(NULL));
	for (j=0; j<height; j++){//rows in A
		for (i=0; i<width; i++){//cols in A
			A[i + width*j] = rand() / (RAND_MAX/99999.9 + 1);
		}
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