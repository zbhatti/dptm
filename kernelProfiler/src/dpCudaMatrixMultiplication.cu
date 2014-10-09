#include "dpCudaMatrixMultiplication.hpp"
#include "errorCheck.hpp"
#define cudaErrChk(ans) { cudaAssert((ans), __FILE__, __LINE__); }
#define BEGIN cudaEventRecord(begin, 0);
#define END cudaEventRecord(end, 0); cudaEventSynchronize(end); cudaEventElapsedTime(&delTime, begin, end);

// CUDA Kernel from:
//http://gpgpu-computing4.blogspot.com/2009/08/matrix-multiplication-1.html
__global__ void matrixMultiplication(int Mdim, int Ndim, int Pdim, float *A_d, float *B_d, float *C_d){
		int k;
		int i = blockIdx.x * blockDim.x + threadIdx.x; //get column
		int j = blockIdx.y * blockDim.y + threadIdx.y; //get row
		
		float tmp = 0.0f;
		if (i >= Mdim || j >= Ndim) 
      return; 
		
		for (k=0; k<Pdim; k++) 
			tmp += A_d[j*Pdim+k] * B_d[k*Mdim+i];
		C_d[j*Mdim+i] = tmp;
}

// CUDA Kernel from (unused):
//http://gpgpu-computing4.blogspot.com/2009/08/matrix-multiplication-2.html
__global__ void matrixMul(float* A, float* B,  float* C, int wA, int wB){
	int TILE_SIZE = 16;
	// 1. 2D Thread ID
	int tx = blockIdx.x * TILE_SIZE + threadIdx.x;
	int ty = blockIdx.y * TILE_SIZE + threadIdx.y;

	// value stores the element that is 
	// computed by the thread
	float value = 0;
	for (int i = 0; i < wA; ++i){
		float elementA = A[ty * wA + i];
		float elementB = B[i * wB + tx];
		value += elementA * elementB;
	}

	// Write the matrix to device memory each 
	// thread writes one element
	C[ty * wA + tx] = value;
}


dpCudaMatrixMultiplication::dpCudaMatrixMultiplication(cl_context ctx, cl_command_queue q){
	workDimension = TWO_D;
	//name is same as cl alternative allowing the analysis script to later figure 
	//out this measurement was from a cuda kernel by inspecting the platform id from dpClient
	name = "MatrixMultiplication";

	cudaEventCreate(&begin);
	cudaEventCreate(&end);
	
	cudaGetDevice(&device);
	cudaGetDeviceProperties(&props, device);
	
}

void dpCudaMatrixMultiplication::setup(int dataMB, int xLocal, int yLocal, int zLocal){
	localSize[0] = localSize[1] = localSize[2] = 1;
	
	for (int i =0; pow(2,i)*pow(2,i)*sizeof(float)/(float) 1048576 <= dataMB;i++){
		N = pow(2,i);
		P = pow(2,i);
		M = pow(2,i);
	}
	
	N=(int)sqrt(1048576*dataMB/sizeof(float));
	P=N;
	M=N;
	
	//calculating total data as MB of matrix A only. MBof(B) <= MBof(A)
	MB=(N*P*sizeof(float))/1048576;
	
}

void dpCudaMatrixMultiplication::init(){
	szA = N*P;
	szB = P*M;
	szC = N*M;
	
	dataParameters.push_back(N);
	dataParameters.push_back(P);
	dataParameters.push_back(M);
	
	dataNames.push_back("N");
	dataNames.push_back("P");
	dataNames.push_back("M");
	
	
	A = (float*) malloc(szA*sizeof(float));
	B = (float*) malloc(szB*sizeof(float));
	C = (float*) malloc(szC*sizeof(float));
	
	if (!A || !B || !C)
		fprintf(stderr,"error in malloc\n");
	generateMatrix(A,N,P);
	generateMatrix(B,P,M);

}

void dpCudaMatrixMultiplication::memoryCopyOut(){
	BEGIN
	cudaErrChk(cudaMalloc((void **) &A_d, szA*sizeof(float)));
	cudaErrChk(cudaMalloc((void **) &B_d, szB*sizeof(float)));
	cudaErrChk(cudaMalloc((void **) &C_d, szC*sizeof(float)));
	cudaErrChk(cudaMemcpy(A_d, A, szA*sizeof(float), cudaMemcpyHostToDevice));
	cudaErrChk(cudaMemcpy(B_d, B, szB*sizeof(float), cudaMemcpyHostToDevice));
	END
	//printf("%0.3f,",delTime);
}

void dpCudaMatrixMultiplication::plan(){
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

int dpCudaMatrixMultiplication::execute(){
	//printf("Kernel SAYS: %s\n", props.name);
	dim3 grid(nBlocks.x, nBlocks.y);
	dim3 block(blockSize.x, blockSize.y);
	cudaError_t err;
	BEGIN
	matrixMultiplication <<< grid, block >>> (M, N, P, A_d, B_d, C_d);
	err = cudaPeekAtLastError() ;
	cudaErrChk(err);
	cudaErrChk(cudaDeviceSynchronize());
	END
	//printf("%0.3f,",delTime);
	if(err!=cudaSuccess)
		return -1;
	return 0;
}

void dpCudaMatrixMultiplication::memoryCopyIn(){
	BEGIN
	cudaErrChk(cudaMemcpy(C, C_d, szC*sizeof(float), cudaMemcpyDeviceToHost));
	END
	//printf("%0.3f,\n",delTime);
}

void dpCudaMatrixMultiplication::cleanUp(){
	// Cleanup and leave
	cudaFree(A_d);
	cudaFree(B_d);
	cudaFree(C_d);
	free(A);
	free(B);
	free(C);
}

void dpCudaMatrixMultiplication::generateMatrix(float *A, int height, int width){
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