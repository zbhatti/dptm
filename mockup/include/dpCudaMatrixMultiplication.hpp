#include "dpKernel.hpp"
#ifndef __dpCudaMatrixMultiplication_H_INCLUDED__
#define __dpCudaMatrixMultiplication_H_INCLUDED__

class dpCudaMatrixMultiplication: public dpKernel{

	uint3 blockSize;
	uint3 nBlocks;
	int szA, szB, szC;
	int N, P, M;
	float *A, *B, *C;       //host pointers
	float *A_d, *B_d, *C_d; //cuda pointers
	
	int device;
	cudaEvent_t begin, end;
	float delTime;
	struct cudaDeviceProp props;
	
	public:
		dpCudaMatrixMultiplication(cl_context, cl_command_queue);
		void setup(int,int,int,int);
		void init();
		void memoryCopyOut();
		void plan();
		int execute();
		void memoryCopyIn();
		void cleanUp();
		void generateMatrix(float*, int, int);
};

#endif