#include "dpKernel.hpp"
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#ifndef __dpCudaMemory_H_INCLUDED__
#define __dpCudaMemory_H_INCLUDED__

class dpCudaMemory: public dpKernel{
	
	int Asize, blockSize, nBlocks, nKernels, lastBlock;
	float *A, *B;      //host pointers
	float *A_d, *B_d; //cuda pointers
	
	int device;
	cudaEvent_t begin, end;
	float delTime;
	struct cudaDeviceProp props;
	
	public:
		dpCudaMemory(cl_context, cl_command_queue);
		void setup(int,int,int,int);
		void init(); //send workgroup dimensions after checking the kernel's type
		void memoryCopyOut();
		void plan();
		int execute();
		void memoryCopyIn();
		void cleanUp();
		void generateArray(float*, int);
};

#endif