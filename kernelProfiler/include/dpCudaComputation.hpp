#include "dpKernel.hpp"
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#ifndef __dpCudaComputation_H_INCLUDED__
#define __dpCudaComputation_H_INCLUDED__

class dpCudaComputation: public dpKernel{
	
	int blockSize, nBlocks;
	float a, b;
	int device;
	cudaEvent_t begin, end;
	float delTime;
	struct cudaDeviceProp props;
	
	public:
		dpCudaComputation(cl_context, cl_command_queue);
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