#include "dpKernel.hpp"
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#ifndef __dpCudaNoMemory_H_INCLUDED__
#define __dpCudaNoMemory_H_INCLUDED__

class dpCudaNoMemory: public dpKernel{
	
	int nBlocks,blockSize;
	int device;
	cudaEvent_t begin, end;
	float delTime;
	struct cudaDeviceProp props;
	
	public:
		dpCudaNoMemory(cl_context, cl_command_queue);
		void setup(int,int,int,int);
		void init(); //send workgroup dimensions after checking the kernel's type
		void memoryCopyOut();
		void plan();
		int execute();
		void memoryCopyIn();
		void cleanUp();
};

#endif
