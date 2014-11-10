#include "dpKernel.hpp"
#include "cmplxCuda.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

#ifndef __dpCudaUux3a_H_INCLUDED__
#define __dpCudaUux3a_H_INCLUDED__

class dpCudaUux3a: public dpKernel{
	
	int nEvents, blockSize, nBlocks, outputBytes, inputBytes;
	float *eventsP, *eventsP_d;
	cmplx *Fo, *Fo_d;
	int device;
	cudaEvent_t begin, end;
	float delTime;
	struct cudaDeviceProp props;
	int nKernels;
	int lastBlock;
	
	public:
		dpCudaUux3a(cl_context, cl_command_queue);
		void setup(int,int,int,int);
		void init(); //send workgroup dimensions after checking the kernel's type
		void memoryCopyOut();
		void plan();
		int execute();
		void memoryCopyIn();
		void cleanUp();
		void generateArray(float *P, int);
};

#endif
