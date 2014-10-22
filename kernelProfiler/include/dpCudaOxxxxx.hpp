#include "dpKernel.hpp"
#include "cmplxCuda.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

#ifndef __dpCudaOxxxxx_H_INCLUDED__
#define __dpCudaOxxxxx_H_INCLUDED__

class dpCudaOxxxxx: public dpKernel{
	
	int Psize, blockSize, nBlocks;
	double *P, *P_d;
	cmplx *Fo, *Fo_d;
	int device;
	cudaEvent_t begin, end;
	float delTime;
	struct cudaDeviceProp props;
	int nKernels;
	int lastBlock;
	
	public:
		dpCudaOxxxxx(cl_context, cl_command_queue);
		void setup(int,int,int,int);
		void init(); //send workgroup dimensions after checking the kernel's type
		void memoryCopyOut();
		void plan();
		int execute();
		void memoryCopyIn();
		void cleanUp();
		void generateArray(double *P, int);
};

#endif
