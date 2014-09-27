#include "dpKernel.hpp"
#include <cuda.h>//cuda functions
#include <cuda_runtime.h>//cuda functions
#include <cufft.h>
#ifndef __dpCudaFFT_H_INCLUDED__
#define __dpCudaFFT_H_INCLUDED__

class dpCudaFFT: public dpKernel{

	//CUDAKernels:
	cudaEvent_t begin, end;
	float delTime;
	

	int Asize;
	cufftHandle plancufft;
	cufftComplex *Aout, *A_d, *Ain;
	
	public:
		dpCudaFFT(cl_context, cl_command_queue);
		void setup(int,int,int,int);
		void init();
		void memoryCopyOut();
		void plan();
		int execute();
		void memoryCopyIn();
		void cleanUp();
		void generate(cufftComplex*,int);
};

#endif