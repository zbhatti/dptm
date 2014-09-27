#include "dpKernel.hpp"
#ifndef __dpCudaMatrixTranspose_H_INCLUDED__
#define __dpCudaMatrixTranspose_H_INCLUDED__

class dpCudaMatrixTranspose: public dpKernel{
	
	float* Aout_d, *Ain_d;
	float* Ain, *Aout;
	size_t mem_size;
	unsigned int M, N;
	uint3 blockSize;
	uint3 nBlocks;
	
	int device;
	cudaEvent_t begin, end;
	float delTime;
	struct cudaDeviceProp props;
	
	
	public:
		dpCudaMatrixTranspose(cl_context, cl_command_queue);
		void setup(int,int,int,int);
		void init();
		void memoryCopyOut();
		void plan();
		int execute();
		void memoryCopyIn();
		void cleanUp();
		void generateMatrix(float*,int,int);
};

#endif