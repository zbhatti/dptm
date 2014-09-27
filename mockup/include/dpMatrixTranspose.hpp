#include "dpKernel.hpp"
#ifndef __dpMatrixTranspose_H_INCLUDED__
#define __dpMatrixTranspose_H_INCLUDED__

class dpMatrixTranspose: public dpKernel{
	
	cl_mem Aout_d, Ain_d;
	float* Ain, *Aout;
	size_t mem_size;
	unsigned int M, N;
	
	public:
		dpMatrixTranspose(cl_context, cl_command_queue);
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