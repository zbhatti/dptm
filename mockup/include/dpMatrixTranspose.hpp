#include "dpKernel.hpp"
#ifndef __dpMatrixTranspose_H_INCLUDED__
#define __dpMatrixTranspose_H_INCLUDED__

class dpMatrixTranspose: public dpKernel{
	
	cl_mem d_odata, d_idata;
	float* h_idata, *h_odata;
	size_t mem_size;
	unsigned int size_x, size_y;
	cl_int err;
	const char* kernelString;
	
	
	public:
		dpMatrixTranspose(cl_context, cl_command_queue);
		void init(int,int,int);
		void memoryCopyOut();
		void plan();
		void execute();
		void memoryCopyIn();
		void cleanUp();
		void generateMatrix(float*,int,int);
};

#endif