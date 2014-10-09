#include "dpKernel.hpp"
#ifndef __dpMemory_H_INCLUDED__
#define __dpMemory_H_INCLUDED__

class dpMemory: public dpKernel{

	float *srcA, *dst;   	// Host buffers for OpenCL test
	// OpenCL Vars
	cl_mem cmDevSrcA;   // OpenCL device source buffer A
	cl_mem cmDevDst;   // OpenCL device destination buffer
	int Asize;				// Length of float arrays to process 

	public:
		dpMemory(cl_context, cl_command_queue);
		void setup(int,int,int,int);
		void init();
		void memoryCopyOut();
		void plan();
		int execute();
		void memoryCopyIn();
		void cleanUp();
		void generateArray(float*, int);
};

#endif