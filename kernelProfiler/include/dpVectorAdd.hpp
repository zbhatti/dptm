#include "dpKernel.hpp"
#ifndef __dpVectorAdd_H_INCLUDED__
#define __dpVectorAdd_H_INCLUDED__

class dpVectorAdd: public dpKernel{

	float *srcA, *srcB, *dst;        // Host buffers for OpenCL test
	// OpenCL Vars
	cl_mem cmDevSrcA;               // OpenCL device source buffer A
	cl_mem cmDevSrcB;               // OpenCL device source buffer B 
	cl_mem cmDevDst;                // OpenCL device destination buffer
	int Asize;	// Length of float arrays to process 

	public:
		dpVectorAdd(cl_context, cl_command_queue);
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