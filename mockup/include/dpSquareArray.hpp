#include "dpKernel.hpp"

#ifndef __dpSquareArray_H_INCLUDED__
#define __dpSquareArray_H_INCLUDED__

class dpSquareArray: public dpKernel{
	cl_mem Ain_d, Aout_d;
	float *Ain, *Aout;
	int Asize;
	
	public:
		dpSquareArray(cl_context, cl_command_queue);
		void init(int,int,int); //send workgroup dimensions after checking the kernel's type
		void memoryCopyOut();
		void plan();
		int execute();
		void memoryCopyIn();
		void cleanUp();
		void generateArray(float*, int);
};

#endif