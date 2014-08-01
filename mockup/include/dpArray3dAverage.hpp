#include "dpKernel.hpp"

#ifndef __dpArray3dAverage_H_INCLUDED__
#define __dpArray3dAverage_H_INCLUDED__

class dpArray3dAverage: public dpKernel{
	cl_mem Ain_d, Aout_d;
	float *Ain, *Aout;
	int Alength;
	int nElements;
	
	public:
		dpArray3dAverage(cl_context, cl_command_queue);
		void init(int,int,int); //send workgroup dimensions after checking the kernel's type
		void memoryCopyOut();
		void plan();
		int execute();
		void memoryCopyIn();
		void cleanUp();
		void generate3dArray(float*, int);
};

#endif