#include "dpKernel.hpp"

#ifndef __dpOxxxxx_H_INCLUDED__
#define __dpOxxxxx_H_INCLUDED__

class dpOxxxxx: public dpKernel{
	cl_mem Ain_d, Aout_d;
	float *Ain, *Aout;
	int Asize;
	
	public:
		dpOxxxxx(cl_context, cl_command_queue);
		void setup(int,int,int,int);
		void init(); //send workgroup dimensions after checking the kernel's type
		void memoryCopyOut();
		void plan();
		int execute();
		void memoryCopyIn();
		void cleanUp();
		void generateArray(float*, int);
};

#endif