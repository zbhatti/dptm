#include "dpKernel.hpp"
#include "cmplx.h"
#ifndef __dpUux3a_H_INCLUDED__
#define __dpUux3a_H_INCLUDED__

class dpUux3a: public dpKernel{
	cl_mem eventsP_d, Amp_d;
	float *eventsP; 
	cmplx *Amp;
	int nEvents, inputBytes, outputBytes;
	
	public:
		dpUux3a(cl_context, cl_command_queue);
		void setup(int,int,int,int);
		void init(); //send workgroup dimensions after checking the kernel's type
		void memoryCopyOut();
		void plan();
		int execute();
		void memoryCopyIn();
		void cleanUp();
		void generateArray(float *P, int);
};

#endif
