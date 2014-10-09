#include "dpKernel.hpp"
#include <clFFT.h>

#ifndef __dpFFT_H_INCLUDED__
#define __dpFFT_H_INCLUDED__

class dpFFT: public dpKernel{

	float *Ain, *Aout;
	int Asize;
	cl_mem buffer;
	clfftPlanHandle planHandle;
	clfftSetupData fftSetup;
	size_t clLengths[1];
	clfftStatus status;
	
	public:
		dpFFT(cl_context, cl_command_queue);
		void setup(int,int,int,int);
		void init();
		void memoryCopyOut();
		void plan();
		int execute();
		void memoryCopyIn();
		void cleanUp();
		void generateInterleaved(float*,int);
};

#endif